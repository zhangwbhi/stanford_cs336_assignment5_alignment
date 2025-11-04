import torch
from torch.utils.data import DataLoader, Dataset
import typer
import re
import random
import json
from tqdm import tqdm
import wandb
import pathlib
from functools import partial
from typing import Callable, List, Tuple, Dict, Any, Optional
from unittest.mock import patch
from argparse import ArgumentParser
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams, RequestOutput
import numpy as np
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import tokenize_prompt_and_output, get_response_log_probs, masked_normalize, log_generations, sft_microbatch_train_step
from cs336_alignment.math_baseline import run_vllm, extract_reference_answer, format_prompt_with_template


MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
# This is the full dataset of questions (D)
MATH_DATA_PATH = "./data/gsm8k/train.jsonl"
VALID_DATA_PATH = "./data/gsm8k/test.jsonl" # Use gsm8k test for validation
PROMPT_PATH = "./cs336_alignment/prompts/r1_zero.prompt"

VLLM_DEVICE = "cuda:0"
POLICY_DEVICE = "cuda:1"


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copies the policy model's state dict into the vLLM instance.
    """
    print("Loading policy weights into vLLM instance...")
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("Finished loading weights.")


class SFTDataset(Dataset):
    """A simple dataset for loading JSONL files."""
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(
        batch: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        prompt_template: str
) -> Dict[str, torch.Tensor]:
    # In EI, the 'question' key holds the original question text
    # and the 'answer' key holds the *generated* correct response.
    prompts = [
        prompt_template.format(question=item["question"]) for item in batch
    ]
    # The generated answer is already in the correct format,
    # but we re-apply the template just in case and to be consistent.
    responses = [item["answer"].replace("\n####", " </think> <answer>") + " </answer>" for item in batch]

    tokenized_batch = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer
    )

    return tokenized_batch


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
):
    # Note: run_vllm is fine for eval as it only ever needs n=1
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)

    stats = dict()
    total_responses = len(responses)
    if total_responses == 0:
        return {"eval/correct": 0, "eval/wrong": 0}

    for response, answer in zip(responses, answers):
        ground_truth = extract_reference_answer(answer)
        metrics = reward_fn(response, ground_truth)

        if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
            stats["eval/correct"] = stats.get("eval/correct", 0) + 1
        elif metrics["format_reward"] == 1 and metrics["answer_reward"] == 0:
            stats["eval/format_correct_answer_wrong"] = stats.get("eval/format_correct_answer_wrong", 0) + 1
        elif metrics["format_reward"] == 0 and metrics["answer_reward"] == 1:
            stats["eval/format_wrong_answer_correct"] = stats.get("eval/format_wrong_answer_correct", 0) + 1
        else:
            stats["eval/wrong"] = stats.get("eval/wrong", 0) + 1


    return stats


def main(
    learning_rate: float = typer.Option(2e-5, help="Learning rate for AdamW."),
    sft_microbatch_size: int = typer.Option(8, help="SFT microbatch size."),
    gradient_accumulation_steps: int = typer.Option(8, help="Number of gradient accumulation steps."),
    n_sft_epochs: int = typer.Option(1, help="Number of SFT epochs *per expert iteration step*."),
    n_ei_steps: int = typer.Option(5, help="Number of expert iteration steps."),
    ei_batch_size: int = typer.Option(1024, help="Number of questions to sample per EI step (Db)."),
    G: int = typer.Option(8, help="Number of rollouts (generations) per question."),
    sampling_temperature: float = typer.Option(1.0, help="Temperature for vLLM sampling."),
    sampling_max_tokens: int = typer.Option(1024, help="Max tokens for vLLM sampling."),
    math_data_path: str = typer.Option(MATH_DATA_PATH, help="Path to MATH training data."),
    eval_every_n_steps: int = typer.Option(100, help="Run evaluation every N global steps."),
    log_every_n_steps: int = typer.Option(10, help="Log training metrics every N global steps."),
    output_dir: str = typer.Option("models/ei_qwen", help="Directory to save the final model."),
    wandb_project: str = typer.Option("cs336-a5-ei", help="WandB project name."),
    seed: int = typer.Option(42, help="Random seed."),
):
    if torch.cuda.device_count() < 2:
        print("Error: this script requires at least 2 GPUs.")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    effective_batch_size = sft_microbatch_size * gradient_accumulation_steps


    wandb.init(
        project=wandb_project,
        config={
            "learning_rate": learning_rate,
            "sft_microbatch_size": sft_microbatch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "n_sft_epochs": n_sft_epochs,
            "n_ei_steps": n_ei_steps,
            "ei_batch_size": ei_batch_size,
            "G": G,
            "sampling_temperature": sampling_temperature,
            "model_path": MODEL_PATH,
        },
    )

    wandb.define_metric("train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("ei/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("ei/*", step_metric="ei/step")

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=POLICY_DEVICE
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    amp_ctx = torch.amp.autocast(device_type=POLICY_DEVICE, dtype=torch.bfloat16)

    vllm = init_vllm(MODEL_PATH, VLLM_DEVICE, seed)

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Load the full MATH dataset (D)
    print(f"Loading MATH dataset from {math_data_path}...")
    with open(math_data_path, "r", encoding="utf-8") as f:
        all_math_data = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(all_math_data)} total questions.")


    with open(VALID_DATA_PATH, "r", encoding="utf-8") as f:
        validation_data = [json.loads(line.strip()) for line in f]

    validation_data_prompts = [
        format_prompt_with_template(example["question"], prompt_template) for example in validation_data
    ]
    validation_data_answers = [
        example["answer"] for example in validation_data
    ]
    log_generation_examples = validation_data[:32]

    collate_fn_with_args = partial(
        collate_fn,
        tokenizer=tokenizer,
        prompt_template=prompt_template
    )

    # Sampling params for evaluation (n=1)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Sampling params for generation (n=G)
    sampling_min_tokens = 4
    generation_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0, # Use top_p=1.0 for diverse sampling
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens, # Per instructions
        n=G, # Generate G rollouts
        seed=seed,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )


    print(f"Starting Expert Iteration for {n_ei_steps} steps...")
    print(f"  Question batch size (Db): {ei_batch_size}")
    print(f"  Rollouts per question (G): {G}")
    print(f"  SFT epochs per step: {n_sft_epochs}")
    print(f"  SFT effective batch size: {effective_batch_size}")

    global_step = 0 # Tracks total optimizer updates

    # --- Outer Expert Iteration Loop ---
    for ei_step in range(n_ei_steps):
        print(f"\n--- Expert Iteration Step {ei_step + 1}/{n_ei_steps} ---")

        # --- Step 3: Sample a batch of questions Db from D ---
        print(f"Sampling {ei_batch_size} questions from MATH dataset...")
        question_batch_db = random.sample(all_math_data, ei_batch_size)
        generation_prompts = [
            format_prompt_with_template(item["question"], prompt_template) for item in question_batch_db
        ]

        # --- Step 4: Set the old policy model πθold ← πθ ---
        # Sync the policy model (πθ) to the vLLM instance (πθold) for generation
        print("Syncing policy to vLLM for generation...")
        load_policy_into_vllm_instance(model, vllm)

        # --- Step 5: Sample G outputs {o(i)} ∼ πθold ---
        print(f"Generating {G} rollouts for {len(generation_prompts)} prompts...")
        # We call vllm.generate directly to get back all G outputs
        # This returns List[RequestOutput]
        vllm_outputs: List[RequestOutput] = vllm.generate(generation_prompts, generation_sampling_params)

        # --- Steps 6 & 7: Compute rewards and filter wrong outputs ---
        print("Filtering generated responses...")
        sft_data_for_step = [] # This will be Dsft
        total_generations = 0

        for item, vllm_output in zip(question_batch_db, vllm_outputs):
            question_text = item["question"]
            ground_truth = extract_reference_answer(item["answer"])

            # vllm_output.outputs is a list of G CompletionOutput objects
            for gen in vllm_output.outputs:
                total_generations += 1
                response_text = gen.text
                metrics = r1_zero_reward_fn(response_text, ground_truth)

                # Check for *correct* generations
                if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
                    sft_data_for_step.append({
                        "question": question_text,
                        "answer": response_text # The *generated* correct answer
                    })

        num_correct = len(sft_data_for_step)
        print(f"Generated {total_generations} responses, found {num_correct} correct examples.")

        wandb.log({
            "ei/step": ei_step,
            "ei/num_questions_sampled": ei_batch_size,
            "ei/num_generations": total_generations,
            "ei/num_correct_generated": num_correct
        })

        if num_correct == 0:
            print("Warning: No correct examples generated in this EI step. Skipping SFT.")
            continue # Go to the next EI step

        # --- Step 8: πθ ← SFT(πθ, Dsft) ---
        print(f"Starting SFT on {num_correct} generated examples for {n_sft_epochs} epochs...")

        sft_dataset = SFTDataset(sft_data_for_step)
        sft_dataloader = DataLoader(
            sft_dataset,
            batch_size=sft_microbatch_size,
            shuffle=True,
            collate_fn=collate_fn_with_args,
            num_workers=4,
        )

        num_batches_per_epoch = len(sft_dataloader)
        print(f"  SFT inner loop: {num_batches_per_epoch} microbatches per epoch.")

        # --- Inner SFT Loop ---
        for epoch in range(n_sft_epochs):
            print(f"  ---SFT Epoch {epoch + 1}/{n_sft_epochs} (EI Step {ei_step + 1})---")

            for batch_idx, batch in enumerate(tqdm(sft_dataloader, desc="SFT Step")):
                input_ids = batch["input_ids"].to(POLICY_DEVICE)
                labels = batch["labels"].to(POLICY_DEVICE)
                response_mask = batch["response_mask"].to(POLICY_DEVICE)

                with amp_ctx:
                    log_probs_data = get_response_log_probs(
                        model,
                        input_ids,
                        labels,
                        return_token_entropy=True # For logging
                    )
                    policy_log_probs = log_probs_data["log_probs"]

                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps
                )

                # --- Logging (Inner Step) ---
                if (batch_idx + 1) % log_every_n_steps == 0:
                    # Calculate average entropy for logging
                    token_entropy = log_probs_data["token_entropy"]
                    masked_entropy = token_entropy * response_mask
                    avg_entropy = masked_entropy.sum() / response_mask.sum().add(1e-8)

                    log_data = {
                        "train/step": global_step, # Use global_step
                        "train/loss_microbatch": loss.item(),
                        "train/unscaled_loss": metadata["unscaled_loss"].item(),
                        "train/avg_token_entropy": avg_entropy.item(), # Entropy logging
                        "train/ei_step": ei_step
                    }
                    wandb.log(log_data)

                # --- Optimizer Step (Inner Step) ---
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Grad clipping
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1 # A global step is one optimizer update

                    # --- Evaluation (Inner Step) ---
                    if (global_step % eval_every_n_steps == 0) and (global_step > 0):
                        print(f"Running evaluation at global step {global_step} (EI Step {ei_step + 1})...")

                        # 1. Sync policy model weights to vLLM model
                        load_policy_into_vllm_instance(model, vllm)

                        # 2. Run quantitative evaluation
                        stats = evaluate_vllm(
                            vllm, r1_zero_reward_fn, validation_data_prompts, validation_data_answers, eval_sampling_params
                        )
                        accuracy = stats["eval/correct"] / len(validation_data_prompts)

                        # 3. Run qualitative logging
                        log_gen_metrics = log_generations(
                            policy_model=model,
                            vllm_model=vllm,
                            tokenizer=tokenizer,
                            reward_fn=r1_zero_reward_fn,
                            prompt_template=prompt_template,
                            examples=log_generation_examples,
                            sampling_params=eval_sampling_params
                        )

                        wandb_log_data = {
                            "eval/step": global_step,
                            "eval/accuracy": accuracy,
                            **stats,
                            "eval/avg_response_length": log_gen_metrics["avg_response_length"],
                            "eval/avg_correct_response_length": log_gen_metrics["avg_correct_response_length"],
                            "eval/avg_incorrect_response_length": log_gen_metrics["avg_incorrect_response_length"],
                            "eval/avg_token_entropy": log_gen_metrics["avg_token_entropy"], # Entropy logging
                            "eval/ei_step": ei_step
                        }
                        wandb.log(wandb_log_data)
        # --- End of Inner SFT Loop ---
    # --- End of Outer EI Loop ---

    print("Expert Iteration complete.")

    print("Running final evaluation...")
    load_policy_into_vllm_instance(model, vllm)
    stats = evaluate_vllm(
        vllm, r1_zero_reward_fn, validation_data_prompts, validation_data_answers, eval_sampling_params
    )
    accuracy = stats["eval/correct"] / len(validation_data_prompts)
    print(f"Final Accuracy: {accuracy:.4f}")
    wandb.log({"eval/final_accuracy": accuracy})

    # --- Save Model ---
    print(f"Saving model to {output_dir}...")
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    print("Model saved.")

    wandb.finish()

if __name__ == "__main__":
    typer.run(main)