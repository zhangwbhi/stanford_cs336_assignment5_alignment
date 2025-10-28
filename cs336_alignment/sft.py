import json
import math
import pathlib
import random
from functools import partial
from typing import List, Dict, Any, Tuple, Callable, Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.optim as optim
import typer
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
    parse_ground_truth
)

MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
SFT_DATA_PATH = "../data/gsm8k/train.jsonl"
VALID_DATA_PATH = "../data/gsm8k/test.jsonl"
PROMPT_PATH = "./prompts/r1_zero.prompt"

VLLM_DEVICE = "cuda: 0"
POLICY_DEVICE = "cuda: 1"


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy. [cite: 259-275]
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
    Copies the policy model's state dict into the vLLM instance. [cite: 276-280]
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
    prompts = [
        prompt_template.format(question=item["question"]) for item in batch
    ]
    responses = [item["answer"] for item in batch]

    tokenized_batch = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer
    )

    return tokenized_batch


def evaluate(
        vllm_model: LLM,
        reward_fn: Callable,
        prompt_template: str,
        validation_examples: List[Dict[str, Any]],
        eval_sampling_params: SamplingParams,
        eval_batch_size: int = 32,
) -> Dict[str, float]:
    print(f"Running evaluation on {len(validation_examples)} examples...")

    all_answer_rewards = []
    all_format_rewards = []

    # split inference into batchs to avoid OOM
    for i in tqdm(range(0, len(validation_examples), eval_batch_size)):
        batch_examples = validation_examples[i : i + eval_batch_size]

        prompts = [
            prompt_template.fomat(question=ex["question"]) for ex in batch_examples
        ]
        ground_truths = [
            parse_ground_truth(ex["answer"]) for ex in batch_examples
        ]

        outputs = vllm_model.generate(prompts, eval_sampling_params)
        generated_responses = [out.output[0].text.strip() for out in outputs]

        for j in range(len(generated_responses)):
            reward_data = reward_fn(generated_responses[j], ground_truths[j])
            all_answer_rewards.append(reward_data.get("answer_reward", 0.0))
            all_format_rewards.append(reward_data.get("format_reward", 0.0))

    accuracy = np.mean(all_answer_rewards)
    format_reward = np.mean(all_format_rewards)

    print(f"Evaluation complete. Accuracy: {accuracy:.4f}")

    return {
        "eval/accuracy": accuracy,
        "eval/format_reward": format_reward,
    }


def main(
    learning_rate: float = typer.Option(5e-6, help="Learning rate for AdamW."),
    batch_size: int = typer.Option(16, help="Microbatch size."),
    gradient_accumulation_steps: int = typer.Option(8, help="Number of gradient accumulation steps."),
    num_train_epochs: int = typer.Option(1, help="Number of training epochs."),
    n_sft_examples: Optional[int] = typer.Option(None, help="Number of SFT examples to use. None for all."),
    eval_every_n_steps: int = typer.Option(100, help="Run evaluation every N global steps."),
    log_every_n_steps: int = typer.Option(10, help="Log training metrics every N global steps."),
    eval_batch_size: int = typer.Option(32, help="Batch size for evaluation."),
    eval_log_examples: int = typer.Option(16, help="Number of examples for qualitative logging."),
    output_dir: str = typer.Option("models/sft_qwen", help="Directory to save the final model."),
    wandb_project: str = typer.Option("cs336-a5-sft", help="WandB project name."),
    device: str = typer.Option("cpu", help="Device type for expriment."),
    seed: int = typer.Option(42, help="Random seed."),
):

    global POLICY_DEVICE, VLLM_DEVICE

    if device == "cpu":
        POLICY_DEVICE = "cpu"
        VLLM_DEVICE = "cuda"

    if device == "cuda" and torch.cuda.device_count() < 2:
        print("Error: this script requires at least 2 GPUs.")
        return



    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    effective_batch_size = batch_size * gradient_accumulation_steps

    wandb.init(
        project=wandb_project,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_train_epochs": num_train_epochs,
            "n_sft_examples": n_sft_examples,
            "model_path": MODEL_PATH,
        },
    )

    wandb.define_metric("train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    print("Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device=="cuda" else None,
    ).to(POLICY_DEVICE)

    policy_model.train()

    vllm_model = init_vllm(MODEL_PATH, VLLM_DEVICE, seed)

    print("Loading data...")
    prompt_template = pathlib.Path(PROMPT_PATH).read_text()

    with open(SFT_DATA_PATH, "r") as f:
        sft_data = [json.loads(line) for line in f]

    if n_sft_examples:
        sft_data = sft_data[:n_sft_examples]

    with open(VALID_DATA_PATH, "r") as f:
        validation_data = [json.loads(line) for line in f]

    train_dataset = SFTDataset(sft_data)

    collate_fn_with_args = partial(
        collate_fn,
        tokenizer=tokenizer,
        prompt_template=prompt_template
    )


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_args,
        num_workers=4,
    )

    optimizer = optim.AdamW(policy_model.parameters(), lr=learning_rate)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    log_generation_examples = validation_data[:eval_log_examples]

    print(f"Starting SFT for {num_train_epochs} epochs...")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Training on {len(train_dataset)} examples.")

    global_step = 0
    num_batches_per_epoch = len(train_dataloader)
    total_steps = (num_batches_per_epoch // gradient_accumulation_steps) * num_train_epochs

    for epoch in range(num_train_epochs):
        print(f"---Epoch {epoch + 1}/{num_train_epochs}---")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            response_mask = batch["response_mask"]

            log_probs_data = get_response_log_probs(
                policy_model,
                input_ids,
                labels,
                return_token_entropy=True # For logging
            )

            policy_log_probs = log_probs_data["log_probs"]

            # --- Backward Pass (sft_microbatch_train_step calls .backward()) ---
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            if (batch_idx + 1) % log_every_n_steps == 0:
                # Calculate average entropy for logging
                token_entropy = log_probs_data["token_entropy"]
                masked_entropy = token_entropy * response_mask
                avg_entropy = masked_entropy.sum() / response_mask.sum().add(1e-8)

                log_data = {
                    "train/step": global_step,
                    "train/loss_microbatch": loss.item(),
                    "train/unscaled_loss": metadata["unscaled_loss"].item(),
                    "train/avg_token_entropy": avg_entropy.item(),
                }
                wandb.log(log_data)

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1 # A global step is one optimizer update

                # --- Evaluation Step ---
                if (global_step % eval_every_n_steps == 0) and (global_step > 0):
                    print(f"Running evaluation at step {global_step}...")

                    # 1. Sync policy model weights to vLLM model
                    load_policy_into_vllm_instance(policy_model, vllm_model)

                    # 2. Run quantitative evaluation
                    eval_metrics = evaluate(
                        vllm_model=vllm_model,
                        reward_fn=r1_zero_reward_fn,
                        prompt_template=prompt_template,
                        validation_examples=validation_data,
                        eval_sampling_params=eval_sampling_params,
                        eval_batch_size=eval_batch_size
                    )

                    # 3. Run qualitative logging
                    log_gen_metrics = log_generations(
                        policy_model=policy_model,
                        vllm_model=vllm_model,
                        tokenizer=tokenizer,
                        reward_fn=r1_zero_reward_fn,
                        prompt_template=prompt_template,
                        examples=log_generation_examples,
                        sampling_params=eval_sampling_params
                    )

                    # Log all metrics
                    wandb_log_data = {
                        "eval/step": global_step,
                        **eval_metrics,
                        "eval/avg_response_length": log_gen_metrics["avg_response_length"],
                        "eval/avg_correct_response_length": log_gen_metrics["avg_correct_response_length"],
                        "eval/avg_incorrect_response_length": log_gen_metrics["avg_incorrect_response_length"],
                        "eval/avg_token_entropy": log_gen_metrics["avg_token_entropy"],
                    }

                    # Log the generations table
                    wandb_log_data["eval/generations"] = wandb.Table(
                        columns=log_gen_metrics["generations_table"]["columns"],
                        data=log_gen_metrics["generations_table"]["data"]
                    )

                    wandb.log(wandb_log_data)

                    # Set policy model back to train mode
                    policy_model.train()

    print("Training complete.")

    # Final evaluation
    print("Running final evaluation...")
    load_policy_into_vllm_instance(policy_model, vllm_model)
    final_metrics = evaluate(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompt_template=prompt_template,
        validation_examples=validation_data,
        eval_sampling_params=eval_sampling_params,
        eval_batch_size=eval_batch_size
    )
    print(f"Final Accuracy: {final_metrics['eval/accuracy']:.4f}")
    wandb.log({"eval/final_accuracy": final_metrics['eval/accuracy']})

    # --- Save Model ---
    print(f"Saving model to {output_dir}...")
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    policy_model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    print("Model saved.")

    wandb.finish()

if __name__ == "__main__":
    typer.run(main)