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
from typing import Callable, List, Tuple, Dict, Any, Optional, Literal
from unittest.mock import patch
from collections import defaultdict
import numpy as np

from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams

# Imports from project modules
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    masked_normalize,
    log_generations,
)
from cs336_alignment.math_baseline import (
    run_vllm,
    extract_reference_answer,
    format_prompt_with_template,
)
from cs336_alignment.grpo_utils import compute_group_normalized_rewards, grpo_microbatch_train_step



MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
TRAIN_DATA_PATH = "./data/gsm8k/train.jsonl"  # Prompts for rollouts
VALID_DATA_PATH = "./data/gsm8k/test.jsonl"
PROMPT_PATH = "./cs336_alignment/prompts/r1_zero.prompt"

VLLM_DEVICE = "cuda:0"
POLICY_DEVICE = "cuda:1"


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


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
):
    """
    Runs evaluation on the vLLM model against a set of prompts and ground-truth answers.
    """
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)

    stats = dict()
    total_rewards = []
    for response, answer in zip(responses, answers):
        ground_truth = extract_reference_answer(answer)
        metrics = reward_fn(response, ground_truth)
        total_rewards.append(metrics["reward"])

        # Detailed breakdown of correctness
        if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
            stats["eval/correct"] = stats.get("eval/correct", 0) + 1
        elif metrics["format_reward"] == 1 and metrics["answer_reward"] == 0:
            stats["eval/format_correct_answer_wrong"] = (
                stats.get("eval/format_correct_answer_wrong", 0) + 1
            )
        elif metrics["format_reward"] == 0 and metrics["answer_reward"] == 1:
            stats["eval/format_wrong_answer_correct"] = (
                stats.get("eval/format_wrong_answer_correct", 0) + 1
            )
        else:
            stats["eval/wrong"] = stats.get("eval/wrong", 0) + 1

    stats["eval/reward_mean"] = np.mean(total_rewards)
    stats["eval/reward_std"] = np.std(total_rewards)
    return stats


def main(
    # --- GRPO Hyperparameters ---
    n_grpo_steps: int = typer.Option(
        200, help="Number of GRPO steps (rollout -> train)."
    ),
    learning_rate: float = typer.Option(1e-5, help="Learning rate for AdamW."),
    advantage_eps: float = typer.Option(
        1e-6, help="Epsilon for advantage normalization."
    ),
    rollout_batch_size: int = typer.Option(
        256, help="Total number of rollouts to generate per step."
    ),
    group_size: int = typer.Option(8, help="Number of responses per prompt."),
    sampling_temperature: float = typer.Option(
        1.0, help="Temperature for sampling rollouts."
    ),
    sampling_min_tokens: int = typer.Option(
        4, help="Minimum tokens to sample for rollouts."
    ),
    sampling_max_tokens: int = typer.Option(
        1024, help="Maximum tokens to sample for rollouts."
    ),
    epochs_per_rollout_batch: int = typer.Option(
        1, help="Number of training epochs per rollout batch (for off-policy)."
    ),
    train_batch_size: int = typer.Option(
        256, help="Effective batch size for one optimizer step."
    ),
    gradient_accumulation_steps: int = typer.Option(
        128, help="Number of microbatches to accumulate for one optimizer step."
    ),
    gpu_memory_utilization: float = typer.Option(
        0.85, help="GPU memory utilization for vLLM."
    ),
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip"
    ] = typer.Option("reinforce_with_baseline", help="Type of GRPO loss to use."),
    use_std_normalization: bool = typer.Option(
        True, help="Normalize advantages by std dev."
    ),
    cliprange: float = typer.Option(
        0.2, help="Clip range for GRPO-Clip loss."
    ),
    # --- General/Eval Hyperparameters ---
    eval_every_n_steps: int = typer.Option(
        10, help="Run evaluation every N global steps."
    ),
    n_eval_examples: int = typer.Option(
        1024, help="Number of validation examples to run."
    ),
    output_dir: str = typer.Option(
        "models/grpo_qwen", help="Directory to save the final model."
    ),
    wandb_project: str = typer.Option("cs336-a5-grpo", help="WandB project name."),
    seed: int = typer.Option(42, help="Random seed."),
):
    if torch.cuda.device_count() < 2:
        print("Error: this script requires at least 2 GPUs.")
        return

    # --- Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Hyperparameter Sanity Checks ---
    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    print(f"Using microbatch size: {micro_train_batch_size}")

    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    print(f"Generating {n_prompts_per_rollout_batch} prompts per rollout step.")

    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    # This check ensures one pass over the rollout data = one optimizer step
    # (in the on-policy case where train_batch_size == rollout_batch_size)
    assert (
        rollout_batch_size % micro_train_batch_size == 0
    ), "rollout_batch_size must be divisible by micro_train_batch_size"
    n_microbatches_per_rollout_batch = (
        rollout_batch_size // micro_train_batch_size
    )
    print(
        f"Processing {n_microbatches_per_rollout_batch} microbatches per rollout batch."
    )

    if (
        loss_type == "grpo_clip"
        and epochs_per_rollout_batch == 1
        and train_batch_size == rollout_batch_size
    ):
        print(
            "Warning: GRPO-Clip is typically used in the off-policy setting"
            " (epochs_per_rollout_batch > 1)."
        )

    # --- WandB ---
    wandb.init(
        project=wandb_project,
        config={
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "advantage_eps": advantage_eps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "sampling_min_tokens": sampling_min_tokens,
            "sampling_max_tokens": sampling_max_tokens,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "micro_train_batch_size": micro_train_batch_size,
            "loss_type": loss_type,
            "use_std_normalization": use_std_normalization,
            "cliprange": cliprange,
            "model_path": MODEL_PATH,
        },
    )
    wandb.define_metric("train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    # --- Model, Tokenizer, Optimizer ---
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=POLICY_DEVICE,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    amp_ctx = torch.amp.autocast(device_type=POLICY_DEVICE, dtype=torch.bfloat16)

    # --- vLLM ---
    vllm = init_vllm(MODEL_PATH, VLLM_DEVICE, seed, gpu_memory_utilization)

    # --- Prompt & Data Loading ---
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Load training data (prompts)
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_data = [json.loads(line.strip()) for line in f]
    train_prompts = [
        format_prompt_with_template(example["question"], prompt_template)
        for example in train_data
    ]
    train_ground_truths = [extract_reference_answer(example["answer"]) for example in train_data]
    print(f"Loaded {len(train_prompts)} training prompts.")

    # Load validation data
    with open(VALID_DATA_PATH, "r", encoding="utf-8") as f:
        validation_data = [json.loads(line.strip()) for line in f]

    if n_eval_examples:
        print(f"Slicing validation data to {n_eval_examples} examples.")
        validation_data = validation_data[:n_eval_examples]

    validation_data_prompts = [
        format_prompt_with_template(example["question"], prompt_template)
        for example in validation_data
    ]
    validation_data_answers = [example["answer"] for example in validation_data]
    log_generation_examples = validation_data[:32]  # For qualitative logging

    # --- Sampling Params ---
    rollout_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0,  # Use temp=1.0 for evaluation as in SFT
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print(f"Starting GRPO for {n_grpo_steps} steps...")
    print(f"  Rollout batch size: {rollout_batch_size}")
    print(f"  Train effective batch size: {train_batch_size}")

    for step in tqdm(range(n_grpo_steps), desc="GRPO Steps"):

        # 1. --- Sync Policy to vLLM ---
        # We sync every step to ensure rollouts are from the latest policy
        load_policy_into_vllm_instance(model, vllm)

        # 2. --- Rollout Phase ---
        print(f"\n[Step {step+1}/{n_grpo_steps}] Generating {rollout_batch_size} rollouts...")

        # Sample prompts
        indices = random.sample(
            range(len(train_prompts)), n_prompts_per_rollout_batch
        )
        batch_prompts = [train_prompts[i] for i in indices]
        batch_ground_truths = [train_ground_truths[i] for i in indices]

        # Repeat for group size
        repeated_prompts = [
            p for p in batch_prompts for _ in range(group_size)
        ]
        repeated_ground_truths = [
            gt for gt in batch_ground_truths for _ in range(group_size)
        ]

        # Generate responses
        rollout_responses = run_vllm(
            vllm, repeated_prompts, rollout_sampling_params
        )

        # 3. --- Reward/Advantage Phase ---
        print(f"[Step {step+1}] Computing rewards and advantages...")
        advantages, raw_rewards, reward_metadata = (
            compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=use_std_normalization,
            )
        )

        # Log training rewards
        wandb.log(
            {"train/step": step, "train/reward_mean": raw_rewards.mean().item()}
        )
        wandb.log(
            {"train/step": step, **reward_metadata}
        )


        # 4. --- Tokenize Rollouts for Training ---
        tokenized_rollouts = tokenize_prompt_and_output(
            prompt_strs=repeated_prompts,
            output_strs=rollout_responses,
            tokenizer=tokenizer,
        )
        rollout_input_ids = tokenized_rollouts["input_ids"]
        rollout_labels = tokenized_rollouts["labels"]
        rollout_response_mask = tokenized_rollouts["response_mask"]

        # 5. --- Compute Old Log Probs (if off-policy) ---
        old_log_probs = None
        if loss_type == "grpo_clip":
            print(f"[Step {step+1}] Computing old log-probs for GRPO-Clip...")
            # We compute this once before the training epochs
            #
            # Note: We must send the *entire* rollout batch to the
            # policy device at once. This might be a memory bottleneck.
            # If so, this loop would need to be refactored to
            # compute old_log_probs in chunks.
            with torch.no_grad(), amp_ctx:
                old_log_probs_data = get_response_log_probs(
                    model,
                    rollout_input_ids.to(POLICY_DEVICE),
                    rollout_labels.to(POLICY_DEVICE),
                )
                # Detach to prevent differentiating
                old_log_probs = old_log_probs_data["log_probs"].detach()

        # 6. --- Training Phase ---
        print(f"[Step {step+1}] Running {epochs_per_rollout_batch} training epoch(s)...")

        for epoch in range(epochs_per_rollout_batch):
            # Shuffle data for each epoch
            epoch_indices = list(range(rollout_batch_size))
            random.shuffle(epoch_indices)

            # Accumulators for logging
            epoch_logs = defaultdict(list)

            optimizer.zero_grad()

            # This loop iterates over the *entire* rollout batch,
            # accumulating gradients for one optimizer step
            # (assuming train_batch_size == rollout_batch_size)
            # If train_batch_size < rollout_batch_size, this loop
            # would perform (rollout / train) optimizer steps.
            # Here we assume on-policy: one step per rollout.

            if rollout_batch_size != train_batch_size:
                 print(
                    f"Warning: rollout_batch_size ({rollout_batch_size}) != "
                    f"train_batch_size ({train_batch_size}). "
                    "This script assumes on-policy (1 optimizer step per rollout batch)."
                 )


            for i in tqdm(
                range(0, rollout_batch_size, micro_train_batch_size),
                desc=f"Epoch {epoch+1} Microbatches",
                leave=False
            ):
                microbatch_indices = epoch_indices[i : i + micro_train_batch_size]

                # --- Prepare Microbatch ---
                mb_input_ids = rollout_input_ids[microbatch_indices].to(
                    POLICY_DEVICE
                )
                mb_labels = rollout_labels[microbatch_indices].to(
                    POLICY_DEVICE
                )
                mb_response_mask = rollout_response_mask[microbatch_indices].to(
                    POLICY_DEVICE
                )

                # Squeeze to (batch_size, 1) for broadcasting
                mb_advantages = advantages[microbatch_indices].to(
                    POLICY_DEVICE
                ).unsqueeze(1)
                mb_raw_rewards = raw_rewards[microbatch_indices].to(
                    POLICY_DEVICE
                ).unsqueeze(1)

                mb_old_log_probs = (
                    old_log_probs[microbatch_indices].to(POLICY_DEVICE)
                    if old_log_probs is not None
                    else None
                )

                # --- Forward Pass (Current Policy) ---
                with amp_ctx:
                    log_probs_data = get_response_log_probs(
                        model,
                        mb_input_ids,
                        mb_labels,
                        return_token_entropy=True,
                    )
                    policy_log_probs = log_probs_data["log_probs"]

                # --- Backward Pass (Microbatch) ---
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=mb_response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=mb_raw_rewards,
                    advantages=mb_advantages,
                    old_log_probs=mb_old_log_probs,
                    cliprange=cliprange,
                )

                # --- Accumulate Logs ---
                epoch_logs["loss"].append(loss.item())

                # Token entropy
                token_entropy = log_probs_data["token_entropy"]
                avg_entropy = (
                    (token_entropy * mb_response_mask).sum()
                    / mb_response_mask.sum().add(1e-8)
                )
                epoch_logs["token_entropy"].append(avg_entropy.item())

                # Clip fraction
                if "clipped" in metadata:
                    clip_frac = metadata["clipped"].float().mean().item()
                    epoch_logs["clip_frac"].append(clip_frac)

            # --- End of Microbatch Loop ---

            # --- Optimizer Step ---
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            # --- Log Training Stats (for this optimizer step) ---
            mean_loss = np.mean(epoch_logs["loss"]) * gradient_accumulation_steps
            mean_entropy = np.mean(epoch_logs["token_entropy"])

            wandb_log = {
                "train/step": step,
                "train/loss": mean_loss,
                "train/grad_norm": grad_norm.item(),
                "train/token_entropy": mean_entropy,
            }
            if epoch_logs["clip_frac"]:
                wandb_log["train/clip_frac"] = np.mean(epoch_logs["clip_frac"])

            wandb.log(wandb_log)

            print(f"[Step {step+1}] Epoch {epoch+1} complete. Loss: {mean_loss:.4f}")

        # 7. --- Evaluation Phase ---
        if (step + 1) % eval_every_n_steps == 0:
            print(f"\n[Step {step+1}] Running evaluation...")

            # 1. Sync policy model (already synced at start of step)
            # load_policy_into_vllm_instance(model, vllm)

            # 2. Run quantitative evaluation
            stats = evaluate_vllm(
                vllm,
                r1_zero_reward_fn,
                validation_data_prompts,
                validation_data_answers,
                eval_sampling_params,
            )
            accuracy = stats["eval/correct"] / len(validation_data_prompts)
            print(f"[Step {step+1}] Eval Accuracy: {accuracy:.4f}")

            # 3. Run qualitative logging
            log_gen_metrics = log_generations(
                policy_model=model,
                vllm_model=vllm,
                tokenizer=tokenizer,
                reward_fn=r1_zero_reward_fn,
                prompt_template=prompt_template,
                examples=log_generation_examples,
                sampling_params=eval_sampling_params,
            )

            wandb_log_data = {
                "eval/step": step,
                "eval/accuracy": accuracy,
                **stats,
                "eval/avg_response_length": log_gen_metrics[
                    "avg_response_length"
                ],
                "eval/avg_correct_response_length": log_gen_metrics[
                    "avg_correct_response_length"
                ],
                "eval/avg_incorrect_response_length": log_gen_metrics[
                    "avg_incorrect_response_length"
                ],
                "eval/avg_token_entropy": log_gen_metrics["avg_token_entropy"],
            }
            wandb.log(wandb_log_data)

    # --- End of Training ---
    print("Training complete.")

    print("Running final evaluation...")
    load_policy_into_vllm_instance(model, vllm)
    stats = evaluate_vllm(
        vllm,
        r1_zero_reward_fn,
        validation_data_prompts,
        validation_data_answers,
        eval_sampling_params,
    )
    accuracy = stats["eval/correct"] / len(validation_data_prompts)
    print(f"Final Accuracy: {accuracy:.4f}")
    wandb.log({"eval/final_accuracy": accuracy, **stats})

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