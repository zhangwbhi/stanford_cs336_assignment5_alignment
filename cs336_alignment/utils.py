import torch
import torch.nn.functional as F

from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any, Callable
from vllm import LLM, SamplingParams
import json
import re

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask
    that is 1 for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:

            input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input ids, i.e., the input ids without the first token.
            response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in the labels.
    """

    # --- 1. Set Pad Token ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We need the pad token ID as an integer for torch.nn.functional.pad
    pad_token_id = tokenizer.pad_token_id

    # --- 2. Tokenize Prompts and Outputs Separately ---
    # This is the logically correct way, as tokenizer(p+o) != tokenizer(p)+tokenizer(o)
    prompt_input_ids_list = [
        torch.tensor(tokenizer.encode(p, add_special_tokens=False), dtype=torch.int64)
        for p in prompt_strs
    ]

    # We add the eos_token to the end of the output, which is required
    # to match the test snapshots.
    output_input_ids_list = [
        torch.tensor(tokenizer.encode(o, add_special_tokens=False), dtype=torch.int64)
        for o in output_strs
    ]

    # --- 3. Get Max Length ---
    full_seq_lens = [
        len(p) + len(o) for p, o in zip(prompt_input_ids_list, output_input_ids_list)
    ]
    max_seq_len = max(full_seq_lens) # This is N

    # --- 4. Manually Pad and Create Tensors ---
    input_ids_batch = []
    labels_batch = []
    response_mask_batch = []

    for p_ids, o_ids in zip(prompt_input_ids_list, output_input_ids_list):
        full_ids = torch.cat([p_ids, o_ids], dim=0)

        # Create a response mask for the full sequence (N)
        # 0 for prompt, 1 for output
        resp_mask = torch.cat(
            [torch.zeros_like(p_ids), torch.ones_like(o_ids)], dim=0
        )

        # Calculate how much padding is needed
        num_padding = max_seq_len - len(full_ids)

        # Pad the full_ids sequence to max_seq_len
        # (0, num_padding) means 0 padding on the left, num_padding on the right
        padded_full_ids = torch.nn.functional.pad(
            full_ids, (0, num_padding), value=pad_token_id
        )

        # Pad the response mask (0s for padded tokens)
        padded_mask = torch.nn.functional.pad(
            resp_mask, (0, num_padding), value=0
        )

        # --- 5. Create input_ids, labels, and response_mask (N-1) ---

        # input_ids = [t1, ..., tN-1]
        input_ids_batch.append(padded_full_ids[:-1])

        # labels = [t2, ..., tN]
        # We also set padding tokens to -100
        labels_unmasked = padded_full_ids[1:]

        # response_mask = [mask_t2, ..., mask_tN]
        response_mask_shifted = padded_mask[1:]


        labels_batch.append(labels_unmasked)
        response_mask_batch.append(response_mask_shifted)

    # --- 6. Stack Tensors ---
    return {
        "input_ids": torch.stack(input_ids_batch),
        "labels": torch.stack(labels_batch),
        "response_mask": torch.stack(response_mask_batch)
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
            prediction.
    """
    # The entropy formula is H(p) = -sum(p(x) * log(p(x)))

    # 1. Get log_probs = log(p(x))
    # F.log_softmax numerically stably computes log(softmax(logits))
    # It uses the log-sum-exp trick internally.
    log_probs = F.log_softmax(logits, dim=-1)

    # 2. Get probs = p(x)
    # We can get this stably by just exponentiating the log_probs
    probs = torch.exp(log_probs)

    # 3. Compute entropy
    # H = -sum(p(x) * log(p(x)))
    # We sum over the last dimension (the vocabulary)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy


def get_response_log_probs(
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1).gather(index=labels.unsqueeze(-1), dim=-1).squeeze(-1)

    if not return_token_entropy:
        return {
            "log_probs": log_probs
        }

    return {
        "log_probs": log_probs,
        "token_entropy": compute_entropy(logits)
    }


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    """
    Execute a forward-and-backward pass on a microbatch for SFT.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length),
            containing the per-token log-probabilities from the SFT policy
            being trained (e.g., from get_response_log_probs).
        response_mask: torch.Tensor of shape (batch_size, sequence_length),
            with 1 for response tokens and 0 for prompt/padding tokens.
        gradient_accumulation_steps: int. The number of microbatches
            per optimizer step.
        normalize_constant: float. The constant by which to divide the sum
            of the losses. Per the PDF, it is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            - loss: A scalar tensor. The microbatch loss, adjusted for
              gradient accumulation. This is the value to call .backward() on.
            - metadata: A dictionary with metadata from the loss calculation,
              such as the unscaled loss.
    """
    masked_normalized_probs = masked_normalize(
        policy_log_probs, response_mask, normalize_constant, - 1
    )
    unscaled_loss = - masked_normalized_probs.mean()
    loss = unscaled_loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        # Detach to prevent saving computational graph during logging
        "unscaled_loss": unscaled_loss.detach(),
        "total_loss_sum": - masked_normalized_probs.detach(),
        "num_response_tokens": torch.sum(response_mask).detach()
    }

    return loss.detach(), metadata


def parse_ground_truth(gt_answer: str) -> str:
    """
    Extracts the final boxed answer (e.g., "694") from the full
    reasoning trace (e.g., "... #### 694").
    """
    parts = gt_answer.split("####")
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        # Fallback if the '####' separator isn't present
        return gt_answer.strip()


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


@torch.inference_mode()
def log_generations(
    policy_model: PreTrainedModel,
    vllm_model: LLM,
    tokenizer: PreTrainedTokenizer,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompt_template: str,
    examples: List[Dict[str, Any]],
    sampling_params: SamplingParams,
) -> Dict[str, Any]:
    """
    Logs generations from the model during training.

    Args:
        policy_model: The policy model being trained (on GPU 0).
        vllm_model: The vLLM instance (on GPU 1) for fast generation.
        tokenizer: The tokenizer.
        reward_fn: The reward function (e.g., r1_zero_reward_fn).
        prompt_template: The prompt template string (e.g., r1_zero.prompt).
        examples: A list of validation example dicts, each with "question" and "answer".
        sampling_params: The vLLM SamplingParams to use for generation.

    Returns:
        A dictionary containing aggregated metrics for logging (e.g., to wandb).
    """
    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]
    ground_truths = [parse_ground_truth(ex["answer"]) for ex in examples]

    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]

    tokenized_data = tokenize_prompt_and_output(
        prompt_strs=prompts,
        output_strs=responses,
        tokenizer=tokenizer
    )

    policy_model_device = next(policy_model.parameters()).device
    log_probs_data = get_response_log_probs(
        model=policy_model,
        input_ids=tokenized_data["input_ids"].to(policy_model_device),
        labels=tokenized_data["labels"].to(policy_model_device),
        return_token_entropy=True
    )

    token_entropies = log_probs_data["token_entropy"] # batch_size x seq_len
    response_mask = tokenized_data["response_mask"].to(policy_model_device) # batch_size x seq_len


    log_table_rows = []

    total_len, correct_len, incorrect_len = 0, 0, 0
    total_correct, total_incorrect, total_avg_entropy = 0, 0, 0


    for i, ex in enumerate(examples):
        reward_data = reward_fn(responses[i], ground_truths[i])
        is_correct = reward_data.get("answer_reward", 0.0) == 1.0

        response_len = response_mask[i].sum().item()
        total_len += response_len

        if is_correct:
            correct_len += response_len
            total_correct += 1
        else:
            incorrect_len += response_len
            total_incorrect += 1

        masked_entropies = token_entropies[i] * response_mask[i]
        avg_entropy = masked_entropies.sum() / response_len if response_len > 0 else 0.0
        avg_entropy =  avg_entropy.item()
        total_avg_entropy  += avg_entropy

        # log first 5 examples
        if i < 5:
            log_table_rows.append([
                prompts[i],
                responses[i],
                ground_truths[i],
                json.dumps(reward_data),
                avg_entropy,
                response_len
            ])

    num_examples = len(examples)

    avg_len = total_len / num_examples if num_examples > 0 else 0.0
    avg_correct_len = correct_len / total_correct if total_correct > 0 else 0.0
    avg_incorrect_len = incorrect_len / total_incorrect if total_incorrect > 0 else 0.0

    final_avg_entropy = total_avg_entropy / num_examples if num_examples > 0 else 0.0

    metrics_to_log = {
        "avg_response_length": avg_len,
        "avg_correct_response_length": avg_correct_len,
        "avg_incorrect_response_length": avg_incorrect_len,
        "avg_token_entropy": final_avg_entropy,
        "generations_table": {
            "columns": ["Prompt", "Generated", "Ground Truth", "Reward", "Avg Entropy", "Length"],
            "data": log_table_rows
        }
    }

    return metrics_to_log