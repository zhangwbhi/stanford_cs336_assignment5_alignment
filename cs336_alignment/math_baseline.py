import sys
import json
import pathlib
from typing import Callable, List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from tqdm import tqdm
from utils import parse_ground_truth

# Import the reward function from the provided grader file
#
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# --- Constants ---
MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B" #
DATA_PATH = "../data/gsm8k/test.jsonl" #
PROMPT_PATH = "prompts/r1_zero.prompt" #
OUTPUT_PATH = "math_baseline_results.jsonl" #





def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    examples: List[Dict[str, Any]],
    prompt_template: str,
    eval_sampling_params: SamplingParams,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of examples,
    compute evaluation metrics, and optionally serialize results to disk.

    Args:
        vllm_model: The vLLM model instance.
        reward_fn: A function that takes (model_output, ground_truth) and
                   returns a dictionary of reward scores.
        examples: A list of example dictionaries, each with "question" and "answer" keys.
        prompt_template: The prompt template string (e.g., from r1_zero.prompt).
        eval_sampling_params: The SamplingParams for vLLM generation.
        output_path: If provided, the file path to serialize results to (e.g., "results.jsonl").

    Returns:
        A dictionary containing aggregate evaluation metrics.
    """

    # (1) Format prompts and parse ground truths
    prompts = []
    ground_truths = []
    for example in examples:
        question = example["question"]
        prompt_text = prompt_template.format(question=question) #
        prompts.append(prompt_text)

        gt_final_answer = parse_ground_truth(example["answer"])
        ground_truths.append(gt_final_answer)

    print(f"Starting generation for {len(prompts)} prompts...")

    # (3) Generate outputs for each example
    # The output is a list of RequestOutput objects
    outputs: List[RequestOutput] = vllm_model.generate(prompts, eval_sampling_params)

    results_list = []

    # (4) Calculate evaluation metrics
    print("Calculating rewards and processing outputs...")
    for i, output in enumerate(tqdm(outputs)):
        prompt = output.prompt
        # Get the first generated sequence and strip whitespace
        generated_text = output.outputs[0].text.strip()
        ground_truth = ground_truths[i]

        # Use the provided reward function
        reward_data = reward_fn(generated_text, ground_truth)

        result_entry = {
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "reward_data": reward_data,
            "prompt_index": i,
            "original_example": examples[i] # Include original for analysis
        }
        results_list.append(result_entry)

    # (5) Serialize the examples, generations, and scores to disk (if path provided)
    if output_path:
        print(f"Serializing {len(results_list)} results to {output_path}...")
        with open(output_path, "w") as f:
            for entry in results_list:
                # Need a custom encoder if 'original_example' is not JSON-serializable
                # For this assignment, it should be fine.
                f.write(json.dumps(entry) + "\n")
    else:
        print("Skipping serialization as output_path is None.")

    # Calculate metrics for Problem (b)
    cat1_correct_all = sum(
        1 for r in results_list
        if r["reward_data"].get("answer_reward") == 1.0 and r["reward_data"].get("format_reward") == 1.0
    )
    cat2_format_only = sum(
        1 for r in results_list
        if r["reward_data"].get("answer_reward") == 0.0 and r["reward_data"].get("format_reward") == 1.0
    )
    cat3_incorrect_all = sum(
        1 for r in results_list
        if r["reward_data"].get("answer_reward") == 0.0 and r["reward_data"].get("format_reward") == 0.0
    )

    total_processed = len(results_list)
    other_cases = total_processed - (cat1_correct_all + cat2_format_only + cat3_incorrect_all)

    # Aggregate metrics for Problem (c)
    total_answer_reward = sum(r["reward_data"].get("answer_reward", 0.0) for r in results_list)
    total_format_reward = sum(r["reward_data"].get("format_reward", 0.0) for r in results_list)
    total_count = len(results_list)

    metrics = {
        "total_examples": total_count,
        "accuracy": total_answer_reward / total_count if total_count > 0 else 0.0,
        "avg_format_reward": total_format_reward / total_count if total_count > 0 else 0.0,
        "count_correct_format_and_answer": cat1_correct_all,
        "count_correct_format_wrong_answer": cat2_format_only,
        "count_wrong_format_and_answer": cat3_incorrect_all,
        "count_other_cases": other_cases
    }

    print("Evaluation complete.")
    print("--- METRICS ---")
    print(json.dumps(metrics, indent=2))
    print("---------------")

    return metrics


def main():
    """
    Main script to run the MATH zero-shot baseline evaluation.
    """
    # (1) Load the MATH validation examples
    print(f"Loading data from {DATA_PATH}...")
    examples = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    # Load the prompt template
    print(f"Loading prompt template from {PROMPT_PATH}...")
    prompt_template = pathlib.Path(PROMPT_PATH).read_text()

    print(f"Loaded {len(examples)} examples.")

    # (3) Generate outputs
    print(f"Initializing model from {MODEL_PATH}...")
    llm = LLM(model=MODEL_PATH) #

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # (4) & (5) Evaluate and serialize
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn, #
        examples=examples,
        prompt_template=prompt_template,
        eval_sampling_params=sampling_params,
        output_path=OUTPUT_PATH #
    )



if __name__ == "__main__":
    main()
