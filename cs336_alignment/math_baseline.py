import json
import os
import re
from typing import List, Callable, Dict
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def run_vllm(
        vllm_model: LLM,
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> List[str]:
    responses = vllm_model.generate(prompts, sampling_params)
    texts = [output.outputs[0].text.strip() for output in responses]
    return texts


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_prompt_with_template(question: str, template: str) -> str:
    return template.format(question=question)


def evaluate_vllm(
    data: List[Dict],
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    # 3. Generate outputs for each example
    generated_texts = run_vllm(vllm_model, prompts, eval_sampling_params)

    # 4. Calculate evaluation metrics and collect examples
    stats = dict()

    correct_examples = []
    format_wrong_answer_correct_examples = []
    format_correct_answer_wrong_examples = []
    wrong_examples = []

    for prompt, generated_text, example in zip(prompts, generated_texts, data):
        ground_truth = example["answer"]
        reference_answer = extract_reference_answer(ground_truth)
        metrics = reward_fn(generated_text, reference_answer)

        if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
            stats["correct"] = stats.get("correct", 0) + 1
            correct_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        elif metrics["format_reward"] == 1 and metrics["answer_reward"] == 0:
            stats["format_correct_answer_wrong"] = stats.get("format_correct_answer_wrong", 0) + 1
            format_correct_answer_wrong_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        elif metrics["format_reward"] == 0 and metrics["answer_reward"] == 1:
            stats["format_wrong_answer_correct"] = stats.get("format_wrong_answer_correct", 0) + 1
            format_wrong_answer_correct_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        else:
            stats["wrong"] = stats.get("wrong", 0) + 1
            wrong_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })



    # 5. Save results
    os.makedirs("outputs", exist_ok=True)
    outputpath = os.path.join("outputs", "eval_results.jsonl")
    with open(outputpath, "w", encoding="utf-8") as f:
        for ex in correct_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in format_correct_answer_wrong_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in format_wrong_answer_correct_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for ex in wrong_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


    print(f"Evaluation results saved to {outputpath}")

    return stats, {
        "correct": correct_examples,
        "format_correct_answer_wrong": format_correct_answer_wrong_examples,
        "format_wrong_answer_correct": format_wrong_answer_correct_examples,
        "wrong": wrong_examples
    }


def main():
    # llm
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

    # prompts
    # 1. load data/gsm8k/test.jsonl
    data = load_jsonl("./data/gsm8k/test.jsonl")

    # 2. format and use r1_zero prompt cs336_alignment/prompts/r1_zero.prompt
    template_path = "./cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    formatted_prompts = [
        format_prompt_with_template(example["question"], template) for example in data
    ]

    # sampling params
    sampling_params =  SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # eval
    stats, evaluated_examples = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        data=data,
        prompts=formatted_prompts,
        eval_sampling_params=sampling_params
    )


    print("\n🔴 Wrong:")
    for ex in evaluated_examples["wrong"][:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n🟡 Format Correct Answer Wrong:")
    for ex in evaluated_examples["format_correct_answer_wrong"][:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n🟡 Format Wrong Answer Correct:")
    for ex in evaluated_examples["format_wrong_answer_correct"][:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print("\n✅ Correct:")
    for ex in evaluated_examples["correct"][:10]:
        print("- Prompt:", ex["prompt"])
        print("  Response:", ex["response"])
        print("  Reference Answer:", ex["reference_answer"])
        print("  Extracted reference Answer:", ex["reference_answer_extracted"])
        print()

    print ("overview:", stats)




if __name__ == "__main__":
    print(os.getcwd())
    main()