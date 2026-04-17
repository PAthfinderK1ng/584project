from __future__ import annotations

import argparse
import re
from pathlib import Path

from sympy import SympifyError, sympify

from lora_merge_project.config import load_config
from lora_merge_project.evaluation.common import (
    generate_text,
    load_model_for_inference,
    load_processed_split,
    load_tokenizer,
    write_metrics,
)
from lora_merge_project.training.formatters import build_math_eval_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def latex_to_plain(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", expr)
    expr = re.sub(r"\\left|\s*\\right", "", expr)
    expr = expr.replace("$", "")
    expr = expr.replace(",", "")
    expr = expr.replace("^", "**")
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = expr.replace("{", "(").replace("}", ")")
    return expr.strip().strip(".")


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", latex_to_plain(answer)).strip().lower()


def extract_reference_answer(record: dict[str, str]) -> str:
    if record["source_dataset"] == "math500_test":
        return str(record["final_answer"]).strip()
    match = re.search(r"####\s*(.+)$", record["solution"], flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return record["solution"].strip().splitlines()[-1].strip()


def extract_predicted_answer(text: str) -> str:
    patterns = [
        r"Final answer:\s*(.+)",
        r"Answer:\s*(.+)",
        r"####\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    non_empty = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty:
        return non_empty[-1]
    return text.strip()


def answers_match(predicted: str, gold: str) -> bool:
    normalized_pred = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    if normalized_pred == normalized_gold:
        return True
    try:
        return bool(sympify(normalized_pred).equals(sympify(normalized_gold)))
    except (SympifyError, TypeError, AttributeError, NotImplementedError):
        return False


def evaluate_dataset(model, tokenizer, config: dict, dataset_name: str, records: list[dict], output_dir: Path, condition: str) -> dict:
    generation_cfg = config["model"]["generation"]
    predictions = []
    correct = 0
    for record in records:
        messages = build_math_eval_messages(record["question"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=generation_cfg["max_new_tokens_math"],
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )
        gold = extract_reference_answer(record)
        predicted = extract_predicted_answer(completion)
        is_correct = answers_match(predicted, gold)
        correct += int(is_correct)
        predictions.append(
            {
                "id": record["id"],
                "condition": condition,
                "dataset": dataset_name,
                "question": record["question"],
                "gold": gold,
                "predicted": predicted,
                "correct": is_correct,
                "raw_completion": completion,
            }
        )
    accuracy = correct / max(len(records), 1)
    metrics = {
        "condition": condition,
        "dataset": dataset_name,
        "num_examples": len(records),
        "accuracy": accuracy,
    }
    write_metrics(output_dir, dataset_name, metrics, predictions)
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    model = load_model_for_inference(config, args.condition)

    output_dir = Path(config["paths"]["evaluations_dir"]) / args.condition
    datasets_to_run = {
        "gsm8k_test": load_processed_split(config, "gsm8k_test.jsonl", max_samples=args.max_samples),
        "math500_test": load_processed_split(config, "math500_test.jsonl", max_samples=args.max_samples),
    }
    for dataset_name, records in datasets_to_run.items():
        evaluate_dataset(model, tokenizer, config, dataset_name, records, output_dir, args.condition)


if __name__ == "__main__":
    main()

