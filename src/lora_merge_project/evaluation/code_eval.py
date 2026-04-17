from __future__ import annotations

import argparse
from pathlib import Path

from lora_merge_project.config import load_config
from lora_merge_project.evaluation.common import (
    extract_code_block,
    generate_text,
    load_model_for_inference,
    load_processed_split,
    load_tokenizer,
    run_python_test,
    write_metrics,
)
from lora_merge_project.training.formatters import build_code_eval_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def build_candidate_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    code = extract_code_block(completion)
    if not code.strip():
        # Model produced no usable code; stub raises so the test records a clean failure
        return (
            f"def {entry_point}(*args, **kwargs):\n"
            f"    raise NotImplementedError('No code was generated')\n\n"
            f"{test}\n\ncheck({entry_point})\n"
        )
    if code.lstrip().startswith("def "):
        solution = code
    else:
        solution = f"{prompt.rstrip()}\n{code.rstrip()}\n"
    return f"{solution}\n\n{test}\n\ncheck({entry_point})\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tokenizer = load_tokenizer(config)
    model = load_model_for_inference(config, args.condition)

    output_dir = Path(config["paths"]["evaluations_dir"]) / args.condition
    records = load_processed_split(config, "humaneval_test.jsonl", max_samples=args.max_samples)
    generation_cfg = config["model"]["generation"]
    timeout_seconds = int(config["evaluation"]["code"]["timeout_seconds"])

    passed = 0
    predictions = []
    for record in records:
        messages = build_code_eval_messages(record["prompt"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        completion = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=generation_cfg["max_new_tokens_code"],
            temperature=float(generation_cfg["temperature"]),
            top_p=float(generation_cfg["top_p"]),
            do_sample=bool(generation_cfg["do_sample"]),
        )
        program = build_candidate_program(record["prompt"], completion, record["test"], record["entry_point"])
        is_pass, error = run_python_test(program, timeout_seconds=timeout_seconds)
        passed += int(is_pass)
        predictions.append(
            {
                "id": record["id"],
                "condition": args.condition,
                "dataset": "humaneval_test",
                "passed": is_pass,
                "error": error,
                "raw_completion": completion,
            }
        )

    metrics = {
        "condition": args.condition,
        "dataset": "humaneval_test",
        "num_examples": len(records),
        "pass_at_1": passed / max(len(records), 1),
    }
    write_metrics(output_dir, "humaneval_test", metrics, predictions)


if __name__ == "__main__":
    main()

