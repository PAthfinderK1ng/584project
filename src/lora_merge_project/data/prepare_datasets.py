from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset

from lora_merge_project.config import load_config
from lora_merge_project.utils import ensure_dir, write_json, write_jsonl


def _limit(records: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    if max_samples is None:
        return records
    return records[:max_samples]


def prepare_math_train(dataset) -> list[dict[str, Any]]:
    records = []
    for index, example in enumerate(dataset):
        records.append(
            {
                "id": f"gsm8k-train-{index}",
                "source_dataset": "gsm8k_train",
                "question": example["question"].strip(),
                "solution": example["answer"].strip(),
            }
        )
    return records


def prepare_math_test(dataset) -> list[dict[str, Any]]:
    records = []
    for index, example in enumerate(dataset):
        records.append(
            {
                "id": f"gsm8k-test-{index}",
                "source_dataset": "gsm8k_test",
                "question": example["question"].strip(),
                "solution": example["answer"].strip(),
            }
        )
    return records


def prepare_math500_test(dataset) -> list[dict[str, Any]]:
    records = []
    for index, example in enumerate(dataset):
        records.append(
            {
                "id": example.get("unique_id", f"math500-{index}"),
                "source_dataset": "math500_test",
                "question": example["problem"].strip(),
                "solution": example["solution"].strip(),
                "final_answer": str(example["answer"]).strip(),
                "subject": example.get("subject", ""),
                "level": example.get("level", ""),
            }
        )
    return records


def prepare_mbpp_split(dataset, split_name: str) -> list[dict[str, Any]]:
    records = []
    for example in dataset:
        prompt = example.get("prompt", example.get("text", "")).strip()
        test_setup_code = example.get("test_setup_code")
        if test_setup_code is None:
            imports = example.get("test_imports", [])
            test_setup_code = "\n".join(imports) if imports else ""
        records.append(
            {
                "id": f"mbpp-{split_name}-{example['task_id']}",
                "source_dataset": f"mbpp_{split_name}",
                "task_id": example["task_id"],
                "prompt": prompt,
                "code": example["code"].strip(),
                "test_list": example.get("test_list", []),
                "challenge_test_list": example.get("challenge_test_list", []),
                "test_setup_code": test_setup_code,
            }
        )
    return records


def prepare_humaneval(dataset) -> list[dict[str, Any]]:
    records = []
    for example in dataset:
        records.append(
            {
                "id": example["task_id"],
                "source_dataset": "humaneval_test",
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "canonical_solution": example["canonical_solution"],
                "test": example["test"],
                "entry_point": example["entry_point"],
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-math-train", type=int)
    parser.add_argument("--max-math-test", type=int)
    parser.add_argument("--max-code-train", type=int)
    parser.add_argument("--max-code-test", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    processed_dir = ensure_dir(config["paths"]["processed_data_dir"])

    gsm8k_cfg = config["datasets"]["gsm8k"]
    gsm8k = load_dataset(gsm8k_cfg["path"], gsm8k_cfg.get("config"))
    math_train = _limit(prepare_math_train(gsm8k[gsm8k_cfg["train_split"]]), args.max_math_train)
    math_test = _limit(prepare_math_test(gsm8k[gsm8k_cfg["test_split"]]), args.max_math_test)

    math500_cfg = config["datasets"]["math500"]
    math500 = load_dataset(math500_cfg["path"])
    math500_test = _limit(prepare_math500_test(math500[math500_cfg["test_split"]]), args.max_math_test)

    mbpp_cfg = config["datasets"]["mbpp"]
    mbpp = load_dataset(mbpp_cfg["path"], mbpp_cfg.get("config"))
    code_train = _limit(prepare_mbpp_split(mbpp[mbpp_cfg["train_split"]], "train"), args.max_code_train)
    code_val = prepare_mbpp_split(mbpp[mbpp_cfg["validation_split"]], "validation")
    code_test = _limit(prepare_mbpp_split(mbpp[mbpp_cfg["test_split"]], "test"), args.max_code_test)

    humaneval_cfg = config["datasets"]["humaneval"]
    humaneval = load_dataset(humaneval_cfg["path"])
    humaneval_test = _limit(prepare_humaneval(humaneval[humaneval_cfg["test_split"]]), args.max_code_test)

    write_jsonl(processed_dir / "gsm8k_train.jsonl", math_train)
    write_jsonl(processed_dir / "gsm8k_test.jsonl", math_test)
    write_jsonl(processed_dir / "math500_test.jsonl", math500_test)
    write_jsonl(processed_dir / "mbpp_train.jsonl", code_train)
    write_jsonl(processed_dir / "mbpp_validation.jsonl", code_val)
    write_jsonl(processed_dir / "mbpp_test.jsonl", code_test)
    write_jsonl(processed_dir / "humaneval_test.jsonl", humaneval_test)

    manifest = {
        "gsm8k_train": len(math_train),
        "gsm8k_test": len(math_test),
        "math500_test": len(math500_test),
        "mbpp_train": len(code_train),
        "mbpp_validation": len(code_val),
        "mbpp_test": len(code_test),
        "humaneval_test": len(humaneval_test),
    }
    write_json(Path(processed_dir) / "manifest.json", manifest)


if __name__ == "__main__":
    main()
