from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from lora_merge_project.config import load_config
from lora_merge_project.training.formatters import (
    build_code_training_messages,
    build_math_training_messages,
)
from lora_merge_project.utils import (
    current_timestamp,
    detect_compute_dtype,
    ensure_dir,
    model_dtype_kwargs,
    read_jsonl,
    seed_everything,
    trainable_parameter_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["math", "code"], required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def dataset_filename(task: str) -> str:
    return {
        "math": "gsm8k_train.jsonl",
        "code": "mbpp_train.jsonl",
    }[task]


def adapter_output_dir(config: dict[str, Any], task: str) -> Path:
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    return checkpoints_dir / f"{task}_adapter"


def format_example(task: str, record: dict[str, Any], tokenizer) -> str:
    if task == "math":
        messages = build_math_training_messages(record["question"], record["solution"])
    else:
        messages = build_code_training_messages(record["prompt"], record["code"])
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize_records(records: list[dict[str, Any]], tokenizer, max_length: int, task: str) -> Dataset:
    dataset = Dataset.from_list(records)

    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = format_example(task, example, tokenizer)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(_tokenize, remove_columns=dataset.column_names)


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padding_length = max_length - len(label)
            padded_labels.append(label + [-100] * padding_length)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_model_and_tokenizer(config: dict[str, Any]):
    model_cfg = config["model"]
    dtype = detect_compute_dtype(model_cfg["compute_dtype"])
    quantization_config = None
    if model_cfg.get("use_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg["quant_type"],
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=model_cfg["double_quant"],
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        quantization_config=quantization_config,
        device_map="auto",
        **model_dtype_kwargs(AutoModelForCausalLM.from_pretrained, dtype),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def compute_warmup_steps(num_examples: int, per_device_batch_size: int, grad_accum_steps: int, num_epochs: float, warmup_ratio: float) -> int:
    num_gpus = max(torch.cuda.device_count(), 1)
    effective_batch_size = per_device_batch_size * num_gpus
    micro_batches = math.ceil(num_examples / max(effective_batch_size, 1))
    optimizer_steps_per_epoch = math.ceil(micro_batches / max(grad_accum_steps, 1))
    total_steps = max(1, math.ceil(optimizer_steps_per_epoch * num_epochs))
    return int(total_steps * warmup_ratio)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["project"]["seed"])

    data_path = Path(config["paths"]["processed_data_dir"]) / dataset_filename(args.task)
    records = read_jsonl(data_path)
    training_cfg = config["training"][args.task]
    max_samples = args.max_samples or training_cfg.get("max_train_samples")
    if max_samples is not None:
        records = records[:max_samples]

    model, tokenizer = build_model_and_tokenizer(config)
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)

    tokenized = tokenize_records(records, tokenizer, config["model"]["max_seq_length"], args.task)
    output_dir = ensure_dir(adapter_output_dir(config, args.task))

    report_to = ["wandb"] if config.get("wandb", {}).get("enabled", False) else []
    run_name = f"{config['project']['name']}-{args.task}"
    compute_dtype = detect_compute_dtype(config["model"]["compute_dtype"])
    warmup_steps = compute_warmup_steps(
        num_examples=len(records),
        per_device_batch_size=training_cfg["per_device_train_batch_size"],
        grad_accum_steps=training_cfg["gradient_accumulation_steps"],
        num_epochs=float(training_cfg["num_train_epochs"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
    )
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": training_cfg["num_train_epochs"],
        "learning_rate": float(training_cfg["learning_rate"]),
        "per_device_train_batch_size": training_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_cfg["gradient_accumulation_steps"],
        "warmup_steps": warmup_steps,
        "weight_decay": float(training_cfg["weight_decay"]),
        "logging_steps": training_cfg["logging_steps"],
        "save_strategy": training_cfg["save_strategy"],
        "report_to": report_to,
        "run_name": run_name,
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = training_cfg["eval_strategy"]
    else:
        training_kwargs["eval_strategy"] = training_cfg["eval_strategy"]
    training_arguments = TrainingArguments(**training_kwargs)

    start = current_timestamp()
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()
    train_runtime = current_timestamp() - start

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    efficiency = trainable_parameter_summary(trainer.model)
    if torch.cuda.is_available():
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_memory_bytes = 0
    metrics = {
        "task": args.task,
        "num_examples": len(records),
        "train_runtime_seconds": train_runtime,
        "peak_memory_bytes": peak_memory_bytes,
        "environment_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        **efficiency,
    }
    write_json(output_dir / "training_metrics.json", metrics)


if __name__ == "__main__":
    main()
