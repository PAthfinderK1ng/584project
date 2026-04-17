from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_merge_project.utils import detect_compute_dtype, ensure_dir, model_dtype_kwargs, read_jsonl


def condition_to_adapter_path(config: dict[str, Any], condition: str) -> Path | None:
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    mapping = {
        "base": None,
        "math_adapter": checkpoints_dir / "math_adapter",
        "code_adapter": checkpoints_dir / "code_adapter",
        "merged_linear": checkpoints_dir / "merged_linear",
        "merged_ties": checkpoints_dir / "merged_ties",
        "merged_dare": checkpoints_dir / "merged_dare",
    }
    if condition not in mapping:
        raise ValueError(f"Unknown condition: {condition}")
    return mapping[condition]


def load_tokenizer(config: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_inference(config: dict[str, Any], condition: str):
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
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        quantization_config=quantization_config,
        device_map="auto",
        **model_dtype_kwargs(AutoModelForCausalLM.from_pretrained, dtype),
    )
    adapter_path = condition_to_adapter_path(config, condition)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.0, top_p: float = 1.0, do_sample: bool = False) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output = model.generate(
            **encoded,
            **generation_kwargs,
        )
    new_tokens = output[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_code_block(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def load_processed_split(config: dict[str, Any], filename: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(Path(config["paths"]["processed_data_dir"]) / filename)
    if max_samples is not None:
        return records[:max_samples]
    return records


def write_metrics(output_dir: Path, stem: str, metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    with (output_dir / f"{stem}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    with (output_dir / f"{stem}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_python_test(program: str, timeout_seconds: int) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "candidate.py"
        temp_path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if completed.returncode == 0:
        return True, ""
    stderr = (completed.stderr or "").strip()
    stdout = (completed.stdout or "").strip()
    message = stderr if stderr else stdout
    return False, message[:500]
