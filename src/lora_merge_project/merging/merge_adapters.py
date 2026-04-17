from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

from lora_merge_project.config import load_config
from lora_merge_project.merging.algorithms import dare_linear_merge, linear_merge, ties_merge
from lora_merge_project.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=["linear", "ties", "dare"], required=True)
    return parser.parse_args()


def load_adapter_state(adapter_dir: Path):
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        return load_file(safetensors_path)
    raise FileNotFoundError(f"Could not find {safetensors_path}")


def copy_adapter_metadata(source_dir: Path, destination_dir: Path) -> None:
    for name in ["adapter_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, destination_dir / name)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    math_dir = checkpoints_dir / "math_adapter"
    code_dir = checkpoints_dir / "code_adapter"
    output_dir = ensure_dir(checkpoints_dir / f"merged_{args.method}")

    math_state = load_adapter_state(math_dir)
    code_state = load_adapter_state(code_dir)
    if set(math_state.keys()) != set(code_state.keys()):
        raise ValueError("Adapter keys do not match; merging requires aligned LoRA checkpoints.")

    shape_mismatches = [
        f"  {key}: math={math_state[key].shape}, code={code_state[key].shape}"
        for key in math_state
        if math_state[key].shape != code_state[key].shape
    ]
    if shape_mismatches:
        raise ValueError(
            "Adapter tensor shape mismatch — ensure both adapters were trained with "
            "the same LoRA rank and target modules:\n" + "\n".join(shape_mismatches)
        )

    merged_state = {}
    for key in math_state.keys():
        tensors = [math_state[key], code_state[key]]
        if args.method == "linear":
            weights = [
                float(config["merging"]["linear"]["weights"]["math"]),
                float(config["merging"]["linear"]["weights"]["code"]),
            ]
            merged_state[key] = linear_merge(tensors, weights=weights)
        elif args.method == "ties":
            merged_state[key] = ties_merge(tensors, density=float(config["merging"]["ties"]["density"]))
        else:
            dare_cfg = config["merging"]["dare"]
            merged_state[key] = dare_linear_merge(
                tensors,
                drop_rate=float(dare_cfg["drop_rate"]),
                weights=None,
                seed=int(dare_cfg["seed"]),
            )

    save_file(merged_state, output_dir / "adapter_model.safetensors")
    copy_adapter_metadata(math_dir, output_dir)
    write_json(
        output_dir / "merge_metadata.json",
        {
            "method": args.method,
            "source_adapters": {
                "math": str(math_dir),
                "code": str(code_dir),
            },
        },
    )


if __name__ == "__main__":
    main()

