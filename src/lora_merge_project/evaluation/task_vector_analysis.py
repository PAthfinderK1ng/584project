from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from safetensors.torch import load_file

from lora_merge_project.config import load_config
from lora_merge_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def flatten_tensor_map(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().float().cpu().flatten() for key, tensor in state.items()}


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = (left.norm() * right.norm()).item()
    if denominator == 0:
        return 0.0
    return float(torch.dot(left, right).item() / denominator)


def sign_conflict_rate(left: torch.Tensor, right: torch.Tensor) -> float:
    active = (left != 0) & (right != 0)
    if int(active.sum()) == 0:
        return 0.0
    conflict = active & (torch.sign(left) != torch.sign(right))
    return float(conflict.sum().item() / active.sum().item())


def magnitude_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.minimum(left.abs(), right.abs()).sum().item()
    denominator = torch.maximum(left.abs(), right.abs()).sum().item()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def analyze_states(math_state: dict[str, torch.Tensor], code_state: dict[str, torch.Tensor]) -> tuple[pd.DataFrame, dict[str, float]]:
    math_flat = flatten_tensor_map(math_state)
    code_flat = flatten_tensor_map(code_state)
    rows = []
    all_math = []
    all_code = []
    for key in sorted(math_flat.keys()):
        math_tensor = math_flat[key]
        code_tensor = code_flat[key]
        rows.append(
            {
                "parameter": key,
                "cosine_similarity": cosine_similarity(math_tensor, code_tensor),
                "sign_conflict_rate": sign_conflict_rate(math_tensor, code_tensor),
                "magnitude_overlap": magnitude_overlap(math_tensor, code_tensor),
                "math_l2_norm": float(math_tensor.norm().item()),
                "code_l2_norm": float(code_tensor.norm().item()),
            }
        )
        all_math.append(math_tensor)
        all_code.append(code_tensor)

    math_global = torch.cat(all_math)
    code_global = torch.cat(all_code)
    summary = {
        "global_cosine_similarity": cosine_similarity(math_global, code_global),
        "global_sign_conflict_rate": sign_conflict_rate(math_global, code_global),
        "global_magnitude_overlap": magnitude_overlap(math_global, code_global),
        "math_global_l2_norm": float(math_global.norm().item()),
        "code_global_l2_norm": float(code_global.norm().item()),
        "num_parameter_tensors": len(rows),
    }
    return pd.DataFrame(rows), summary


def plot_layer_metrics(layer_df: pd.DataFrame, output_path: Path) -> None:
    if layer_df.empty:
        return
    top = layer_df.copy()
    top["short_parameter"] = top["parameter"].str.replace("base_model.model.", "", regex=False)
    top = top.head(20)
    ax = top.plot(
        x="short_parameter",
        y=["cosine_similarity", "sign_conflict_rate", "magnitude_overlap"],
        kind="bar",
        figsize=(14, 6),
    )
    ax.set_title("Adapter Task-Vector Relationship by Parameter Tensor")
    ax.set_ylabel("Score")
    ax.set_ylim(-1, 1)
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    analysis_dir = ensure_dir(config["paths"]["analysis_dir"])

    math_path = checkpoints_dir / "math_adapter" / "adapter_model.safetensors"
    code_path = checkpoints_dir / "code_adapter" / "adapter_model.safetensors"
    math_state = load_file(math_path)
    code_state = load_file(code_path)

    layer_df, summary = analyze_states(math_state, code_state)
    layer_df.to_csv(Path(analysis_dir) / "task_vector_layer_metrics.csv", index=False)
    with (Path(analysis_dir) / "task_vector_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_layer_metrics(layer_df.sort_values("sign_conflict_rate", ascending=False), Path(analysis_dir) / "task_vector_relationships.png")


if __name__ == "__main__":
    main()
