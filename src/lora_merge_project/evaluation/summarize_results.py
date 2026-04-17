from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import pandas as pd

from lora_merge_project.config import load_config
from lora_merge_project.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def collect_metrics(evaluations_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_file in evaluations_dir.glob("*/*_metrics.json"):
        rows.append(pd.read_json(metrics_file, typ="series").to_dict())
    return pd.DataFrame(rows)


def collect_training_metrics(checkpoints_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_file in checkpoints_dir.glob("*/training_metrics.json"):
        with metrics_file.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        condition = metrics_file.parent.name
        row["condition"] = condition
        adapter_file = metrics_file.parent / "adapter_model.safetensors"
        row["adapter_size_mb"] = round(adapter_file.stat().st_size / (1024 * 1024), 3) if adapter_file.exists() else None
        rows.append(row)
    return pd.DataFrame(rows)


def compute_delta_table(summary: pd.DataFrame, math_reference: str, code_reference: str) -> pd.DataFrame:
    rows = []
    math_columns = ["gsm8k_test", "math500_test"]
    code_columns = ["humaneval_test"]
    for _, row in summary.iterrows():
        entry = {"condition": row["condition"]}
        for column in math_columns:
            reference = summary.loc[summary["condition"] == math_reference, column]
            if not reference.empty and pd.notna(row.get(column)):
                entry[f"delta_{column}"] = row[column] - reference.iloc[0]
        for column in code_columns:
            reference = summary.loc[summary["condition"] == code_reference, column]
            if not reference.empty and pd.notna(row.get(column)):
                entry[f"delta_{column}"] = row[column] - reference.iloc[0]
        rows.append(entry)
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, columns: list[str], output_path: Path, title: str) -> None:
    available_columns = [column for column in columns if column in summary.columns]
    if not available_columns:
        return
    ax = summary.set_index("condition")[available_columns].plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_efficiency(efficiency: pd.DataFrame, output_path: Path) -> None:
    if efficiency.empty:
        return
    plot_data = efficiency.set_index("condition")[["train_runtime_seconds", "peak_memory_gb"]]
    ax = plot_data.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Training Efficiency")
    ax.set_ylabel("Value")
    ax.legend(["Train Runtime (s)", "Peak Memory (GB)"], loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    evaluations_dir = Path(config["paths"]["evaluations_dir"])
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    analysis_dir = ensure_dir(config["paths"]["analysis_dir"])

    metrics = collect_metrics(evaluations_dir)
    if metrics.empty:
        raise ValueError("No evaluation metrics found. Run the evaluation scripts first.")

    metrics["value"] = metrics.apply(
        lambda row: row.get("accuracy") if pd.notna(row.get("accuracy")) else row.get("pass_at_1"),
        axis=1,
    )
    summary = metrics.pivot_table(index="condition", columns="dataset", values="value", aggfunc="first").reset_index()
    summary.to_csv(Path(analysis_dir) / "summary_table.csv", index=False)

    delta = compute_delta_table(
        summary,
        math_reference=config["analysis"]["math_reference_condition"],
        code_reference=config["analysis"]["code_reference_condition"],
    )
    delta.to_csv(Path(analysis_dir) / "delta_table.csv", index=False)

    plot_metric(summary, ["gsm8k_test", "math500_test"], Path(analysis_dir) / "math_performance.png", "Math Benchmark Performance")
    plot_metric(summary, ["humaneval_test"], Path(analysis_dir) / "code_performance.png", "HumanEval pass@1")

    efficiency = collect_training_metrics(checkpoints_dir)
    if not efficiency.empty:
        efficiency["peak_memory_gb"] = efficiency["peak_memory_bytes"] / (1024 ** 3)
        selected_columns = [
            "condition",
            "task",
            "num_examples",
            "trainable_parameters",
            "total_parameters",
            "train_runtime_seconds",
            "peak_memory_gb",
            "adapter_size_mb",
        ]
        efficiency[selected_columns].sort_values("condition").to_csv(Path(analysis_dir) / "training_efficiency.csv", index=False)
        plot_efficiency(efficiency, Path(analysis_dir) / "training_efficiency.png")


if __name__ == "__main__":
    main()
