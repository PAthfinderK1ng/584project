# LoRA Adapter Merging for Multi-Task Reasoning and Code Generation

This repository turns the proposal into a runnable research project for COMP 584.
It implements the full experiment pipeline described in the PDF:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- PEFT method: LoRA with 4-bit NF4 quantization
- Tasks: mathematical reasoning and code generation
- Adapters: one math adapter and one code adapter
- Merge methods: Linear, TIES, DARE
- Evaluation: GSM8K, MATH-500, HumanEval, efficiency tracking, delta-performance analysis

## Project layout

```text
configs/
  experiment.yaml
docs/
  final_report_template.md
scripts/
  run_full_pipeline.ps1
  run_smoke_test.ps1
src/lora_merge_project/
  config.py
  utils.py
  data/prepare_datasets.py
  training/formatters.py
  training/train_lora.py
  merging/algorithms.py
  merging/merge_adapters.py
  evaluation/common.py
  evaluation/math_eval.py
  evaluation/code_eval.py
  evaluation/summarize_results.py
  evaluation/task_vector_analysis.py
results/
```

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

If you want experiment tracking:

```powershell
wandb login
```

## Default workflow

Activate the conda environment you want to use before running the scripts below:

```powershell
conda activate ELEC576
cd "D:\COMP 584"
```

### 1. Prepare datasets

```powershell
python -m lora_merge_project.data.prepare_datasets --config configs/experiment.yaml
```

This writes normalized JSONL files into `data/processed/`.

### 2. Train the task-specific adapters

```powershell
python -m lora_merge_project.training.train_lora --config configs/experiment.yaml --task math
python -m lora_merge_project.training.train_lora --config configs/experiment.yaml --task code
```

Outputs are written to:

- `results/checkpoints/math_adapter/`
- `results/checkpoints/code_adapter/`

### 3. Merge the adapters

```powershell
python -m lora_merge_project.merging.merge_adapters --config configs/experiment.yaml --method linear
python -m lora_merge_project.merging.merge_adapters --config configs/experiment.yaml --method ties
python -m lora_merge_project.merging.merge_adapters --config configs/experiment.yaml --method dare
```

Merged adapters are written to:

- `results/checkpoints/merged_linear/`
- `results/checkpoints/merged_ties/`
- `results/checkpoints/merged_dare/`

### 4. Evaluate all conditions

Math benchmarks:

```powershell
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition base
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition math_adapter
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition code_adapter
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition merged_linear
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition merged_ties
python -m lora_merge_project.evaluation.math_eval --config configs/experiment.yaml --condition merged_dare
```

Code benchmark:

```powershell
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition base
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition math_adapter
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition code_adapter
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition merged_linear
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition merged_ties
python -m lora_merge_project.evaluation.code_eval --config configs/experiment.yaml --condition merged_dare
```

### 5. Summarize and plot

```powershell
python -m lora_merge_project.evaluation.summarize_results --config configs/experiment.yaml
python -m lora_merge_project.evaluation.task_vector_analysis --config configs/experiment.yaml
```

This produces:

- `results/analysis/summary_table.csv`
- `results/analysis/delta_table.csv`
- `results/analysis/training_efficiency.csv`
- `results/analysis/math_performance.png`
- `results/analysis/code_performance.png`
- `results/analysis/task_vector_summary.json`
- `results/analysis/task_vector_relationships.png`

## One-command runs

Full experiment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_pipeline.ps1
```

Smoke test with small sample counts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_smoke_test.ps1
```

The smoke test uses `configs/smoke_test.yaml`, which swaps in `Qwen/Qwen2.5-1.5B-Instruct`
and small sample limits so you can debug the pipeline before launching the full 7B run.

## Notes and assumptions

- The proposal names `MBPP train`; this repo uses the `mbpp` sanitized split for training because it exposes train/validation/test partitions that fit the experiment design.
- The math evaluation uses answer extraction plus symbolic normalization for common numeric and algebraic forms.
- HumanEval execution runs generated Python code in a subprocess with a timeout. Use a controlled environment.
- The included TIES and DARE implementations are adapter-space implementations tailored for LoRA state dicts, which is sufficient for the proposed comparison.

## Suggested deliverables for submission

- Source code in this repository
- `results/analysis/*` figures and tables after running experiments
- A short written report using `docs/final_report_template.md`
- Optional: W&B dashboard screenshots and training logs
