$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$config = "configs/smoke_test.yaml"

python -m lora_merge_project.data.prepare_datasets --config $config --max-math-train 64 --max-math-test 32 --max-code-train 64 --max-code-test 16

python -m lora_merge_project.training.train_lora --config $config --task math --max-samples 64
python -m lora_merge_project.training.train_lora --config $config --task code --max-samples 64

python -m lora_merge_project.merging.merge_adapters --config $config --method linear
python -m lora_merge_project.merging.merge_adapters --config $config --method ties
python -m lora_merge_project.merging.merge_adapters --config $config --method dare

$conditions = @(
  "base",
  "math_adapter",
  "code_adapter",
  "merged_linear",
  "merged_ties",
  "merged_dare"
)

foreach ($condition in $conditions) {
  python -m lora_merge_project.evaluation.math_eval --config $config --condition $condition --max-samples 16
  python -m lora_merge_project.evaluation.code_eval --config $config --condition $condition --max-samples 8
}

python -m lora_merge_project.evaluation.summarize_results --config $config
python -m lora_merge_project.evaluation.task_vector_analysis --config $config
