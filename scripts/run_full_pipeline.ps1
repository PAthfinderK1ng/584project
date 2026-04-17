$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$config = "configs/experiment.yaml"

python -m lora_merge_project.data.prepare_datasets --config $config

python -m lora_merge_project.training.train_lora --config $config --task math
python -m lora_merge_project.training.train_lora --config $config --task code

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
  python -m lora_merge_project.evaluation.math_eval --config $config --condition $condition
  python -m lora_merge_project.evaluation.code_eval --config $config --condition $condition
}

python -m lora_merge_project.evaluation.summarize_results --config $config
python -m lora_merge_project.evaluation.task_vector_analysis --config $config
