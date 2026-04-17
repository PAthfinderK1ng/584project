#!/usr/bin/env bash
#SBATCH --job-name=comp584_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00               # 6 conditions × 3 benchmarks，约 6-8 h
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kl212@rice.edu

set -euo pipefail

PROJECT_DIR="${SCRATCH}/584project"
ENV_DIR="${SCRATCH}/envs/comp584"
CONFIG="configs/experiment.yaml"

module purge
module load GCC/12.3.0
module load CUDA/12.4.1
module load Anaconda3/2024.02-1

source activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

echo "=== Job info ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# ── 合并 adapters（CPU，很快）────────────────────────────────
echo "--- 合并 adapters ---"
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method linear
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method ties
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method dare

# ── 评估 6 个 condition ──────────────────────────────────────
CONDITIONS=(base math_adapter code_adapter merged_linear merged_ties merged_dare)

for condition in "${CONDITIONS[@]}"; do
    echo "--- [math eval] ${condition} ---"
    python -m lora_merge_project.evaluation.math_eval \
        --config $CONFIG --condition "${condition}"

    echo "--- [code eval] ${condition} ---"
    python -m lora_merge_project.evaluation.code_eval \
        --config $CONFIG --condition "${condition}"
done

# ── 汇总结果 + task vector 分析 ─────────────────────────────
echo "--- 生成结果表格和图表 ---"
python -m lora_merge_project.evaluation.summarize_results  --config $CONFIG
python -m lora_merge_project.evaluation.task_vector_analysis --config $CONFIG

echo ""
echo "=== 全部完成！结果文件 ==="
ls "${PROJECT_DIR}/results/analysis/"
