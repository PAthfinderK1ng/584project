#!/usr/bin/env bash
#SBATCH --job-name=comp584_pipeline
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:59:59
#SBATCH --output=comp584_pipeline_%j.out
#SBATCH --error=comp584_pipeline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kl212@rice.edu

set -euo pipefail

# ── 路径（$SCRATCH 没定义时自动 fallback）────────────────────
SCRATCH="${SCRATCH:-/scratch/${USER}}"
PROJECT_DIR="${SCRATCH}/584project"
ENV_DIR="${SCRATCH}/envs/comp584"
CONFIG="configs/experiment.yaml"

# ── 加载模块 ─────────────────────────────────────────────────
module purge
module load GCC/12.3.0
module load CUDA/12.4.1
module load Miniforge3/25.3.0-3

source activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

# ── 打印环境信息 ─────────────────────────────────────────────
echo "================================================="
echo " COMP 584 Full Pipeline"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Node    : ${SLURMD_NODENAME}"
echo " GPUs    : ${CUDA_VISIBLE_DEVICES:-unset}"
echo " Start   : $(date)"
echo "================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ── Step 1: 准备数据集 ───────────────────────────────────────
echo "[$(date +%H:%M:%S)] === Step 1/5: 准备数据集 ==="
python -m lora_merge_project.data.prepare_datasets --config $CONFIG

# ── Step 2: 训练 math adapter ────────────────────────────────
echo "[$(date +%H:%M:%S)] === Step 2/5: 训练 math adapter ==="
python -m lora_merge_project.training.train_lora \
    --config $CONFIG --task math

# ── Step 3: 训练 code adapter ────────────────────────────────
echo "[$(date +%H:%M:%S)] === Step 3/5: 训练 code adapter ==="
python -m lora_merge_project.training.train_lora \
    --config $CONFIG --task code

# ── Step 4: 合并 adapters ────────────────────────────────────
echo "[$(date +%H:%M:%S)] === Step 4/5: 合并 adapters ==="
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method linear
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method ties
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method dare

# ── Step 5: 评估所有 conditions ──────────────────────────────
echo "[$(date +%H:%M:%S)] === Step 5/5: 评估 ==="
CONDITIONS=(base math_adapter code_adapter merged_linear merged_ties merged_dare)
for condition in "${CONDITIONS[@]}"; do
    echo "[$(date +%H:%M:%S)] -- math eval: ${condition}"
    python -m lora_merge_project.evaluation.math_eval \
        --config $CONFIG --condition "${condition}"
    echo "[$(date +%H:%M:%S)] -- code eval: ${condition}"
    python -m lora_merge_project.evaluation.code_eval \
        --config $CONFIG --condition "${condition}"
done

# ── 汇总 ─────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] === 汇总结果 ==="
python -m lora_merge_project.evaluation.summarize_results  --config $CONFIG
python -m lora_merge_project.evaluation.task_vector_analysis --config $CONFIG

echo ""
echo "================================================="
echo " 全部完成！结束时间：$(date)"
echo " 结果目录：${PROJECT_DIR}/results/analysis/"
ls "${PROJECT_DIR}/results/analysis/"
echo "================================================="
