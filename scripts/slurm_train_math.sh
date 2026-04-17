#!/usr/bin/env bash
#SBATCH --job-name=comp584_train_math
#SBATCH --partition=gpu               # sinfo 查分区名，可能是 nots / gpu / v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2             # 2× V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00               # math: ~3-5 h on 2×V100, 7473 samples × 2 epochs
#SBATCH --output=%x_%j.out            # 输出到提交目录
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kl212@rice.edu

set -euo pipefail

# ── 路径 ────────────────────────────────────────────────────
PROJECT_DIR="${SCRATCH}/584project"
ENV_DIR="${SCRATCH}/envs/comp584"

# ── 加载模块 ─────────────────────────────────────────────────
module purge
module load GCC/12.3.0
module load CUDA/12.4.1
module load Miniforge3/25.3.0-3

source activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

echo "=== Job info ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Project  : ${PROJECT_DIR}"
echo "================"

# ── 准备数据集（幂等，已存在则跳过）─────────────────────────
python -m lora_merge_project.data.prepare_datasets \
    --config configs/experiment.yaml

# ── 训练 math adapter ────────────────────────────────────────
python -m lora_merge_project.training.train_lora \
    --config configs/experiment.yaml \
    --task math

echo "=== Math adapter 训练完成 ==="
echo "Checkpoint: ${PROJECT_DIR}/results/checkpoints/math_adapter/"
