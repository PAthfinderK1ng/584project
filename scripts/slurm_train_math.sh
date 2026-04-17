#!/usr/bin/env bash
#SBATCH --job-name=comp584_train_math
#SBATCH --partition=gpu               # Rice NOTS GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2             # 2× V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00               # math: ~3-5 h on 2×V100 with 7473 samples
#SBATCH --output=logs/train_math_%j.out
#SBATCH --error=logs/train_math_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kl212@rice.edu

set -euo pipefail
mkdir -p logs

# ── modules ─────────────────────────────────────────────────
module purge
module load GCC/12.3.0
module load CUDA/12.1.0
module load Anaconda3/2024.02-1

source activate comp584

# ── working dir ─────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"

echo "=== Job info ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# ── step 1: prepare datasets (idempotent) ───────────────────
python -m lora_merge_project.data.prepare_datasets \
    --config configs/experiment.yaml

# ── step 2: train math adapter ──────────────────────────────
python -m lora_merge_project.training.train_lora \
    --config configs/experiment.yaml \
    --task math

echo "=== Math adapter training complete ==="
echo "Checkpoint: results/checkpoints/math_adapter/"
