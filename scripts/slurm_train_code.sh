#!/usr/bin/env bash
#SBATCH --job-name=comp584_train_code
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2             # 2× V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00               # code: ~30-60 min (MBPP 120 samples × 3 epochs)
#SBATCH --output=logs/train_code_%j.out
#SBATCH --error=logs/train_code_%j.err
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

cd "${SLURM_SUBMIT_DIR}"

echo "=== Job info ==="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================"

# ── train code adapter ──────────────────────────────────────
python -m lora_merge_project.training.train_lora \
    --config configs/experiment.yaml \
    --task code

echo "=== Code adapter training complete ==="
echo "Checkpoint: results/checkpoints/code_adapter/"
