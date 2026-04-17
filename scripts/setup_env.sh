#!/usr/bin/env bash
# ============================================================
# One-time environment setup — run this ONCE on a login node
# before submitting any SLURM jobs.
#
# Usage:
#   bash scripts/setup_env.sh
# ============================================================
set -euo pipefail

# ── 1. Load system modules ───────────────────────────────────
module purge
module load GCC/12.3.0
module load CUDA/12.1.0
module load Anaconda3/2024.02-1   # adjust to whatever is on your cluster

# ── 2. Create conda environment ──────────────────────────────
CONDA_ENV="comp584"

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "[setup] conda env '${CONDA_ENV}' already exists — skipping creation"
else
    conda create -y -n "${CONDA_ENV}" python=3.11
fi

source activate "${CONDA_ENV}"

# ── 3. Install PyTorch with CUDA 12.1 ────────────────────────
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 4. Install project dependencies ─────────────────────────
pip install -e .

echo ""
echo "======================================================="
echo " Setup complete. Conda env: ${CONDA_ENV}"
echo " Run 'conda activate ${CONDA_ENV}' before any script."
echo "======================================================="
