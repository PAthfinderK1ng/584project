#!/usr/bin/env bash
#SBATCH --job-name=comp584_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00               # 6 conditions × 3 benchmarks, ~6-8 h total
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
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

CONFIG="configs/experiment.yaml"

# ── step 3: merge adapters (CPU, fast) ──────────────────────
echo "--- Merging adapters ---"
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method linear
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method ties
python -m lora_merge_project.merging.merge_adapters --config $CONFIG --method dare

# ── step 4: evaluate all 6 conditions ───────────────────────
CONDITIONS=(base math_adapter code_adapter merged_linear merged_ties merged_dare)

for condition in "${CONDITIONS[@]}"; do
    echo "--- Evaluating math: ${condition} ---"
    python -m lora_merge_project.evaluation.math_eval \
        --config $CONFIG --condition "${condition}"

    echo "--- Evaluating code: ${condition} ---"
    python -m lora_merge_project.evaluation.code_eval \
        --config $CONFIG --condition "${condition}"
done

# ── step 5: summarize + task vector analysis ─────────────────
echo "--- Summarizing results ---"
python -m lora_merge_project.evaluation.summarize_results  --config $CONFIG
python -m lora_merge_project.evaluation.task_vector_analysis --config $CONFIG

echo ""
echo "=== All done. Results in results/analysis/ ==="
ls results/analysis/
