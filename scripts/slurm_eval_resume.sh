#!/usr/bin/env bash
#SBATCH --job-name=comp584_eval
#SBATCH --partition=commons
#SBATCH --reservation=classroom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=comp584_eval_%j.out
#SBATCH --error=comp584_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kl212@rice.edu

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch/${USER}}"
PROJECT_DIR="${SCRATCH}/584project"
ENV_DIR="${SCRATCH}/envs/comp584"
CONFIG="configs/experiment.yaml"
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

module purge
module load GCC/12.3.0
module load CUDA/12.4.1
module load Miniforge3/25.3.0-3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

cd "${PROJECT_DIR}"

echo "================================================="
echo " COMP 584 Eval Resume"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Node    : ${SLURMD_NODENAME}"
echo " GPUs    : ${CUDA_VISIBLE_DEVICES:-unset}"
echo " Start   : $(date)"
echo "================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

CONDITIONS=(code_adapter merged_linear merged_ties merged_dare)
for condition in "${CONDITIONS[@]}"; do
    echo "[$(date +%H:%M:%S)] -- math eval: ${condition}"
    python -m lora_merge_project.evaluation.math_eval \
        --config $CONFIG --condition "${condition}"
    echo "[$(date +%H:%M:%S)] -- code eval: ${condition}"
    python -m lora_merge_project.evaluation.code_eval \
        --config $CONFIG --condition "${condition}"
done

echo "[$(date +%H:%M:%S)] === 汇总结果 ==="
python -m lora_merge_project.evaluation.summarize_results --config $CONFIG
python -m lora_merge_project.evaluation.task_vector_analysis --config $CONFIG

echo ""
echo "================================================="
echo " 全部完成！结束时间：$(date)"
echo "================================================="
