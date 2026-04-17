#!/usr/bin/env bash
# ============================================================
# Submit the full pipeline with SLURM job dependencies.
# Run from the project root:
#   bash scripts/submit_all.sh
#
# Job order:
#   train_math ─┐
#               ├─► eval (merge + evaluate + analyze)
#   train_code ─┘
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # ensure we are at project root
mkdir -p logs

echo "Submitting math training job..."
MATH_JOB=$(sbatch --parsable scripts/slurm_train_math.sh)
echo "  → math job ID: ${MATH_JOB}"

echo "Submitting code training job..."
CODE_JOB=$(sbatch --parsable scripts/slurm_train_code.sh)
echo "  → code job ID: ${CODE_JOB}"

echo "Submitting eval job (depends on ${MATH_JOB} and ${CODE_JOB})..."
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${MATH_JOB}:${CODE_JOB} \
    scripts/slurm_eval.sh)
echo "  → eval job ID: ${EVAL_JOB}"

echo ""
echo "======================================================="
echo " All jobs submitted."
echo "   train_math : ${MATH_JOB}"
echo "   train_code : ${CODE_JOB}"
echo "   eval       : ${EVAL_JOB}  (runs after both finish)"
echo ""
echo " Monitor:  squeue -u \$USER"
echo " Logs:     logs/train_math_${MATH_JOB}.out"
echo "           logs/train_code_${CODE_JOB}.out"
echo "           logs/eval_${EVAL_JOB}.out"
echo "======================================================="
