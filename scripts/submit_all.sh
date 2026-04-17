#!/usr/bin/env bash
# ============================================================
# 一键提交完整流水线（带 job 依赖）
# 在登录节点从项目根目录执行：
#   bash scripts/submit_all.sh
# ============================================================
set -euo pipefail

# 确保从项目根目录执行
cd "${SCRATCH}/584project"

echo "提交 math 训练 job..."
MATH_JOB=$(sbatch --parsable scripts/slurm_train_math.sh)
echo "  → math job ID: ${MATH_JOB}"

echo "提交 code 训练 job..."
CODE_JOB=$(sbatch --parsable scripts/slurm_train_code.sh)
echo "  → code job ID: ${CODE_JOB}"

echo "提交 eval job（依赖 ${MATH_JOB} 和 ${CODE_JOB} 均成功）..."
EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${MATH_JOB}:${CODE_JOB} \
    scripts/slurm_eval.sh)
echo "  → eval job ID: ${EVAL_JOB}"

echo ""
echo "======================================================="
echo " 全部已提交"
echo "   train_math : ${MATH_JOB}   (~6h)"
echo "   train_code : ${CODE_JOB}   (~2h)"
echo "   eval       : ${EVAL_JOB}   (~8h，两个训练完后自动开始)"
echo ""
echo " 监控：     squeue -u \$USER"
echo " 实时日志：  tail -f comp584_train_math_${MATH_JOB}.out"
echo " 取消 job：  scancel ${EVAL_JOB}"
echo "======================================================="
