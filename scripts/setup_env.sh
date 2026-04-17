#!/usr/bin/env bash
# ============================================================
# 一次性环境配置 —— 在登录节点执行，ENV 和项目都放在 $SCRATCH
#
# 使用方法：
#   bash scripts/setup_env.sh
# ============================================================
set -euo pipefail

# ── 0. 检查 $SCRATCH ─────────────────────────────────────────
if [[ -z "${SCRATCH:-}" ]]; then
    echo "[ERROR] \$SCRATCH 未定义。请先确认集群变量："
    echo "        echo \$SCRATCH"
    echo "        如果不存在，改用：export SCRATCH=/scratch/\$USER"
    exit 1
fi

PROJECT_DIR="${SCRATCH}/584project"
ENV_DIR="${SCRATCH}/envs/comp584"

echo "[setup] SCRATCH      = ${SCRATCH}"
echo "[setup] PROJECT_DIR  = ${PROJECT_DIR}"
echo "[setup] ENV_DIR      = ${ENV_DIR}"

# ── 1. 加载系统模块 ──────────────────────────────────────────
module purge
module load GCC/12.3.0           # module avail GCC   查可用版本
module load CUDA/12.1.0          # module avail cuda  查可用版本
module load Anaconda3/2024.02-1  # module avail anaconda

# ── 2. 克隆或更新项目代码到 $SCRATCH ─────────────────────────
if [[ -d "${PROJECT_DIR}/.git" ]]; then
    echo "[setup] 项目已存在，执行 git pull..."
    cd "${PROJECT_DIR}"
    git pull
else
    echo "[setup] 克隆项目到 ${PROJECT_DIR}..."
    git clone git@github.com:PAthfinderK1ng/584project.git "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
fi

# ── 3. 创建 conda 环境到 $SCRATCH（避免 home 配额）──────────
if [[ -d "${ENV_DIR}" ]]; then
    echo "[setup] conda env 已存在：${ENV_DIR}"
    echo "[setup] 如需重建，先运行：  conda env remove -p ${ENV_DIR}"
else
    echo "[setup] 创建 conda env 到 ${ENV_DIR}..."
    conda create -y -p "${ENV_DIR}" python=3.11
fi

# ── 4. 激活环境并安装依赖 ─────────────────────────────────────
source activate "${ENV_DIR}"

# PyTorch（CUDA 12.1）
pip install torch==2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 项目依赖
pip install -e .

echo ""
echo "======================================================="
echo " 配置完成！"
echo ""
echo " 以后每次登录手动激活：                                "
echo "   module load GCC/12.3.0 CUDA/12.1.0 Anaconda3/2024.02-1"
echo "   conda activate ${ENV_DIR}"
echo ""
echo " 项目路径：${PROJECT_DIR}"
echo "======================================================="
