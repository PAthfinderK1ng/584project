#!/usr/bin/env bash
# ============================================================
# 清理旧的混乱 conda 环境，然后重建干净的环境到 $SCRATCH
#
# 使用方法（在登录节点执行）：
#   bash scripts/clean_env.sh
# ============================================================

# ── 0. 加载 anaconda module ──────────────────────────────────
module purge
module load Anaconda3/2024.02-1    # 按需改版本：module avail anaconda

echo ""
echo "========== 当前所有 conda 环境 =========="
conda env list
echo "=========================================="
echo ""

# ── 1. 删除旧的混乱环境（按需取消注释，或手动执行）───────────
# 删除 named 环境（在 ~/.conda/envs/ 下）：
#   conda env remove -n <环境名>   例如：
#   conda env remove -n ELEC576
#   conda env remove -n comp584
#   conda env remove -n base_backup

# 删除 prefix 环境（在任意路径下）：
#   conda env remove -p /path/to/env

# 清理 conda 包缓存（通常几个 GB）：
#   conda clean --all -y

echo ""
echo "请手动运行上面注释掉的命令来删除你不需要的环境。"
echo "例如：  conda env remove -n ELEC576"
echo ""
echo "删完后运行：  bash scripts/setup_env.sh"
