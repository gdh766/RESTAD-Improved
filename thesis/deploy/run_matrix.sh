#!/usr/bin/env bash
# 正式实验：3 数据集 × 7 变体 × 20 种子 = 420 组，默认 8 个进程并发。
#
# 用法：
#     bash RESTAD/thesis/deploy/run_matrix.sh
#
# 想改规模就设环境变量，例如：
#     WORKERS=6 DATASETS="MSL PSM" SEEDS="$(seq 0 15)" bash RESTAD/thesis/deploy/run_matrix.sh
#
# 显存要求：每个进程约 1.3~1.8 GB。8 并发建议 16 GB 以上显存（4090 / A5000 等）；
# 12 GB 用 6 并发，6 GB 用 3 并发。
# 中途想停：Ctrl+C 或 kill 掉 python 进程即可，已完成的 run 目录都是完整的。
set -euo pipefail

# ---------------------------------------------------------------------------
# 每个进程只用一个 CPU 线程。
# Torch 默认按 CPU 核数开 intra-op 线程；云服务器动辄 100+ 核，于是 N 个并发进程
# 会各自开 100+ 线程互相抢占，实测能把单组实验从十几秒拖到几分钟（GPU 利用率掉到个位数）。
# 我们跑的是小模型，单线程足够，把线程压到 1 之后并发才真正有效。
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RESTAD_TORCH_THREADS=${RESTAD_TORCH_THREADS:-1}

cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

if [ -f "$ROOT/.runtime_python" ]; then
    PY="$(cat "$ROOT/.runtime_python")"
else
    PY="${PYTHON:-python}"
fi

WORKERS=${WORKERS:-8}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"}
DATASETS=${DATASETS:-"MSL PSM SMD"}
VARIANTS=${VARIANTS:-"baseline M1 M2 M3 M5 M1+M2+M5 Full"}

echo "== 解释器  ：$PY"
echo "== 并发数  ：$WORKERS"
echo "== 数据集  ：$DATASETS"
echo "== 变体    ：$VARIANTS"
echo "== 种子    ：$SEEDS"
echo

# shellcheck disable=SC2086
"$PY" RESTAD/thesis/run_matrix.py \
    --workers "$WORKERS" \
    --datasets $DATASETS \
    --variants $VARIANTS \
    --seeds $SEEDS \
    --protocol val \
    --tag full

LATEST=$(ls -dt RESTAD/thesis/results/*_full | head -1)
echo
echo "== 结果目录：$LATEST"

echo
echo "== 配对统计（乘性融合、加平滑、仅重建、仅 RBF 各一张表）"
for SC in mul mul_ema rec_ema rbf_only; do
    echo "--- $SC ---"
    "$PY" RESTAD/thesis/compare_variants.py --results "$LATEST" --scoring "$SC" || true
done

echo
echo "== 生成论文图表"
"$PY" RESTAD/thesis/make_figures.py --results "$LATEST" --out "$LATEST/figures"

echo
echo "== 全部完成。把下面这些带回来即可："
echo "    $LATEST/pairwise_*.md     配对统计结果表（可直接抄进论文）"
echo "    $LATEST/figures/*.png     图"
echo
echo "打包命令（只带表和图，几十 KB）："
echo "    cd $LATEST && tar czf /root/restad_results.tar.gz pairwise_*.md figures/"
