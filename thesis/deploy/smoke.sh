#!/usr/bin/env bash
# 正式跑之前先做 2 分钟冒烟检查，确认代码、数据、GPU、解释器都正常。
#
# 用法：
#     bash RESTAD/thesis/deploy/smoke.sh
#
# 它只跑 2 组实验（各 3 轮训练），成功会打印两行 "εr×εs F1=..."，
# 并对刚生成的 2 个权重做一次重叠窗口重推理。
set -euo pipefail

cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"

# 优先用 setup_env.sh 记住的那个解释器
if [ -f "$ROOT/.runtime_python" ]; then
    PY="$(cat "$ROOT/.runtime_python")"
else
    PY="${PYTHON:-python}"
fi
echo "== 使用解释器：$PY"

echo
echo "== 冒烟 1/2：MSL baseline + M1，各 1 个种子、3 轮训练"
"$PY" RESTAD/thesis/run_variants.py \
    --datasets MSL \
    --variants baseline M1 \
    --seeds 0 \
    --epochs 3 \
    --protocol val \
    --tag smoke

LATEST=$(ls -dt RESTAD/thesis/results/*_smoke | head -1)

echo
echo "== 冒烟 2/2：对刚生成的 2 个权重做重叠窗口重推理"
"$PY" RESTAD/thesis/eval_reinfer.py --results "$LATEST" --step 10 --limit 2

echo
echo "== 冒烟通过，可以执行：bash RESTAD/thesis/deploy/run_matrix.sh"
