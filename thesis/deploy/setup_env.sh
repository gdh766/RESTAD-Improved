#!/usr/bin/env bash
# 在服务器上准备 RESTAD 毕设实验环境。
#
# 用法（在解压后的包根目录，也就是含 RESTAD/ 的那一层）：
#     bash RESTAD/thesis/deploy/setup_env.sh
#
# 说明：
#   - 会自动挑一个**已经装好 PyTorch** 的解释器。云镜像（AutoDL 等）把 PyTorch
#     放在 /root/miniconda3 的 base 环境里，但非交互式 SSH 登录时 PATH 里可能没有
#     conda，`python` 会落到系统自带的解释器上；不探测就装会白下几个 GB。
#   - 选中的解释器会写进包根目录的 .runtime_python，后续 smoke.sh / run_matrix.sh
#     自动读取，避免出现"装到了 A 解释器、跑的时候用了 B 解释器"这种问题。
set -euo pipefail

# deploy -> thesis -> RESTAD -> 包根，所以要往上三层
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
echo "== 工作目录：$ROOT"

# ---------------- 1. 选解释器 ----------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in /root/miniconda3/bin/python /opt/conda/bin/python python3 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1; then
            PY="$cand"
            echo "== 自动选中已带 PyTorch 的解释器：$PY"
            break
        fi
    done
fi
PY="${PY:-python}"
echo "== Python：$("$PY" -V 2>&1)  （$PY）"

# ---------------- 2. 必要时安装 PyTorch ----------------
if "$PY" -c "import torch" 2>/dev/null; then
    echo "== 已检测到 PyTorch，跳过安装"
else
    echo "== 未检测到 PyTorch，开始安装（CUDA 12.1 版）"
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

# ---------------- 3. 其余依赖 ----------------
# requirements.txt 里刻意不写 torch / numpy：镜像已预装，
# 写了反而会把能用的版本升级掉（torch 2.1.2 与 numpy 2.x 不兼容）。
echo "== 安装其余依赖"
"$PY" -m pip install -r RESTAD/thesis/requirements.txt

# ---------------- 4. 记住解释器 ----------------
echo "$PY" > "$ROOT/.runtime_python"
echo "== 已记录解释器到 $ROOT/.runtime_python"

# ---------------- 5. 自检 ----------------
echo "== 环境自检"
"$PY" - <<'PYEOF'
import torch, numpy, sklearn, scipy, hydra, omegaconf
print("torch         ", torch.__version__)
print("numpy         ", numpy.__version__)
print("sklearn       ", sklearn.__version__)
print("scipy         ", scipy.__version__)
print("cuda available", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("！CUDA 不可用。如果用的是「无卡模式」开机，这是正常的，切到有卡模式即可；")
    print("  否则检查是不是选错解释器了。")
    raise SystemExit(0)

props = torch.cuda.get_device_properties(0)
print("gpu           ", props.name)
print("vram (GB)     ", round(props.total_memory / 1024 ** 3, 1))

# 只报告 is_available() 是不够的：numpy 2.x 与 torch 2.1.x 的 ABI 不兼容时，
# 轻微一点的症状是 import 就报错，隐蔽一点的症状是真正做运算时才崩。
# 所以这里真的跑一次 GPU 矩阵乘，把问题在使用前暴露出来。
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b
torch.cuda.synchronize()
print("gpu matmul     OK  (sum=%.4f)" % float(c.sum()))

# 顺带验证 numpy 与 torch 的互操作，这一步最容易踩 numpy 2.x 的坑
n = numpy.arange(8, dtype="float32")
t = torch.from_numpy(n).cuda()
print("torch<->numpy  OK  (%s)" % tuple(t.shape))
PYEOF

# ---------------- 6. 数据集检查 ----------------
echo
echo "== 检查三个数据集"
missing=0
for d in MSL PSM SMD; do
    f="RESTAD/restad/datasets/$d/${d}_train.npy"
    if [ -f "$f" ]; then echo "  $d OK"; else echo "  $d 缺失：$f"; missing=1; fi
done
if [ "$missing" = "1" ]; then
    echo "！有数据集缺失，正常流程不该出现，请检查打包是否完整。"
fi

echo
echo "== 完成。下一步："
echo "   bash RESTAD/thesis/deploy/smoke.sh        # 2 分钟冒烟检查（可选）"
echo "   bash RESTAD/thesis/deploy/run_matrix.sh   # 正式实验"
