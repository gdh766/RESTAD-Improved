# -*- coding: utf-8 -*-
"""用已保存的权重做"重叠窗口"重推理，验证推理期改动是否有效（不需要重新训练）。

背景
----
原作者用 `create_windows(data, 100, step)` 且 step == window_size == 100，
即**不重叠**切窗。后果是：
  (1) 每个时间点只被一个窗口覆盖，拿到的分数完全来自那一个窗口，
      相邻窗口之间没有任何过渡，分数序列呈块状跳变；
  (2) 序列尾部不足一个窗长的点被直接丢弃（MSL 测试集 73729 个点里最后 29 个
      没有参与评价）。
把推理步长降到 step << 100 后，每个点会落在多个窗口里，把这些窗口给出的
分数取平均，等价于对分数做了一次滑动平均，块状跳变被抹平。

本脚本只做**推理**，训练部分沿用 run_variants.py 已跑出的 `model.pth`，
所以能几分钟内验证完整个实验矩阵。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\eval_reinfer.py ^
        --results RESTAD\\thesis\\results\\20260921-164308_val --step 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

THESIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THESIS_DIR.parent
RESTAD_DIR = REPO_DIR / "restad"
# 记下调用时的目录：后面要把工作目录切到 restad/，命令行里的相对路径
# 必须基于调用者的目录来解析，不能用切换后的目录。
CALLER_CWD = Path.cwd()
sys.path.insert(0, str(RESTAD_DIR))
os.chdir(RESTAD_DIR)

from Utils import get_data  # noqa: E402
from Transformer_Model import Transformer_RBF  # noqa: E402


def point_scores(model, series, cfg, step, device, want, batch_size=256):
    """重叠滑窗推理，把所有覆盖到同一时间点的窗口分数取平均。

    series: (L, d) 的 numpy 数组
    want:   {"rec", "rbf"} 的子集，决定返回哪些逐点分数
    返回 dict，每个值是长度 L 的数组；尾部未被任何窗口覆盖的点填 0。
    """
    L, _ = series.shape
    W = int(cfg.dataset.window_size)
    if L < W:
        raise ValueError(f"序列长度 {L} 小于窗口长度 {W}")

    n_windows = (L - W) // step + 1
    valid_len = (n_windows - 1) * step + W

    acc = {k: np.zeros(L, dtype=np.float64) for k in want}
    cnt = np.zeros(L, dtype=np.float64)

    rec_error_type = str(cfg.model.get("rec_error_type", "mse")).lower()

    buf, starts = [], []
    def flush():
        if not buf:
            return
        x = torch.tensor(np.stack(buf), dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs, _, rbf_out = model(x)
        if "rec" in want:
            if rec_error_type == "l2":
                per_point = torch.norm(x - outputs, dim=2)
            else:
                per_point = torch.mean((x - outputs) ** 2, dim=2)
            per_point = per_point.cpu().numpy()
        if "rbf" in want:
            rbf_point = torch.mean(rbf_out, dim=2).cpu().numpy()
        for j, s in enumerate(starts):
            if "rec" in want:
                acc["rec"][s:s + W] += per_point[j]
            if "rbf" in want:
                acc["rbf"][s:s + W] += rbf_point[j]
            cnt[s:s + W] += 1.0
        buf.clear()
        starts.clear()

    model.eval()
    for i in range(n_windows):
        s = i * step
        buf.append(series[s:s + W])
        starts.append(s)
        if len(buf) >= batch_size:
            flush()
    flush()

    covered = cnt > 0
    out = {}
    for k in want:
        vals = np.zeros(L, dtype=np.float64)
        vals[covered] = acc[k][covered] / cnt[covered]
        out[k] = vals
    out["_valid_len"] = valid_len
    return out


def minmax(train_values, test_values, clip=True):
    lo, hi = float(np.min(train_values)), float(np.max(train_values))
    span = hi - lo if hi > lo else 1.0
    tr = np.clip((train_values - lo) / span, 0.0, 1.0) if clip else (train_values - lo) / span
    te = np.clip((test_values - lo) / span, 0.0, 1.0) if clip else (test_values - lo) / span
    return tr, te


def ema(scores, alpha):
    out = np.empty_like(scores, dtype=np.float64)
    acc = 0.0
    for i, s in enumerate(scores):
        acc = alpha * float(s) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def evaluate(train_scores, test_scores, labels, ratio):
    thresh = np.percentile(np.concatenate([train_scores, test_scores]), 100.0 - ratio)
    pred = (test_scores > thresh).astype(int)
    return {
        "auc": float(roc_auc_score(labels, test_scores)),
        "auc_pr": float(average_precision_score(labels, test_scores)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
    }


ANOMALY_RATIO = {"MSL": 1.0, "PSM": 1.0, "SMD": 0.5}


def main() -> int:
    parser = argparse.ArgumentParser(description="重叠窗口重推理评测")
    parser.add_argument("--results", required=True)
    parser.add_argument("--step", type=int, default=10, help="推理滑窗步长（原作为 100）")
    parser.add_argument("--train-step", type=int, default=None,
                        help="训练序列的滑窗步长，默认与 --step 相同")
    parser.add_argument("--ema-alpha", type=float, default=0.0,
                        help="可选：对测试分数再做一次 EMA，0 表示不做")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    args = parser.parse_args()

    train_step = args.train_step if args.train_step is not None else args.step

    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = CALLER_CWD / results_dir

    run_dirs = sorted(p for p in results_dir.iterdir()
                      if p.is_dir() and (p / "model.pth").exists())
    if args.limit:
        run_dirs = run_dirs[:args.limit]

    rows = []
    for i, run_dir in enumerate(run_dirs, 1):
        cfg = OmegaConf.load(run_dir / "config.yaml")
        device = cfg.device
        model = Transformer_RBF(cfg).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pth", map_location=device))

        (x_train, _), (x_test, y_test) = get_data(cfg)
        tr = point_scores(model, x_train, cfg, train_step, device, {"rec", "rbf"})
        te = point_scores(model, x_test, cfg, args.step, device, {"rec", "rbf"})

        n = min(te["_valid_len"], len(y_test))
        labels = np.asarray(y_test[:n]).astype(int)
        rec_tr, rbf_tr = tr["rec"][:tr["_valid_len"]], tr["rbf"][:tr["_valid_len"]]
        rec_te, rbf_te = te["rec"][:n], te["rbf"][:n]

        ratio = ANOMALY_RATIO[cfg.dataset.name]
        r_tr, r_te = minmax(rec_tr, rec_te)
        s_tr, s_te = minmax(rbf_tr, rbf_te)
        prod_tr, prod_te = r_tr * (1.0 - s_tr), r_te * (1.0 - s_te)

        if args.ema_alpha > 0:
            prod_te = ema(prod_te, args.ema_alpha)
            r_te_ = ema(r_te, args.ema_alpha)
        else:
            r_te_ = r_te

        m = {
            "dataset": cfg.dataset.name,
            "variant": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["variant"],
            "seed": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["seed"],
            "step": args.step,
            "ema_alpha": args.ema_alpha,
            "mul_overlap": evaluate(prod_tr, prod_te, labels, ratio),
            "rec_overlap": evaluate(r_tr, r_te_, labels, ratio),
            "n_points": int(n),
            "tail_dropped": int(len(y_test) - n),
        }
        rows.append(m)
        print(f"[{i}/{len(run_dirs)}] {m['dataset']:<4} {m['variant']:<10} s{m['seed']}  "
              f"mul AUC={m['mul_overlap']['auc']:.4f} AP={m['mul_overlap']['auc_pr']:.4f}  "
              f"rec AUC={m['rec_overlap']['auc']:.4f} AP={m['rec_overlap']['auc_pr']:.4f}",
              flush=True)

    suffix = f"step{args.step}" + (f"_ema{args.ema_alpha}" if args.ema_alpha > 0 else "")
    out_path = results_dir / f"reinfer_{suffix}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 汇总（各变体 5 种子均值）===")
    print(f"{'数据集':<6}{'变体':<12}{'重叠-乘性 AUC':>16}{'AP':>10}{'重叠-仅重建 AUC':>18}{'AP':>10}")
    for dataset in sorted({r["dataset"] for r in rows}):
        for variant in ["baseline", "M1", "M2", "M2s", "M3", "M1+M2s"]:
            sub = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            if not sub:
                continue
            a1 = np.mean([r["mul_overlap"]["auc"] for r in sub])
            p1 = np.mean([r["mul_overlap"]["auc_pr"] for r in sub])
            a2 = np.mean([r["rec_overlap"]["auc"] for r in sub])
            p2 = np.mean([r["rec_overlap"]["auc_pr"] for r in sub])
            print(f"{dataset:<6}{variant:<12}{a1:>16.4f}{p1:>10.4f}{a2:>18.4f}{p2:>10.4f}")

    print(f"\n已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
