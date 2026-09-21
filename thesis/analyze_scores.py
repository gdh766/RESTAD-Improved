# -*- coding: utf-8 -*-
"""离线评分策略分析：直接在 run_variants.py 已保存的逐点分数上试各种评分/聚合方式。

为什么要有这个脚本
------------------
训练一次要几十秒，但"怎么把逐点分数合成异常分数"这一步完全可以在已有的
`scores.npz` 上离线重算。所以先用它快速筛掉没用的想法，只把有希望的方案
再拿回训练循环里验证。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\analyze_scores.py ^
        --results RESTAD\\thesis\\results\\20260921-164308_val
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# 各数据集用于定阈值的异常比例（见 configs/dataset/*.yaml 的 anormly_ratio）
ANOMALY_RATIO = {"MSL": 1.0, "PSM": 1.0, "SMD": 0.5}


def minmax(train_values, test_values, clip=True):
    lo, hi = float(np.min(train_values)), float(np.max(train_values))
    span = hi - lo if hi > lo else 1.0
    tr = (train_values - lo) / span
    te = (test_values - lo) / span
    if clip:
        tr = np.clip(tr, 0.0, 1.0)
        te = np.clip(te, 0.0, 1.0)
    return tr, te


def ema(scores, alpha):
    """
    一阶指数移动平均：z[t] = alpha * s[t] + (1 - alpha) * z[t-1]。

    用 lfilter 做递推，等价于朴素循环但跑在 C 层；SMD 单条序列 70 万点，
    纯 Python 循环会明显拖慢整批分析。
    """
    from scipy.signal import lfilter
    a = float(alpha)
    return lfilter([a], [1.0, -(1.0 - a)], np.asarray(scores, dtype=np.float64))


def metrics_with_ratio_threshold(train_scores, test_scores, labels, ratio):
    combined = np.concatenate([train_scores, test_scores])
    thresh = np.percentile(combined, 100.0 - ratio)
    pred = (test_scores > thresh).astype(int)
    return {
        "auc": float(roc_auc_score(labels, test_scores)),
        "auc_pr": float(average_precision_score(labels, test_scores)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "threshold": float(thresh),
    }


def all_strategies(rec_tr, rec_te, rbf_tr, rbf_te, labels, ratio):
    """返回 {策略名: 指标 dict}。"""
    out = {}

    # ---- 各种归一化组合下的乘性融合 ----
    for tag, clip in (("noclip", False), ("clip", True)):
        r_tr, r_te = minmax(rec_tr, rec_te, clip=clip)
        s_tr, s_te = minmax(rbf_tr, rbf_te, clip=clip)
        base_tr = r_tr * (1.0 - s_tr)
        base_te = r_te * (1.0 - s_te)
        out[f"mul_{tag}"] = metrics_with_ratio_threshold(base_tr, base_te, labels, ratio)

        # ---- 时间聚合（EMA）----
        for alpha in (0.1, 0.2, 0.3, 0.5):
            out[f"mul_{tag}_ema{alpha}"] = metrics_with_ratio_threshold(
                base_tr, ema(base_te, alpha), labels, ratio)

    # ---- 仅重建误差 + 平滑（对照：RBF 项到底有没有用）----
    r_tr, r_te = minmax(rec_tr, rec_te, clip=True)
    out["rec_clip"] = metrics_with_ratio_threshold(r_tr, r_te, labels, ratio)
    for alpha in (0.1, 0.2, 0.3, 0.5):
        out[f"rec_clip_ema{alpha}"] = metrics_with_ratio_threshold(
            r_tr, ema(r_te, alpha), labels, ratio)

    # ---- 加性融合 ----
    s_tr, s_te = minmax(rbf_tr, rbf_te, clip=True)
    out["add_clip"] = metrics_with_ratio_threshold(
        r_tr + (1.0 - s_tr), r_te + (1.0 - s_te), labels, ratio)

    # ---- 诊断项：完全不用重建误差，只看 RBF 相似度本身的判别力 ----
    # 这一行是判断"RBF 分支到底有没有学到东西"最直接的证据。
    # 如果它接近 0.5，说明 εs 就是在瞎猜，那么乘性融合里的 (1-εs) 因子
    # 只是在给重建误差乘一个与异常无关的随机系数。
    out["rbf_only"] = metrics_with_ratio_threshold(
        1.0 - s_tr, 1.0 - s_te, labels, ratio)
    out["rbf_only_ema0.1"] = metrics_with_ratio_threshold(
        1.0 - s_tr, ema(1.0 - s_te, 0.1), labels, ratio)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评分策略分析")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None, help="输出 json 路径，默认写在结果目录下")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir

    rows = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        npz_path = run_dir / "scores.npz"
        meta_path = run_dir / "metrics.json"
        if not (npz_path.exists() and meta_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "error" in meta:
            continue
        data = np.load(npz_path)
        ratio = ANOMALY_RATIO[meta["dataset"]]
        strat = all_strategies(
            data["rec_train"], data["rec_test"],
            data["rbf_train"], data["rbf_test"],
            data["labels"], ratio)
        for name, m in strat.items():
            rows.append(dict(dataset=meta["dataset"], variant=meta["variant"],
                             seed=meta["seed"], strategy=name, **m))

    if not rows:
        print("没有找到可分析的结果")
        return 1

    out_path = Path(args.out) if args.out else results_dir / "scoring_analysis.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印聚合表：按 数据集 × 评分策略 × 变体 ----
    datasets = sorted({r["dataset"] for r in rows})
    variants = sorted({r["variant"] for r in rows},
                      key=lambda v: ["baseline", "M1", "M2", "M2s", "M3", "M1+M2s",
                                     "M5", "M1+M2+M5"].index(v)
                      if v in ["baseline", "M1", "M2", "M2s", "M3", "M1+M2s",
                               "M5", "M1+M2+M5"] else 99)
    strategies = ["mul_noclip", "mul_clip", "mul_clip_ema0.1", "mul_clip_ema0.2",
                  "mul_clip_ema0.3", "mul_clip_ema0.5", "add_clip",
                  "rec_clip", "rec_clip_ema0.1", "rec_clip_ema0.2", "rec_clip_ema0.3",
                  "rbf_only", "rbf_only_ema0.1"]

    for dataset in datasets:
        print(f"\n===== {dataset} =====")
        print(f"{'策略':<20}" + "".join(f"{v:>16}" for v in variants))
        for strat in strategies:
            cells = []
            for variant in variants:
                sub = [r for r in rows if r["dataset"] == dataset
                       and r["variant"] == variant and r["strategy"] == strat]
                if not sub:
                    cells.append(f"{'—':>16}")
                    continue
                auc = np.mean([r["auc"] for r in sub])
                ap = np.mean([r["auc_pr"] for r in sub])
                cells.append(f"{auc:.4f}/{ap:.3f}".rjust(16))
            print(f"{strat:<20}" + "".join(cells))

    print(f"\n已写出：{out_path}")
    print("单元格格式：AUC-ROC / AUC-PR（各 5 个种子取均值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
