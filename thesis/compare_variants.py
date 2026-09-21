# -*- coding: utf-8 -*-
"""合并多个结果目录，做配对统计，输出论文用的结果表。

为什么用配对比较
----------------
同一随机种子下的 baseline 与变体共享完全相同的数据划分、初始化顺序和训练轮数，
两者的差异比"不同种子之间"的差异小得多。因此比较变体的正确做法是
**同种子配对求差**，而不是把两组独立均值相减——后者会被种子带来的巨大方差淹没。
本脚本对每个种子算 Δ = 变体 - baseline，再对 Δ 序列做检验。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\compare_variants.py ^
        --results RESTAD\\thesis\\results\\20260921-164308_val ^
                  RESTAD\\thesis\\results\\20260921-172057_m5 ^
        --scoring mul_clip_ema0.1
"""

import argparse
import json
import sys
from pathlib import Path

# Windows 控制台默认是 GBK，直接 print 含特殊符号的 Markdown 会抛 UnicodeEncodeError。
# 报告本身已经以 UTF-8 写到文件里，这里只是让屏幕输出不崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                     # noqa: BLE001
    pass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

ANOMALY_RATIO = {"MSL": 1.0, "PSM": 1.0, "SMD": 0.5}
VARIANT_ORDER = ["baseline", "M1", "M2", "M2s", "M3", "M1+M2s", "M5", "M1+M2+M5"]


def minmax(train_values, test_values, clip=True):
    lo, hi = float(np.min(train_values)), float(np.max(train_values))
    span = hi - lo if hi > lo else 1.0
    tr = (train_values - lo) / span
    te = (test_values - lo) / span
    if clip:
        tr, te = np.clip(tr, 0, 1), np.clip(te, 0, 1)
    return tr, te


def ema(scores, alpha):
    """
    一阶指数移动平均：y[t] = alpha * x[t] + (1 - alpha) * y[t-1]，y[-1] = 0。

    用 scipy 的 lfilter 做递推，等价于朴素的 Python 循环但跑在 C 层。
    别小看这一步：SMD 每个序列有 70 万个点，纯 Python 循环算一次要好几秒，
    420 组实验 × 每种评分方式算两遍，光这一个函数就要多花二十多分钟。
    """
    from scipy.signal import lfilter
    a = float(alpha)
    return lfilter([a], [1.0, -(1.0 - a)], np.asarray(scores, dtype=np.float64))


def metrics(train_scores, test_scores, labels, ratio):
    thresh = np.percentile(np.concatenate([train_scores, test_scores]), 100.0 - ratio)
    return {
        "auc": float(roc_auc_score(labels, test_scores)),
        "auc_pr": float(average_precision_score(labels, test_scores)),
        "f1": float(f1_score(labels, (test_scores > thresh).astype(int), zero_division=0)),
    }


def scoring_variants(rec_tr, rec_te, rbf_tr, rbf_te, labels, ratio):
    """返回 {评分方式: 指标}。命名规则见 README/报告。"""
    r_tr, r_te = minmax(rec_tr, rec_te, clip=True)
    s_tr, s_te = minmax(rbf_tr, rbf_te, clip=True)
    mul_tr, mul_te = r_tr * (1.0 - s_tr), r_te * (1.0 - s_te)

    out = {
        # 原作者口径：乘性融合，不做平滑
        "mul": metrics(mul_tr, mul_te, labels, ratio),
        # 推荐方案：乘性融合 + 时序 EMA 平滑 α=0.1
        "mul_ema": metrics(mul_tr, ema(mul_te, 0.1), labels, ratio),
        # 消融对照：只用重建误差 + 平滑，用来回答"RBF 项到底有没有用"
        "rec_ema": metrics(r_tr, ema(r_te, 0.1), labels, ratio),
        # 机制诊断：完全不用重建误差，只看 RBF 相似度本身
        "rbf_only": metrics(1.0 - s_tr, 1.0 - s_te, labels, ratio),
    }
    return out


def collect(result_dirs, use_cache=True):
    """
    遍历所有结果目录，返回 {(dataset, variant, seed): 各评分方式指标}。

    带缓存：一次遍历要把 420 组实验的逐点分数全部读进来算指标，
    其中 SMD 每条序列 70 万个点，算一次 roc_auc / AP 就要一两秒，
    整体要几分钟。而画图、换评分方式、调阈值这些后续操作都要用同一张表，
    所以算完直接存成 json，下次直接读，避免重复算。
    """
    cache_path = Path(result_dirs[0]) / "metrics_table.json"
    if use_cache and cache_path.exists():
        try:
            recs = json.loads(cache_path.read_text(encoding="utf-8"))
            table = {(r["dataset"], r["variant"], r["seed"]): r["scores"] for r in recs}
            print(f"[collect] 命中缓存 {cache_path.name}，{len(table)} 条记录")
            return table
        except (json.JSONDecodeError, KeyError, OSError):
            print("[collect] 缓存损坏，重新计算")

    table = {}
    for results_dir in result_dirs:
        for run_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir()):
            npz_path, meta_path = run_dir / "scores.npz", run_dir / "metrics.json"
            if not (npz_path.exists() and meta_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if "error" in meta:
                continue
            key = (meta["dataset"], meta["variant"], meta["seed"])
            if key in table:
                continue                      # 同一组合出现在多个目录时以先出现的为准
            data = np.load(npz_path)
            table[key] = scoring_variants(
                data["rec_train"], data["rec_test"],
                data["rbf_train"], data["rbf_test"], data["labels"],
                ANOMALY_RATIO[meta["dataset"]])

    if use_cache:
        recs = [{"dataset": k[0], "variant": k[1], "seed": k[2], "scores": v}
                for k, v in table.items()]
        cache_path.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        print(f"[collect] 已缓存 {len(table)} 条记录 -> {cache_path.name}")
    return table


ALL_SCORINGS = ["mul", "mul_ema", "rec_ema", "rbf_only"]


def build_report(table, scoring):
    """基于已算好的指标表，生成某种评分方式下的 Markdown 报告和结构化行。"""
    datasets = sorted({k[0] for k in table})
    variants = [v for v in VARIANT_ORDER if any(k[1] == v for k in table)]
    lines, rows = [], []

    lines.append(f"# RESTAD 消融结果（评分方式：{scoring}）")
    lines.append("")
    lines.append("配对比较：同一随机种子下 `Δ = 变体 - baseline`，"
                 "对 Δ 序列做双侧配对 t 检验（n = 种子数）。")
    lines.append("")

    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| 变体 | n | AUC-ROC | AUC-PR | F1 | ΔAUC-ROC | ΔAUC-PR | p(AUC-ROC) |")
        lines.append("|---|---|---|---|---|---|---|---|")

        base = {k[2]: table[k][scoring] for k in table
                if k[0] == dataset and k[1] == "baseline"}
        for variant in variants:
            sub = {k[2]: table[k][scoring] for k in table
                   if k[0] == dataset and k[1] == variant}
            if not sub:
                continue
            aucs = np.array([sub[s]["auc"] for s in sorted(sub)])
            aps = np.array([sub[s]["auc_pr"] for s in sorted(sub)])
            f1s = np.array([sub[s]["f1"] for s in sorted(sub)])
            d_auc = d_ap = np.array([np.nan])
            p_val = np.nan
            if variant != "baseline" and base:
                shared = sorted(set(sub) & set(base))
                if len(shared) >= 2:
                    d_auc = np.array([sub[s]["auc"] - base[s]["auc"] for s in shared])
                    d_ap = np.array([sub[s]["auc_pr"] - base[s]["auc_pr"] for s in shared])
                    _t, p_val = stats.ttest_rel([sub[s]["auc"] for s in shared],
                                                [base[s]["auc"] for s in shared])
                    p_val = float(p_val)
            lines.append(
                f"| {variant} | {len(sub)} | {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f} | "
                f"{aps.mean():.4f} ± {aps.std(ddof=1):.4f} | {f1s.mean():.4f} | "
                f"{'—' if variant == 'baseline' else f'{np.mean(d_auc):+.4f}'} | "
                f"{'—' if variant == 'baseline' else f'{np.mean(d_ap):+.4f}'} | "
                f"{'—' if variant == 'baseline' else f'{p_val:.3f}'} |")
            rows.append(dict(dataset=dataset, variant=variant, scoring=scoring,
                             n=len(sub), auc_mean=float(aucs.mean()),
                             auc_std=float(aucs.std(ddof=1)),
                             ap_mean=float(aps.mean()), ap_std=float(aps.std(ddof=1)),
                             f1_mean=float(f1s.mean()),
                             d_auc=float(np.mean(d_auc)) if variant != "baseline" else None,
                             d_ap=float(np.mean(d_ap)) if variant != "baseline" else None,
                             p_auc=None if variant == "baseline" else p_val))
        lines.append("")

    return "\n".join(lines), rows


def main() -> int:
    parser = argparse.ArgumentParser(description="变体配对比较")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--scoring", default="mul_ema",
                        help="mul / mul_ema / rec_ema / rbf_only，"
                             "或 all（一次遍历出全部四张表），也可用逗号分隔多个")
    parser.add_argument("--out", default=None, help="只在指定单个 scoring 时生效")
    parser.add_argument("--no-cache", action="store_true", help="忽略已缓存的指标表，强制重算")
    args = parser.parse_args()

    if args.scoring.strip().lower() == "all":
        scorings = list(ALL_SCORINGS)
    else:
        scorings = [s.strip() for s in args.scoring.split(",") if s.strip()]
    bad = [s for s in scorings if s not in ALL_SCORINGS]
    if bad:
        print(f"未知的评分方式：{bad}，可选 {ALL_SCORINGS} 或 all")
        return 2

    dirs = [Path(p) if Path(p).is_absolute() else Path.cwd() / p for p in args.results]

    # 只算一次：collect 内部会把四种评分方式全部算好，后面复用同一张表。
    # 之前是每种评分方式各调一次脚本，等于把 420 组实验的指标算了四遍。
    table = collect(dirs, use_cache=not args.no_cache)
    if not table:
        print("没有收集到结果")
        return 1

    for scoring in scorings:
        md, rows = build_report(table, scoring)
        if args.out and len(scorings) == 1:
            out_md = Path(args.out)
        else:
            out_md = dirs[0] / f"pairwise_{scoring}.md"
        out_md.write_text(md, encoding="utf-8")
        (out_md.with_suffix(".json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(md)
        print(f"已写出：{out_md}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
