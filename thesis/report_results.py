# -*- coding: utf-8 -*-
"""把 run_variants.py 的实验结果汇总成表格、CSV 和图。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\report_results.py ^
        --results RESTAD\\thesis\\results\\<时间戳>_main

输出（写在同一个结果目录下）：
    summary.md          按数据集/变体聚合的均值±标准差表（Markdown）
    summary.csv         同样内容的 CSV，便于粘进论文
    fig_variant_compare.png   各变体在 AUC / AP 上的对比柱状图
    fig_per_seed.png          逐种子散点，观察方差
"""

import argparse
import json
from pathlib import Path

import numpy as np

VARIANTS = ["baseline", "M1", "M2", "M3", "M1+M2+M3"]
DATASETS = ["PSM", "MSL", "SMD"]
METRIC_KEYS = [
    ("simrec_mul", "f1", "F1"),
    ("simrec_mul", "auc", "AUC-ROC"),
    ("simrec_mul", "auc_pr", "AUC-PR"),
]
SCORINGS = [
    ("rec_only", "仅重建误差 εr"),
    ("simrec_mul", "乘性融合 εr×εs（论文主结果）"),
    ("simrec_add", "加性融合 εr+εs"),
]


def load_rows(results_dir: Path):
    path = results_dir / "metrics_all.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行 run_variants.py")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if "error" not in r]


def aggregate(rows, dataset, variant, scoring, metric):
    vals = [r[scoring][metric] for r in rows
            if r["dataset"] == dataset and r["variant"] == variant]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def build_markdown(rows):
    lines = ["# RESTAD 消融实验结果", "",
             "协议：沿用原作者流程（窗口 100、不重叠；阈值取训练+测试分数的分位数；",
             "不使用 point adjustment）。每个 (数据集, 变体) 重复 3 个随机种子，",
             "表中为均值 ± 样本标准差。", ""]

    for scoring, scoring_name in SCORINGS:
        lines.append(f"## 评分方式：{scoring_name}")
        lines.append("")
        header = "| 数据集 | 变体 | n | F1 | AUC-ROC | AUC-PR |"
        sep = "|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for dataset in DATASETS:
            for variant in VARIANTS:
                cells = []
                for _, metric, _ in METRIC_KEYS:
                    agg = aggregate(rows, dataset, variant, scoring, metric)
                    cells.append("—" if agg is None else f"{agg[0]:.4f} ± {agg[1]:.4f}")
                n = sum(1 for r in rows if r["dataset"] == dataset and r["variant"] == variant)
                if n == 0:
                    continue
                lines.append(f"| {dataset} | {variant} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- 相对基线的增量（论文最关心的一栏）----
    lines.append("## 变体相对基线的增量（乘性融合 εr×εs）")
    lines.append("")
    lines.append("| 数据集 | 变体 | Δ AUC-ROC | Δ AUC-PR | Δ F1 |")
    lines.append("|---|---|---|---|---|")
    for dataset in DATASETS:
        for variant in VARIANTS:
            if variant == "baseline":
                continue
            base_auc = aggregate(rows, dataset, "baseline", "simrec_mul", "auc")
            base_ap = aggregate(rows, dataset, "baseline", "simrec_mul", "auc_pr")
            base_f1 = aggregate(rows, dataset, "baseline", "simrec_mul", "f1")
            cur_auc = aggregate(rows, dataset, variant, "simrec_mul", "auc")
            cur_ap = aggregate(rows, dataset, variant, "simrec_mul", "auc_pr")
            cur_f1 = aggregate(rows, dataset, variant, "simrec_mul", "f1")
            if None in (base_auc, base_ap, base_f1, cur_auc, cur_ap, cur_f1):
                continue
            lines.append(
                f"| {dataset} | {variant} | {cur_auc[0]-base_auc[0]:+.4f} | "
                f"{cur_ap[0]-base_ap[0]:+.4f} | {cur_f1[0]-base_f1[0]:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def build_csv(rows, out_path: Path):
    lines = ["dataset,variant,scoring,seed,f1,auc,auc_pr"]
    for r in rows:
        for scoring, _ in SCORINGS:
            m = r[scoring]
            lines.append(f"{r['dataset']},{r['variant']},{scoring},{r['seed']},"
                         f"{m['f1']:.6f},{m['auc']:.6f},{m['auc_pr']:.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- 图 1：变体对比（AUC-ROC 与 AUC-PR 两个子图）----
    datasets = [d for d in DATASETS if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (scoring, metric, label) in zip(
            axes, [("simrec_mul", "auc", "AUC-ROC"), ("simrec_mul", "auc_pr", "AUC-PR")]):
        width = 0.15
        xs = np.arange(len(datasets))
        for i, variant in enumerate(VARIANTS):
            means, errs = [], []
            for d in datasets:
                agg = aggregate(rows, d, variant, scoring, metric)
                means.append(agg[0] if agg else np.nan)
                errs.append(agg[1] if agg else 0.0)
            ax.bar(xs + i * width, means, width, yerr=errs, capsize=2.5, label=variant)
        ax.set_xticks(xs + width * (len(VARIANTS) - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(label)
        ax.set_title(f"{label}（乘性融合 εr×εs）")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_variant_compare.png", dpi=200)
    plt.close(fig)

    # ---- 图 2：逐种子散点，看方差 ----
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.2), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for i, variant in enumerate(VARIANTS):
            sub = [r for r in rows if r["dataset"] == d and r["variant"] == variant]
            if not sub:
                continue
            ys = [r["simrec_mul"]["auc"] for r in sub]
            ax.scatter([i] * len(ys), ys, s=28, zorder=3)
            ax.hlines(np.mean(ys), i - 0.25, i + 0.25, color="crimson", zorder=4)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(d)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_per_seed.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 RESTAD 消融实验结果")
    parser.add_argument("--results", required=True, help="run_variants.py 生成的结果目录")
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir
    rows = load_rows(results_dir)
    if not rows:
        print("metrics_all.json 里没有成功的实验记录")
        return 1

    md = build_markdown(rows)
    (results_dir / "summary.md").write_text(md, encoding="utf-8")
    build_csv(rows, results_dir / "summary.csv")
    make_figures(rows, results_dir)

    print(md)
    print(f"\n已写出：{results_dir}\\summary.md / summary.csv / fig_variant_compare.png / fig_per_seed.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
