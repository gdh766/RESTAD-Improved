# -*- coding: utf-8 -*-
"""按论文需要生成图表。复用 compare_variants 的收集与评分逻辑，避免两处口径不一致。

中文字体处理
------------
云服务器（AutoDL 等）的镜像里通常**一个中文字体都没有**，
matplotlib 找不到字体时中文会渲染成方框（豆腐块），图直接废掉。
本脚本的做法：
  1. 先把 `~/.fonts/` 和本脚本同级的 `fonts/` 目录里的字体文件注册进 matplotlib
     （部署时把 simhei.ttf 之类的字体丢进这两个目录即可）；
  2. 再检查系统里有没有可用的中文字体；
  3. 有的话用中文标签，没有就**自动退回英文标签**——宁可全英文，也不要满屏方框。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\make_figures.py ^
        --results RESTAD\\thesis\\results\\<dir1> RESTAD\\thesis\\results\\<dir2> ^
        --out RESTAD\\thesis\\results\\figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_variants import collect  # noqa: E402

VARIANT_ORDER = ["baseline", "M1", "M2", "M2s", "M3", "M1+M2s", "M5", "M1+M2+M5", "Full"]

# 中文字体优先级；都没有就退回英文
CJK_FONTS = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
             "Noto Sans CJK SC", "Source Han Sans SC", "AR PL UMing CN"]

LABELS = {
    "zh": {
        "baseline": "baseline（原论文）", "M1": "M1 门控残差", "M2": "M2 中心正则",
        "M2s": "M2s 强正则", "M3": "M3 L2 重建误差", "M1+M2s": "M1+M2s",
        "M5": "M5 密度对齐", "M1+M2+M5": "M1+M2+M5（配套方案）", "Full": "Full 全部修改",
    },
    "en": {
        "baseline": "baseline (paper)", "M1": "M1 gated residual", "M2": "M2 center reg.",
        "M2s": "M2s strong reg.", "M3": "M3 L2 rec. error", "M1+M2s": "M1+M2s",
        "M5": "M5 density align", "M1+M2+M5": "M1+M2+M5 (full)", "Full": "Full",
    },
}
DOMAIN = {
    "zh": {"MSL": "MSL（航天器遥测）", "PSM": "PSM（服务器指标）", "SMD": "SMD（服务器机器）"},
    "en": {"MSL": "MSL (spacecraft telemetry)", "PSM": "PSM (server metrics)",
           "SMD": "SMD (server machines)"},
}
TEXT = {
    "zh": dict(
        auc_title="各变体的 AUC-ROC（乘性融合 εr×εs + 时序 EMA 平滑）",
        auc_ylabel="AUC-ROC", legend_base="baseline",
        # 注意：这里用 ASCII 连字符 "-"，不要用 U+2212 减号。
        # SimHei 等中文字体没有 U+2212 的字形，会出现 "Glyph missing" 警告并缺字。
        delta_title="同种子配对差值", delta_xlabel="随机种子",
        delta_ylabel="ΔAUC-ROC（变体 - baseline）",
        mech_title="机制诊断：只用 RBF 相似度（不用重建误差）能多大程度区分异常",
        mech_ylabel="RBF 分支单独 AUC-ROC", mech_ref="随机水平 0.5",
        abl_title="评分方式消融：RBF 项与平滑各自贡献多少",
        s_mul="εr×εs（原论文）", s_mul_ema="εr×εs + EMA",
        s_rec_ema="仅 εr + EMA", s_rbf="仅 RBF 相似度",
    ),
    "en": dict(
        auc_title="AUC-ROC per variant (product fusion εr×εs + temporal EMA)",
        auc_ylabel="AUC-ROC", legend_base="baseline",
        delta_title="Paired per-seed difference", delta_xlabel="random seed",
        delta_ylabel="ΔAUC-ROC (variant - baseline)",
        mech_title="Mechanism check: RBF similarity alone (no reconstruction error)",
        mech_ylabel="AUC-ROC of RBF branch alone", mech_ref="chance level 0.5",
        abl_title="Scoring ablation: contribution of RBF term vs. smoothing",
        s_mul="εr×εs (paper)", s_mul_ema="εr×εs + EMA",
        s_rec_ema="εr only + EMA", s_rbf="RBF similarity only",
    ),
}


def setup_font():
    """注册本地字体并挑一个能显示中文的；返回 (plt, 语言)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 1) 把随项目一起放进来的字体注册进去
    for d in (Path(__file__).resolve().parent / "fonts", Path.home() / ".fonts",
              Path.home() / ".local/share/fonts"):
        if d.is_dir():
            for f in list(d.glob("*.ttf")) + list(d.glob("*.ttc")) + list(d.glob("*.otf")):
                try:
                    font_manager.fontManager.addfont(str(f))
                except Exception:                        # noqa: BLE001
                    pass

    # 2) 看现在有没有中文字体
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in CJK_FONTS if n in available), None)

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[make_figures] 使用中文字体：{chosen}")
        return plt, "zh"

    print("[make_figures] 未找到中文字体，图内文字自动改用英文（避免出现方框）")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, "en"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", choices=["auto", "zh", "en"], default="auto",
                        help="图内文字语言；auto = 有中文字体就用中文")
    args = parser.parse_args()

    dirs = [Path(p) if Path(p).is_absolute() else Path.cwd() / p for p in args.results]
    table = collect(dirs)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt, auto_lang = setup_font()
    lang = auto_lang if args.lang == "auto" else args.lang
    L, D, T = LABELS[lang], DOMAIN[lang], TEXT[lang]

    datasets = sorted({k[0] for k in table})
    variants = [v for v in VARIANT_ORDER if any(k[1] == v for k in table)]

    # ---------------------------------------------------------------
    # 图 1：各变体的 AUC-ROC（误差棒 = 种子标准差）
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        means, stds = [], []
        for variant in variants:
            vals = [table[k]["mul_ema"]["auc"] for k in table
                    if k[0] == dataset and k[1] == variant]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        colors = ["#9aa0a6" if v == "baseline" else "#1a73e8" for v in variants]
        ax.bar(range(len(variants)), means, yerr=stds, capsize=3, color=colors)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([L.get(v, v) for v in variants], rotation=35, ha="right", fontsize=8)
        if "baseline" in variants:
            ax.axhline(means[variants.index("baseline")], color="crimson", ls="--",
                       lw=1.2, label=T["legend_base"])
            ax.legend(fontsize=8)
        ax.set_title(D.get(dataset, dataset))
        ax.set_ylabel(T["auc_ylabel"])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(T["auc_title"], y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_variant_auc.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 图 2：同种子配对差值
    # ---------------------------------------------------------------
    pairs = [(d, v) for d, v in [("MSL", "M3"), ("MSL", "M1"), ("PSM", "M1+M2+M5"), ("PSM", "M1")]
             if d in datasets and v in variants]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 4.2), squeeze=False)
        for ax, (dataset, variant) in zip(axes[0], pairs):
            base = {k[2]: table[k]["mul_ema"]["auc"] for k in table
                    if k[0] == dataset and k[1] == "baseline"}
            cur = {k[2]: table[k]["mul_ema"]["auc"] for k in table
                   if k[0] == dataset and k[1] == variant}
            seeds = sorted(set(base) & set(cur))
            deltas = [cur[s] - base[s] for s in seeds]
            ax.axhline(0, color="gray", lw=1)
            ax.bar([str(s) for s in seeds], deltas,
                   color=["#1a73e8" if d > 0 else "#d93025" for d in deltas])
            ax.set_title(f"{dataset}: {L.get(variant, variant)}\n"
                         f"mean ΔAUC {np.mean(deltas):+.4f}", fontsize=9)
            ax.set_xlabel(T["delta_xlabel"])
            ax.set_ylabel(T["delta_ylabel"])
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_paired_delta.png", dpi=200)
        plt.close(fig)

    # ---------------------------------------------------------------
    # 图 3：机制诊断 —— RBF 分支单独作为异常分数
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        means = []
        for variant in variants:
            vals = [table[k]["rbf_only"]["auc"] for k in table
                    if k[0] == dataset and k[1] == variant]
            means.append(np.mean(vals) if vals else np.nan)
        colors = ["#9aa0a6" if v == "baseline" else "#0f9d58" for v in variants]
        ax.bar(range(len(variants)), means, color=colors)
        ax.axhline(0.5, color="crimson", ls="--", lw=1.2, label=T["mech_ref"])
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([L.get(v, v) for v in variants], rotation=35, ha="right", fontsize=8)
        ax.set_title(D.get(dataset, dataset))
        ax.set_ylabel(T["mech_ylabel"])
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(T["mech_title"], y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_rbf_diagnostic.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------
    # 图 4：评分方式消融
    # ---------------------------------------------------------------
    scorings = [("mul", T["s_mul"]), ("mul_ema", T["s_mul_ema"]),
                ("rec_ema", T["s_rec_ema"]), ("rbf_only", T["s_rbf"])]
    show_variants = [v for v in ["baseline", "M1+M2+M5", "M3", "M1"] if v in variants]
    if show_variants:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.2), squeeze=False)
        width = 0.8 / len(scorings)
        for ax, dataset in zip(axes[0], datasets):
            xs = np.arange(len(show_variants))
            for i, (sc, name) in enumerate(scorings):
                means = []
                for variant in show_variants:
                    vals = [table[k][sc]["auc"] for k in table
                            if k[0] == dataset and k[1] == variant]
                    means.append(np.mean(vals) if vals else np.nan)
                ax.bar(xs + i * width - 0.4 + width / 2, means, width, label=name)
            ax.set_xticks(xs)
            ax.set_xticklabels([L.get(v, v) for v in show_variants],
                               rotation=20, ha="right", fontsize=8)
            ax.set_title(D.get(dataset, dataset))
            ax.set_ylabel(T["auc_ylabel"])
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].legend(fontsize=7, ncol=2)
        fig.suptitle(T["abl_title"], y=0.98)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_scoring_ablation.png", dpi=200)
        plt.close(fig)

    print(f"图表已写出到 {out_dir}")
    for p in sorted(out_dir.glob("*.png")):
        print("  -", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
