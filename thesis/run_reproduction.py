# -*- coding: utf-8 -*-
"""复现实验：直接用作者提供的预训练权重推理，逐项对齐论文 Table 1 的 RESTAD (R) 行。

和 run_variants.py 的区别
-------------------------
run_variants.py 里的 `baseline` 是**从零随机初始化训练**的对照，用来做消融；
本脚本不训练，直接加载仓库里 `restad/trained_models/` 的作者权重做推理和评分，
这才是"复现论文数值"该有的做法。

注意：三个数据集的 RBF 中心数不同（MSL 128 / PSM 32 / SMD 256），必须按数据集
设置 model.rbf_dim，否则加载权重会 size mismatch。

用法（工作区根目录）：
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\run_reproduction.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

THESIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THESIS_DIR.parent
RESTAD_DIR = REPO_DIR / "restad"
CALLER_CWD = Path.cwd()

sys.path.insert(0, str(RESTAD_DIR))
os.chdir(RESTAD_DIR)

from Utils import (  # noqa: E402
    get_data,
    generate_loaders,
    set_seed,
    calculate_reconstruction_errors,
    calculate_rbf_scores,
    evaluate_rec,
    evaluate_RBFrec,
    evaluate_RBFrec_Addition,
)
from stages_training import train_with_rbf  # noqa: E402

# 各数据集的 RBF 中心数（必须与作者权重匹配）
RBF_DIM = {"MSL": 128, "PSM": 32, "SMD": 256}

# 论文 Table 1 里 RESTAD (R) 那一行的值，用于逐项对照
PAPER_TABLE1 = {
    "SMD": {"f1": 0.23, "auc": 0.78, "auc_pr": 0.23},
    "MSL": {"f1": 0.07, "auc": 0.68, "auc_pr": 0.18},
    "PSM": {"f1": 0.15, "auc": 0.79, "auc_pr": 0.59},
}


def extract(result_dict):
    e = result_dict[("ratio", False)]
    return {
        "f1": float(e.get("RBFrec f1-score", e.get("f1-score"))),
        "precision": float(e.get("RBFrec precision", e.get("precision"))),
        "recall": float(e.get("RBFrec recall", e.get("recall"))),
        "auc": float(e.get("RBFrec AUC", e.get("AUC"))),
        "auc_pr": float(e.get("RBFrec AUC-PR", e.get("AUC-PR"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="作者权重推理复现")
    parser.add_argument("--datasets", nargs="+", default=["MSL", "PSM", "SMD"],
                        choices=sorted(RBF_DIM))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else THESIS_DIR / "results" / f"{stamp}_reproduction"
    if not out_dir.is_absolute():
        out_dir = CALLER_CWD / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        overrides = [f"dataset={dataset}", f"model.rbf_dim={RBF_DIM[dataset]}", "seed=0"]
        with initialize_config_dir(version_base=None, config_dir=str(RESTAD_DIR / "configs")):
            cfg = compose(config_name="base_config", overrides=overrides)
        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'=' * 60}\n数据集：{dataset}   rbf_dim={RBF_DIM[dataset]}\n{'=' * 60}")
        set_seed(args.seed)

        (x_train, _), (x_test, y_test) = get_data(cfg)
        train_dl, test_dl = generate_loaders(x_train, x_test, y_test, cfg)

        # 关键：load_model=True -> 加载作者权重，不训练
        model = train_with_rbf(cfg, train_dl, test_dl, args.seed,
                               encoder_output=None, save_mode=False, load_model=True)

        rec_tr = calculate_reconstruction_errors(model, train_dl, cfg)
        rec_te, labels = calculate_reconstruction_errors(model, test_dl, cfg, test_mode=True)
        rbf_tr = calculate_rbf_scores(model, train_dl, cfg)
        rbf_te = calculate_rbf_scores(model, test_dl, cfg)

        res_rec = evaluate_rec(rec_tr, rec_te, labels, cfg,
                               thresh_type_list=["ratio"], adjustment_mode_list=[False])
        res_mul = evaluate_RBFrec(rec_tr, rec_te, rbf_tr, rbf_te, labels, cfg,
                                  thresh_type_list=["ratio"], adjustment_mode_list=[False])
        res_add = evaluate_RBFrec_Addition(rec_tr, rec_te, rbf_tr, rbf_te, labels, cfg,
                                           thresh_type_list=["ratio"], adjustment_mode_list=[False])

        paper = PAPER_TABLE1[dataset]
        mul = extract(res_mul)
        row = {
            "dataset": dataset,
            "rbf_dim": RBF_DIM[dataset],
            "rec_only": extract(res_rec),
            "reproduction_eps_r_times_eps_s": mul,
            "addition_eps_r_plus_eps_s": extract(res_add),
            "paper_table1_RESTAD_R": paper,
            "delta_vs_paper": {
                "f1": mul["f1"] - paper["f1"],
                "auc": mul["auc"] - paper["auc"],
                "auc_pr": mul["auc_pr"] - paper["auc_pr"],
            },
        }
        rows.append(row)

        np.savez_compressed(out_dir / f"scores_{dataset}.npz",
                            rec_train=rec_tr, rec_test=rec_te,
                            rbf_train=rbf_tr, rbf_test=rbf_te, labels=labels)

        print(f"\n  论文 Table 1 (RESTAD (R)) : F1 {paper['f1']:.2f}  "
              f"AUC-ROC {paper['auc']:.2f}  AUC-PR {paper['auc_pr']:.2f}")
        print(f"  本地复现 (εr×εs)         : F1 {mul['f1']:.4f}  "
              f"AUC-ROC {mul['auc']:.4f}  AUC-PR {mul['auc_pr']:.4f}")
        print(f"  差值                      : ΔF1 {row['delta_vs_paper']['f1']:+.4f}  "
              f"ΔAUC {row['delta_vs_paper']['auc']:+.4f}  "
              f"ΔAUC-PR {row['delta_vs_paper']['auc_pr']:+.4f}")

    (out_dir / "reproduction.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}\n汇总（作者权重推理 vs 论文 Table 1）\n{'=' * 60}")
    print(f"{'数据集':<8}{'指标':<10}{'论文':>10}{'复现':>10}{'差值':>10}")
    for row in rows:
        m, p = row["reproduction_eps_r_times_eps_s"], row["paper_table1_RESTAD_R"]
        for key, label in (("f1", "F1-Score"), ("auc", "AUC-ROC"), ("auc_pr", "AUC-PR")):
            print(f"{row['dataset']:<8}{label:<10}{p[key]:>10.2f}{m[key]:>10.4f}"
                  f"{m[key] - p[key]:>+10.4f}")
    print(f"\n已写出：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
