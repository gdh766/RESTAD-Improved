# -*- coding: utf-8 -*-
"""RESTAD 毕设消融实验运行器。

设计原则
--------
1. 不修改作者的任何原始脚本入口（main.py / solver.py / stages_training.py），
   而是直接调用它们暴露出来的函数，自己控制实验矩阵。
2. 每个变体都通过 hydra override 打开，默认值 = 原论文行为，
   因此 baseline 行就是"原作者代码 + 原论文协议"的结果。
3. 每次运行单独落盘：配置、指标、逐点分数、模型权重，便于复核与画图。

用法（工作区根目录下运行）
--------------------------
    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\run_variants.py ^
        --datasets PSM MSL --variants baseline M1 M2 M3 M1+M2+M3 --seeds 0 1 2

常用参数
--------
--datasets   数据集子集，可选 PSM / MSL / SMD
--variants   变体名，见下方 VARIANTS
--seeds      随机种子
--epochs     覆盖训练轮数（默认沿用各数据集的标准设置，MSL/PSM/SMD 均为 100）
--tag        结果目录后缀，便于区分不同批次
--extra      额外 hydra override，可重复传入，例如 --extra model.center_div_lambda=0.05
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# 线程数：默认让 Torch 按 CPU 核数开 intra-op 线程（云服务器上可能是 100+ 核）。
# 但我们是**多进程并发**跑小模型，如果每个进程都开满线程，N 个进程 × 100+ 线程
# 会互相抢占，实测能把单组实验从十几秒拖到几分钟。
# run_matrix.py 会设 RESTAD_TORCH_THREADS=1 并在子进程环境里禁掉 OMP 线程；
# 这里再兜一层，保证单独直接运行 run_variants.py 时也受控。
# ---------------------------------------------------------------------------
_THREADS = int(os.environ.get("RESTAD_TORCH_THREADS", "0"))
if _THREADS > 0:
    torch.set_num_threads(_THREADS)

# ---------------------------------------------------------------------------
# 路径与导入：作者的 restad/ 目录用的是扁平 import（from Utils import ...），
# 所以必须把该目录加进 sys.path，并把工作目录切过去（配置里的
# data_prefix="datasets/MSL" 是相对 restad/ 的）。
# ---------------------------------------------------------------------------
THESIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THESIS_DIR.parent
RESTAD_DIR = REPO_DIR / "restad"
# 记下调用时的目录：下面要把工作目录切到 restad/，命令行里的相对路径
# 必须基于调用者的目录解析，不能用切换后的目录。
CALLER_CWD = Path.cwd()

sys.path.insert(0, str(RESTAD_DIR))
os.chdir(RESTAD_DIR)

from hydra import compose, initialize_config_dir  # noqa: E402

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


# ---------------------------------------------------------------------------
# 各数据集的标准超参数（取自 Hyperparameters_info.md 与模型配置里的注释）
# ---------------------------------------------------------------------------
DATASET_HP = {
    "MSL": dict(rbf_dim=128, lr=1e-2, weight_decay=1e-5, dropout=0.0,
                clip_grad=3.0, batch_size=64, num_epochs=100),
    "PSM": dict(rbf_dim=32, lr=1e-2, weight_decay=1e-3, dropout=0.5,
                clip_grad=3.0, batch_size=64, num_epochs=100),
    "SMD": dict(rbf_dim=256, lr=1e-2, weight_decay=1e-3, dropout=0.0,
                clip_grad=1.5, batch_size=64, num_epochs=100),
}

# ---------------------------------------------------------------------------
# 变体定义。空 dict = 不加任何 override = 原论文行为。
# ---------------------------------------------------------------------------
VARIANTS = {
    "baseline": {},
    # M1：RBF 门控残差注入，替代原作者的"整段替换"
    "M1": {
        "model.use_residual_rbf": "true",
    },
    # M2：RBF 中心多样性 + 尺度正则
    "M2": {
        "model.center_div_lambda": "0.1",
        "model.gamma_reg_lambda": "0.01",
        "model.gamma_reg_target": "-3.0",
    },
    # M3：稳健归一化 + 重建误差改用 L2（对齐论文式(2)）
    "M3": {
        "model.robust_norm": "true",
        "model.rec_error_type": "l2",
    },
    # M2 的强正则档：把中心推得更开、把核宽拉得更紧
    "M2s": {
        "model.center_div_lambda": "0.5",
        "model.gamma_reg_lambda": "0.1",
        "model.gamma_reg_target": "-3.0",
    },
    # 三者叠加
    "M1+M2+M3": {
        "model.use_residual_rbf": "true",
        "model.center_div_lambda": "0.1",
        "model.gamma_reg_lambda": "0.01",
        "model.gamma_reg_target": "-3.0",
        "model.robust_norm": "true",
        "model.rec_error_type": "l2",
    },
    # M1 + 强正则 M2s
    "M1+M2s": {
        "model.use_residual_rbf": "true",
        "model.center_div_lambda": "0.5",
        "model.gamma_reg_lambda": "0.1",
        "model.gamma_reg_target": "-3.0",
    },
    # M5：密度对齐损失（单独使用，用于隔离它的贡献）
    "M5": {
        "model.density_align_lambda": "1.0",
        "model.density_align_target": "0.9",
    },
    # M1 + M2 + M5：配套的完整方案
    #   M1 把相似度分支与主干特征解耦，M2 防止中心塌缩，M5 给相似度分支明确的优化目标。
    "M1+M2+M5": {
        "model.use_residual_rbf": "true",
        "model.center_div_lambda": "0.1",
        "model.gamma_reg_lambda": "0.01",
        "model.gamma_reg_target": "-3.0",
        "model.density_align_lambda": "1.0",
        "model.density_align_target": "0.9",
    },
    # Full：四处修改全开（M1 架构 + M2 正则 + M3 评分口径 + M5 密度对齐）
    "Full": {
        "model.use_residual_rbf": "true",
        "model.center_div_lambda": "0.1",
        "model.gamma_reg_lambda": "0.01",
        "model.gamma_reg_target": "-3.0",
        "model.density_align_lambda": "1.0",
        "model.density_align_target": "0.9",
        "model.robust_norm": "true",
        "model.rec_error_type": "l2",
    },
}


def build_cfg(dataset: str, overrides: list) -> OmegaConf:
    """用 hydra 组合出一次实验的配置（等价于 Utils.load_config，但支持 override）。"""
    with initialize_config_dir(version_base=None, config_dir=str(RESTAD_DIR / "configs")):
        cfg = compose(config_name="base_config", overrides=overrides)
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def variant_overrides(dataset: str, variant: str, seed: int, epochs: int, extra: list) -> list:
    hp = DATASET_HP[dataset]
    ov = [f"dataset={dataset}", f"seed={seed}"]
    for key, value in hp.items():
        if key == "num_epochs":
            continue
        ov.append(f"model.{key}={value}")
    ov.append(f"model.num_epochs={epochs}")
    ov.extend(f"{k}={v}" for k, v in VARIANTS[variant].items())
    ov.extend(extra)
    return ov


def extract_ratio_metrics(result_dict: dict) -> dict:
    """从评价函数返回的嵌套字典里取出 (ratio, False) 这一档的指标。"""
    entry = result_dict[("ratio", False)]
    return {
        "f1": float(entry.get("RBFrec f1-score", entry.get("f1-score"))),
        "precision": float(entry.get("RBFrec precision", entry.get("precision"))),
        "recall": float(entry.get("RBFrec recall", entry.get("recall"))),
        "auc": float(entry.get("RBFrec AUC", entry.get("AUC"))),
        "auc_pr": float(entry.get("RBFrec AUC-PR", entry.get("AUC-PR"))),
    }


def build_loaders(cfg, protocol: str, val_ratio: float):
    """
    构造数据加载器。

    protocol="author"
        完全沿用原作者流程：用完整训练序列训练，并且——这是原代码的既有行为——
        模型选择 / 早停 / 学习率调度全都盯着**测试集**损失。
        保留它是为了能与论文表格对齐，但它不是严格的独立测试。

    protocol="val"
        从训练序列**尾部按时间顺序**切出 val_ratio 比例作验证集，
        模型选择、早停与 ReduceLROnPlateau 只看验证集，
        测试集只在最后算分数时出现一次。
        按时间顺序切（而不是随机切）是为了不把未来信息漏进验证集；
        代价是验证集只覆盖训练序列末段的数据分布，这一点在论文里要写明。

    返回 (train_loader, eval_loader, test_loader)。
    """
    (x_train, _), (x_test, y_test) = get_data(cfg)

    if protocol == "author":
        train_dl, test_dl = generate_loaders(x_train, x_test, y_test, cfg)
        return train_dl, test_dl, test_dl

    cut = int(len(x_train) * (1.0 - val_ratio))
    train_part, val_part = x_train[:cut], x_train[cut:]
    train_dl, val_dl = generate_loaders(
        train_part, val_part, np.zeros(len(val_part), dtype=int), cfg)

    # 单独构造测试集 loader：第一个参数只是占位，给一个刚好一个窗口的空序列，
    # 避免把整条测试序列重复切两遍（SMD 序列很长，会白吃内存和时间）。
    placeholder = np.zeros((cfg.dataset.window_size, cfg.dataset.x_dim), dtype=np.float32)
    _, test_dl = generate_loaders(placeholder, x_test, y_test, cfg)
    return train_dl, val_dl, test_dl


def run_one(dataset: str, variant: str, seed: int, epochs: int, extra: list,
            out_root: Path, protocol: str = "author", val_ratio: float = 0.2,
            quiet: bool = False, skip_existing: bool = False) -> dict:
    """训练 + 评分单个 (dataset, variant, seed) 组合，返回指标与落盘路径。"""
    run_name = f"{dataset}_{variant}_s{seed}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 【断点续跑】已经成功跑完的组合直接复用磁盘上的 metrics.json。
    # 判断标准是"有 metrics.json 且里面没有 error 字段"——metrics.json 是在
    # 训练和评分全部结束后才写的，所以它存在就等于这一组是完整结论。
    meta_path = run_dir / "metrics.json"
    if skip_existing and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if "error" not in cached:
                return {**cached, "skipped": True}
        except (json.JSONDecodeError, OSError):
            pass                      # 文件坏了就重跑，不要因为一条脏数据卡住整批

    ov = variant_overrides(dataset, variant, seed, epochs, extra)
    cfg = build_cfg(dataset, ov)

    devnull = open(os.devnull, "w", encoding="utf-8") if quiet else None
    old_stdout = sys.stdout
    if quiet:
        sys.stdout = devnull
    try:
        t0 = time.time()
        set_seed(42)                      # 与 main.py 保持一致
        cfg.seed = seed

        train_dl, eval_dl, test_dl = build_loaders(cfg, protocol, val_ratio)

        # ---- 训练 ----
        # 关键：第二个 dataloader 就是 Trainer 内部用来做早停/选最优的那个。
        # protocol="author" 时它就是测试集（原作者行为）；
        # protocol="val" 时它是从训练序列切出来的验证集，测试集不再参与训练决策。
        model = train_with_rbf(cfg, train_dl, eval_dl, seed,
                               encoder_output=None, save_mode=False, load_model=False)

        # ---- 逐点分数 ----
        rec_tr = calculate_reconstruction_errors(model, train_dl, cfg)
        rec_te, labels = calculate_reconstruction_errors(model, test_dl, cfg, test_mode=True)
        rbf_tr = calculate_rbf_scores(model, train_dl, cfg)
        rbf_te = calculate_rbf_scores(model, test_dl, cfg)

        # ---- 三种评分方式 ----
        res_rec = evaluate_rec(rec_tr, rec_te, labels, cfg,
                               thresh_type_list=["ratio"], adjustment_mode_list=[False])
        res_simrec = evaluate_RBFrec(rec_tr, rec_te, rbf_tr, rbf_te, labels, cfg,
                                     thresh_type_list=["ratio"], adjustment_mode_list=[False])
        res_add = evaluate_RBFrec_Addition(rec_tr, rec_te, rbf_tr, rbf_te, labels, cfg,
                                           thresh_type_list=["ratio"], adjustment_mode_list=[False])
        elapsed = time.time() - t0
    finally:
        if quiet:
            sys.stdout = old_stdout
            devnull.close()

    metrics = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "protocol": protocol,
        "val_ratio": val_ratio if protocol == "val" else None,
        "rec_only": extract_ratio_metrics(res_rec),
        "simrec_mul": extract_ratio_metrics(res_simrec),   # εr × εs，论文主结果
        "simrec_add": extract_ratio_metrics(res_add),      # εr + εs，对照
        "elapsed_sec": round(elapsed, 2),
        "overrides": ov,
        "use_residual_rbf": bool(cfg.model.use_residual_rbf),
        "center_div_lambda": float(cfg.model.center_div_lambda),
        "gamma_reg_lambda": float(cfg.model.gamma_reg_lambda),
        "robust_norm": bool(cfg.model.robust_norm),
        "rec_error_type": str(cfg.model.rec_error_type),
    }

    # ---- 落盘：配置 / 指标 / 逐点分数 / 权重 ----
    OmegaConf.save(OmegaConf.create(dict(OmegaConf.to_container(cfg))), run_dir / "config.yaml")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "scores.npz",
        rec_train=rec_tr, rec_test=rec_te, rbf_train=rbf_tr, rbf_test=rbf_te,
        labels=labels,
    )
    torch.save(model.state_dict(), run_dir / "model.pth")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="RESTAD 消融实验运行器")
    parser.add_argument("--datasets", nargs="+", default=["PSM", "MSL"],
                        choices=sorted(DATASET_HP))
    parser.add_argument("--variants", nargs="+", default=["baseline", "M1", "M2", "M3", "M1+M2+M3"],
                        choices=sorted(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=100,
                        help="覆盖训练轮数；MSL/PSM/SMD 的标准设置均为 100")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out-root", type=str, default="",
                        help="直接指定结果目录；多个分片并行时必须用同一个目录")
    parser.add_argument("--skip-existing", action="store_true",
                        help="断点续跑：已存在完整 metrics.json 的组合直接跳过")
    parser.add_argument("--protocol", choices=["author", "val"], default="author",
                        help="author = 原作者流程（用测试集选模型）；"
                             "val = 从训练序列尾部切验证集，测试集不参与训练决策")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="protocol=val 时验证集占训练序列的比例（按时间顺序切尾段）")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="额外的 hydra override，例如 model.center_div_lambda=0.05")
    parser.add_argument("--shard", type=int, default=0,
                        help="只跑第 shard 份任务（配合 --shards 做多进程并发）")
    parser.add_argument("--shards", type=int, default=1,
                        help="把任务均分成几份，供多个进程并行消费")
    parser.add_argument("--verbose", action="store_true", help="打印训练过程中的全部输出")
    args = parser.parse_args()

    if args.out_root:
        out_root = Path(args.out_root)
        if not out_root.is_absolute():
            out_root = CALLER_CWD / out_root
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = f"_{args.tag}" if args.tag else ""
        out_root = THESIS_DIR / "results" / f"{stamp}{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.variants) * len(args.seeds)
    print(f"[run_variants] 共 {total} 组实验 -> {out_root}")
    print(f"[run_variants] 协议={args.protocol}"
          + (f"（验证集比例 {args.val_ratio}，按时间切训练序列尾段）" if args.protocol == "val" else "（原作者流程）"))

    rows = []
    idx = 0
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                idx += 1
                # 分片：把任务按轮转方式分给多个进程。用轮转而不是整段切分，
                # 是为了让每个进程拿到的数据集/变体组合尽量均匀（SMD 单次耗时长得多，
                # 整段切分会让某个进程全是 SMD 而拖后腿）。
                if args.shards > 1 and (idx - 1) % args.shards != args.shard:
                    continue
                print(f"[{idx}/{total}] dataset={dataset} variant={variant} seed={seed}", flush=True)
                try:
                    m = run_one(dataset, variant, seed, args.epochs, args.extra,
                                out_root, protocol=args.protocol,
                                val_ratio=args.val_ratio, quiet=not args.verbose,
                                skip_existing=args.skip_existing)
                    rows.append(m)
                    s = m["simrec_mul"]
                    if m.get("skipped"):
                        print("        跳过（磁盘上已有完整结果）", flush=True)
                    else:
                        print(f"        εr×εs  F1={s['f1']:.4f}  AUC={s['auc']:.4f}  "
                              f"AP={s['auc_pr']:.4f}  ({m['elapsed_sec']}s)", flush=True)
                except Exception as exc:                       # noqa: BLE001
                    print(f"        FAILED: {type(exc).__name__}: {exc}", flush=True)
                    rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                                 "error": f"{type(exc).__name__}: {exc}"})

    (out_root / "metrics_all.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in rows if "error" not in r]
    print(f"\n[run_variants] 成功 {len(ok)}/{total}，结果目录：{out_root}")
    if ok:
        print("\n数据集 变体           n   F1(εr×εs)        AUC(εr×εs)       AP(εr×εs)")
        for dataset in args.datasets:
            for variant in args.variants:
                sub = [r for r in ok if r["dataset"] == dataset and r["variant"] == variant]
                if not sub:
                    continue
                f1 = np.mean([r["simrec_mul"]["f1"] for r in sub])
                auc = np.mean([r["simrec_mul"]["auc"] for r in sub])
                ap = np.mean([r["simrec_mul"]["auc_pr"] for r in sub])
                print(f"{dataset:<6} {variant:<13} {len(sub):<3} {f1:.4f}           "
                      f"{auc:.4f}           {ap:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
