# -*- coding: utf-8 -*-
"""把消融矩阵拆成多份并发跑，用满一张卡。

为什么要这个
------------
RESTAD 模型很小：单次训练峰值显存约 1.3 GB，而消费级显卡动辄 6~24 GB，
串行跑等于让显卡大部分时间闲着。实测瓶颈是"一次只能跑一个实验"，
而不是单次实验有多慢。本脚本把任务按轮转方式分给 N 个进程并发执行。

一个进程一份任务，进程之间写的是各自独立的 run 目录，互不冲突；
唯一的共享文件 `metrics_all.json` 由各分片在结束时**追加**写入，
不会被覆盖（格式是行式 JSON，每行一条记录）。

用法（工作区根目录）
--------------------
本地开 3 个并发（6 GB 显存比较稳妥的上限）：

    .\\.venv-restad\\Scripts\\python.exe RESTAD\\thesis\\run_matrix.py ^
        --datasets MSL PSM SMD ^
        --variants baseline M1 M2 M3 M5 M1+M2+M5 ^
        --seeds 0 1 2 3 4 5 6 7 8 9 ^
        --protocol val --workers 3

服务器上显存宽裕时可以开 8 个：

    python RESTAD/thesis/run_matrix.py --workers 8 --datasets MSL PSM SMD ^
        --variants baseline M1 M2 M3 M5 M1+M2+M5 --seeds 0 1 2 3 4 --protocol val
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent
CALLER_CWD = Path.cwd()


def main() -> int:
    parser = argparse.ArgumentParser(description="并发跑消融矩阵")
    parser.add_argument("--datasets", nargs="+", default=["MSL", "PSM"])
    parser.add_argument("--variants", nargs="+", default=["baseline", "M1", "M3", "M1+M2+M5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--protocol", choices=["author", "val"], default="val")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tag", type=str, default="matrix")
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--resume-out-root", type=str, default="",
                        help="断点续跑：复用已有的结果目录，配合 --skip-existing 跳过已完成组合")
    parser.add_argument("--skip-existing", action="store_true",
                        help="已存在完整 metrics.json 的组合直接跳过")
    args = parser.parse_args()

    if args.resume_out_root:
        out_root = Path(args.resume_out_root)
        if not out_root.is_absolute():
            out_root = CALLER_CWD / out_root
        print(f"[run_matrix] 断点续跑模式，复用结果目录：{out_root}")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_root = THESIS_DIR / "results" / f"{stamp}_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.variants) * len(args.seeds)
    print(f"[run_matrix] 共 {total} 组任务，{args.workers} 个并发进程 -> {out_root}")

    common = [
        sys.executable, str(THESIS_DIR / "run_variants.py"),
        "--datasets", *args.datasets,
        "--variants", *args.variants,
        "--seeds", *[str(s) for s in args.seeds],
        "--epochs", str(args.epochs),
        "--protocol", args.protocol,
        "--val-ratio", str(args.val_ratio),
        "--out-root", str(out_root),
        "--extra", *args.extra,
    ]
    if args.skip_existing:
        common.append("--skip-existing")

    procs, logs = [], []
    t0 = time.time()

    # 子进程环境：把 CPU 线程数压到 1。
    # Torch / OpenMP 默认按核数开线程，云服务器 100+ 核时，N 个并发进程
    # 会各自开满线程互相抢占，实测能把单组实验从十几秒拖到几分钟。
    child_env = dict(os.environ)
    child_env.setdefault("OMP_NUM_THREADS", "1")
    child_env.setdefault("MKL_NUM_THREADS", "1")
    child_env.setdefault("OPENBLAS_NUM_THREADS", "1")
    child_env.setdefault("NUMEXPR_NUM_THREADS", "1")
    child_env.setdefault("RESTAD_TORCH_THREADS", "1")
    print(f"[run_matrix] 每个子进程限制 CPU 线程数为 {child_env['RESTAD_TORCH_THREADS']}"
          f"（OMP_NUM_THREADS={child_env['OMP_NUM_THREADS']}）")

    for shard in range(args.workers):
        log_path = out_root / f"shard{shard}.log"
        log = open(log_path, "w", encoding="utf-8")
        logs.append(log)
        procs.append(subprocess.Popen(
            common + ["--shard", str(shard), "--shards", str(args.workers)],
            stdout=log, stderr=subprocess.STDOUT, env=child_env,
        ))
        print(f"  启动分片 {shard} -> {log_path.name}")

    failed = 0
    for shard, proc in enumerate(procs):
        rc = proc.wait()
        logs[shard].close()
        if rc != 0:
            failed += 1
            print(f"  分片 {shard} 退出码 {rc}，见 {out_root / f'shard{shard}.log'}")

    # 汇总：把每个 run 目录里的 metrics.json 收集成一份总表
    rows = []
    for run_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        meta_path = run_dir / "metrics.json"
        if meta_path.exists():
            rows.append(json.loads(meta_path.read_text(encoding="utf-8")))
    (out_root / "metrics_all.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in rows if "error" not in r]
    elapsed = time.time() - t0
    print(f"\n[run_matrix] 完成 {len(ok)}/{total}，用时 {elapsed/60:.1f} 分钟"
          f"（失败分片 {failed} 个）")
    print(f"  结果目录：{out_root}")
    print(f"  下一步：python RESTAD/thesis/compare_variants.py "
          f"--results {out_root} --scoring mul_ema")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
