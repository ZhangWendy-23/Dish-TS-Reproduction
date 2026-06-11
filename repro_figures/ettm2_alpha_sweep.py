"""ETTm2 alpha sweep.

Runs ``train.py`` for a grid of (alpha, pred_len) with Autoformer + Dish-TS,
using the SAME seed every run so differences are only due to alpha.

Outputs
-------
* Appends one row per run to ``results/figure3_runs.csv`` (done by train.py).
* Writes console log per run to ``logs/ettm2_alpha_{alpha}_pd{pred_len}_seed{seed}.log``.

Typical usage
-------------
    # default grid: alpha in [0, 0.1, 0.25, 0.5, 0.75, 1.0] x pred_len in [24, 96, 168, 336]
    python3 repro_figures/ettm2_alpha_sweep.py --gpu 0

    # custom grid:
    python3 repro_figures/ettm2_alpha_sweep.py \
        --alphas 0 0.25 0.5 1.0 --pred_lens 24 96 168 336 \
        --seeds 2023 2024 2025 --gpu 0

Once the sweep finishes, plot Figure-3 style curves with:
    python3 repro_figures/plot_figure3.py --input results/figure3_runs.csv \
        --dataset ETTm2 --model Autoformer --norm dishts --zscore \
        --output results/figures/figure3_ETTm2.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.py")
LOG_DIR = os.path.join(ROOT, "logs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--pred_lens", type=int, nargs="+",
                    default=[24, 96, 168, 336])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2023])
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (Dish-TS often needs > paper's 7).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only print commands, do not launch training.")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    t0 = datetime.now()
    total = len(args.alphas) * len(args.pred_lens) * len(args.seeds)
    print(f"[{t0:%Y-%m-%d %H:%M}] ETTm2 alpha sweep: {total} runs "
          f"(alphas={args.alphas}, pred_lens={args.pred_lens}, seeds={args.seeds})")

    n_failed = 0
    done = 0
    for seed in args.seeds:
        for alpha in args.alphas:
            for pl in args.pred_lens:
                log_file = os.path.join(
                    LOG_DIR,
                    f"ETTm2_Autoformer_dishts_alpha{alpha}_pd{pl}_seed{seed}.log",
                )
                cmd = [
                    sys.executable, TRAIN,
                    "--data", "ETTm2",
                    "--model", "Autoformer",
                    "--norm", "dishts",
                    "--seq_len", str(args.seq_len),
                    "--pred_len", str(pl),
                    "--alpha", str(alpha),
                    "--batch_size", str(args.batch_size),
                    "--patience", str(args.patience),
                    "--seed", str(seed),
                    "--gpu", str(args.gpu),
                ]
                done += 1
                print(f"\n[{done}/{total}] seed={seed} alpha={alpha} pred_len={pl}\n"
                      f"  -> {' '.join(cmd)}\n"
                      f"  -> log: {log_file}")
                if args.dry_run:
                    continue
                with open(log_file, "w") as fh:
                    rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
                if rc.returncode != 0:
                    n_failed += 1
                    print(f"  !!! FAILED (rc={rc.returncode}); see {log_file}")

    t1 = datetime.now()
    print(f"\n[{t1:%Y-%m-%d %H:%M}] sweep done.  failures: {n_failed}/{total}")


if __name__ == "__main__":
    main()
