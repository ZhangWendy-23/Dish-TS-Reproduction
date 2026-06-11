"""Re-run the full experimental suite (Tables 2 & 3) with StandardScaler.

This script re-runs every (dataset, model, norm, pred_len, seed) combination
that was executed previously, but with the StandardScaler data preprocessing
enabled so that MSE values match the scale reported in the Dish-TS paper.

Before running, it backs up the old ``results/paper_summary.csv`` (raw-scale
MSE) and clears the current results so that new scaled values are not mixed
with old raw-scale values.

Usage
-----
    python3 repro_figures/run_all_exps.py --gpu 0
    python3 repro_figures/run_all_exps.py --gpu 0 --seeds 2023 2024 2025  # default
    python3 repro_figures/run_all_exps.py --gpu 0 --dry-run               # print only
    python3 repro_figures/run_all_exps.py --gpu 0 --no-auto-backup        # skip backup
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.py")
LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")


# ---------- experiment matrix (paper Tables 2 & 3) ----------

# Each entry: (dataset, model, pred_len, norms_to_test)
STAGE1 = [
    # --- Table 2 multivariate & Table 3 RevIN comparison ---
    ("ECL",   "Autoformer", [24, 168],           ["dishts", "revin"]),
    ("ETTh1", "Autoformer", [24, 168],           ["dishts", "revin"]),
    ("ETTm2", "Autoformer", [24, 96, 168],       ["dishts", "revin", "none"]),
    ("WTH",   "Autoformer", [24, 168],           ["dishts", "revin"]),
]

# Long-horizon (Table 4) — only if explicitly requested via --long-horizon
STAGE2_LONG = [
    ("ETTh1", "Autoformer", [336], ["dishts", "revin"]),
    ("ETTm2", "Autoformer", [336], ["dishts", "revin"]),
    ("WTH",   "Autoformer", [336], ["dishts", "revin"]),
    ("ECL",   "Autoformer", [336], ["dishts", "revin"]),
]

DEFAULT_ALPHA = 0.5
BATCH_SIZE = 128
PATIENCE = 15
SEQ_LEN = 96


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[2023, 2024, 2025])
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--long-horizon", action="store_true",
                    help="Also run pred_len=336 experiments (Table 4).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-auto-backup", action="store_true",
                    help="Skip backing up old paper_summary.csv.")
    ap.add_argument("--no-scale", action="store_true",
                    help="Disable StandardScaler (use raw data scale).")
    args = ap.parse_args()

    # -------- backup old results --------
    summary_path = os.path.join(RESULTS_DIR, "paper_summary.csv")
    if not args.dry_run and not args.no_auto_backup:
        _backup_old_results(summary_path)

    # -------- build job list --------
    matrix = list(STAGE1)
    if args.long_horizon:
        matrix.extend(STAGE2_LONG)

    jobs: list[list[str]] = []
    for data, model, pred_lens, norms in matrix:
        for pl in pred_lens:
            for norm in norms:
                for seed in args.seeds:
                    log_name = _log_filename(data, model, norm, seed,
                                             args.seq_len, pl, args.alpha)
                    jobs.append([
                        sys.executable, TRAIN,
                        "--data", data,
                        "--model", model,
                        "--norm", norm,
                        "--seq_len", str(args.seq_len),
                        "--pred_len", str(pl),
                        "--alpha", str(args.alpha),
                        "--batch_size", str(args.batch_size),
                        "--patience", str(args.patience),
                        "--seed", str(seed),
                        "--gpu", str(args.gpu),
                    ] + (["--no-scale"] if args.no_scale else []))

    # -------- print summary --------
    t0 = datetime.now()
    total = len(jobs)
    print(f"[{t0:%Y-%m-%d %H:%M}] Full repro experimental suite: {total} runs")
    print(f"  datasets:  {sorted(set(j[3] for j in jobs))}")
    print(f"  norms:     {sorted(set(j[7] for j in jobs))}")
    print(f"  pred_lens: {sorted(set(int(j[11]) for j in jobs))}")
    print(f"  seeds:     {args.seeds}")
    print(f"  alpha:     {args.alpha}")
    print(f"  StandardScaler: {'OFF' if args.no_scale else 'ON'}")
    print()
    if args.dry_run:
        print("DRY RUN — commands would be:")
        for cmd in jobs:
            print("  " + " ".join(cmd))
        return

    # -------- execute --------
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_failed = 0
    for i, cmd in enumerate(jobs):
        # Extract values from the command list (fixed-index format)
        data  = cmd[3]
        model = cmd[5]
        norm  = cmd[7]
        seq   = int(cmd[9])
        pl    = int(cmd[11])
        alpha = float(cmd[13])
        seed  = int(cmd[19])
        log_file = os.path.join(LOG_DIR,
                                _log_filename(data, model, norm, seed,
                                              seq, pl, alpha))
        meta = f"data={data} norm={norm} pred_len={pl} seed={seed}"
        print(f"\n[{i+1}/{total}] {meta}")
        print(f"  log: {log_file}")
        try:
            with open(log_file, "w") as f:
                f.write(f"# {meta}\n# {' '.join(cmd)}\n\n")
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      text=True)
            if proc.returncode != 0:
                print(f"  FAILED (return code {proc.returncode})")
                n_failed += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as exc:
            print(f"  ERROR: {exc}")
            n_failed += 1

    elapsed = datetime.now() - t0
    print(f"\n{'='*60}")
    print(f"Done. {total - n_failed}/{total} succeeded "
          f"({n_failed} failed) in {elapsed}")
    print(f"Results: {summary_path}")
    print(f"Logs:    {LOG_DIR}")


def _log_filename(data: str, model: str, norm: str, seed: int,
                  seq_len: int, pred_len: int, alpha: float) -> str:
    return f"{data}_{model}_{norm}_s{seq_len}_p{pred_len}_seed{seed}.log"


def _backup_old_results(summary_path: str) -> None:
    """Move old paper_summary.csv to a timestamped backup."""
    if not os.path.exists(summary_path):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(RESULTS_DIR, f"paper_summary_rawScale_backup_{ts}.csv")
    os.rename(summary_path, backup)
    print(f"[backup] old raw-scale results → {backup}")


if __name__ == "__main__":
    main()
