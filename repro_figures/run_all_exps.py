"""Reproduce the Dish-TS experiment suite on GPU.

There are two phases:
  A) ``--phase baseline`` — the paper-default configuration: raw
     data (no StandardScaler), dish_init='avg', train_epochs=50,
     patience=7, Autoformer backbone, alpha sweep for dishts
     under three prior-loss variants (none / paper / legacy).
     This is the default phase.
  B) ``--phase scaled``  — same jobs but with ``--scale``
     (z-score preprocessing).  Useful to check which paper tables
     were reported in z-score space; NOT the paper default.

Use ``--dry-run`` to preview the generated commands before running
the real jobs.  All runs log to ``logs/<name>.log`` and append one
row per run to ``results/figure3_runs.csv``.

Examples
--------
    # default: phase A (raw scale, 3 seeds × many configs)
    python3 repro_figures/run_all_exps.py --gpu 0

    # smaller probe first
    python3 repro_figures/run_all_exps.py --gpu 0 --seeds 2023 \
        --datasets ETTm2 --pred-lens 24 --train-epochs 5

    # phase B (z-scale reference)
    python3 repro_figures/run_all_exps.py --gpu 0 --phase scaled
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

# ---------- fixed defaults (match the paper / official repo as closely as possible) ------
DEFAULT_SEQ_LEN = 96
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRAIN_EPOCHS = 50
DEFAULT_PATIENCE = 7
DEFAULT_SEEDS = [2023, 2024, 2025]

# Datasets / horizons we care about (paper Tables 2/3-ish).
# ETTm2 is the most heavily tested dataset in the paper; the
# others are used for cross-dataset sanity.
DEFAULT_DATASETS = ["ETTm2", "ETTh1", "ECL", "WTH"]
DEFAULT_PRED_LENS = {
    "ETTm2": [24, 96, 168, 336],
    "ETTh1": [24, 168],
    "ECL":   [24, 168],
    "WTH":   [24, 168],
}

# Baseline norm methods.  For dishts we always include alpha=0
# ("official repo equivalent") as the baseline, and we sweep
# alpha > 0 only with the --prior variants configured below.
BASELINE_NORMS = ["none", "revin", "dishts"]

# Alpha values to sweep ONLY for the dishts + --prior variants.
# Chosen to be dense enough to see the shape of the loss vs. alpha
# curve while staying computationally reasonable.
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

# Three "prior loss" variants.  Their semantic is explained in
# train.py — briefly:
#   - none   : no prior loss (matches official Dish-TS repo)
#   - paper  : align horizon CONET to the TRUE horizon mean
#              (matches the reading of "prediction window mean"
#               that the paper likely intended)
#   - legacy : align the model forecast MEAN to the horizon CONET
#              (older reading; empirically makes alpha>0 worse)
PRIOR_VARIANTS = ["none", "paper", "legacy"]


# ---------------- command-line interface -------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU id (passed to train.py --gpu).")
    ap.add_argument("--phase", type=str, default="baseline",
                    choices=["baseline", "scaled"],
                    help="baseline = raw data (paper default); "
                         "scaled = StandardScaler pre-processing.")
    ap.add_argument("--datasets", type=str, nargs="+",
                    default=DEFAULT_DATASETS,
                    help="Subset of datasets to run. Default: all 4.")
    ap.add_argument("--pred-lens", type=int, nargs="+", default=None,
                    help="Override per-dataset prediction lengths "
                         "(default is paper default per dataset).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--dish-init", type=str, default="avg",
                    choices=["standard", "avg", "uniform"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print generated commands but do not run them.")
    args = ap.parse_args()

    scale_flag = args.phase == "scaled"

    # ---------- build job list ----------
    jobs: list[dict] = []
    for data in args.datasets:
        pred_lens = args.pred_lens or DEFAULT_PRED_LENS.get(data, [24, 168])
        for pl in pred_lens:
            # ---- baselines (no "prior loss": --prior none --alpha 0.0) ----
            for norm in BASELINE_NORMS:
                for seed in args.seeds:
                    kwargs = dict(norm=norm, alpha=0.0, prior="none")
                    jobs.append(dict(data=data, pl=pl, seed=seed, **kwargs))
            # ---- dishts alpha sweep for each prior variant ----
            for prior in PRIOR_VARIANTS:
                for alpha in ALPHA_GRID:
                    # (alpha=0 under prior=X is equivalent to prior=none
                    #  because prior loss is gated on alpha>0; but we keep
                    #  the explicit row for clarity, deduplication is done
                    #  at reporting time.)
                    for seed in args.seeds:
                        jobs.append(dict(
                            data=data, pl=pl, seed=seed,
                            norm="dishts", alpha=alpha, prior=prior,
                        ))

    total = len(jobs)
    t0 = datetime.now()
    print(f"[{t0:%Y-%m-%d %H:%M}] Phase: {args.phase}  "
          f"(StandardScaler = {scale_flag})")
    print(f"  datasets:    {args.datasets}")
    print(f"  pred_lens:   {args.pred_lens or 'per-dataset paper default'}")
    print(f"  seeds:       {args.seeds}")
    print(f"  dish_init:   {args.dish_init}")
    print(f"  train_epoch: {args.train_epochs}  patience: {args.patience}")
    print(f"  total jobs:  {total}")
    print()
    if args.dry_run:
        print("DRY RUN — commands would be:")
        for j in jobs:
            print("  " + " ".join(_job_to_cmd(args, j, scale_flag)))
        return

    # ---------- execute ----------
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_failed = 0
    for i, job in enumerate(jobs):
        cmd = _job_to_cmd(args, job, scale_flag)
        meta = (f"data={job['data']} norm={job['norm']} "
                f"pred_len={job['pl']} seed={job['seed']} "
                f"alpha={job['alpha']} prior={job['prior']}")
        log_file = _log_filename(phase_suffix="scaled" if scale_flag else "raw",
                                 **job)
        print(f"[{i+1}/{total}] {meta}")
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
    print(f"Done. {total - n_failed}/{total} succeeded ({n_failed} failed) in {elapsed}")
    print(f"Results CSV: {os.path.join(RESULTS_DIR, 'figure3_runs.csv')}")
    print(f"Logs:        {LOG_DIR}")


# ---------------- helpers -------------------
def _job_to_cmd(args_ns, job: dict, scale_flag: bool) -> list[str]:
    cmd = [
        sys.executable, TRAIN,
        "--data", job["data"],
        "--model", "Autoformer",
        "--norm", job["norm"],
        "--seq_len", str(args_ns.seq_len),
        "--pred_len", str(job["pl"]),
        "--label_len", str(max(48, job["pl"] // 2)),
        "--batch_size", str(args_ns.batch_size),
        "--train_epochs", str(args_ns.train_epochs),
        "--patience", str(args_ns.patience),
        "--seed", str(job["seed"]),
        "--gpu", str(args_ns.gpu),
        "--alpha", str(job["alpha"]),
        "--prior", job["prior"],
        "--dish_init", args_ns.dish_init,
        "--figure3",
    ]
    if scale_flag:
        cmd.append("--scale")
    return cmd


def _log_filename(*, phase_suffix: str, data: str, pl: int,
                  seed: int, norm: str, alpha: float,
                  prior: str, **_) -> str:
    return (f"{data}_Autoformer_{norm}_s96_p{pl}_seed{seed}_"
            f"alpha{alpha}_prior{prior}_{phase_suffix}.log")


if __name__ == "__main__":
    main()
