"""Reproduce Figure 3 from the Dish-TS paper.

Figure 3 shows MSE z-score curves as a function of lookback/horizon window
length (24, 48, 96, 168, 336) for four values of the prior-guidance weight
alpha (0, 0.25, 0.5, 1.0) on four datasets (Electricity, ETTh1, ETTm2, Weather).

For each (dataset, alpha, window_length) the script calls `train.py` via
subprocess, collects the resulting MSE, and saves a wide CSV table. Then it
normalizes MSE to z-score *per dataset* and plots four sub-figures (one per
dataset), just like Figure 3.

Typical usage
-------------
    # Step 1 - run experiments (can take hours on a single GPU)
    python3 repro_figures/figure3_alpha_sensitivity.py --run

    # Step 2 - aggregate results from logs/ into a CSV and plot
    python3 repro_figures/figure3_alpha_sensitivity.py --plot

Both steps can be combined (default behaviour):
    python3 repro_figures/figure3_alpha_sensitivity.py
"""

import argparse
import os
import subprocess
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ----- project layout -----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "train.py")
LOGS = os.path.join(ROOT, "logs")
OUT_CSV = os.path.join(ROOT, "paper_results", "figure3_alpha_sensitivity.csv")
OUT_PNG = os.path.join(ROOT, "paper_results", "figure3_alpha_sensitivity.png")

# ----- Figure 3 grid -----
DATASETS = ["Electricity", "ETTh1", "ETTm2", "Weather"]
# table3-style settings: multivariate, Autoformer backbone, lookback = horizon
BACKBONE = "Autoformer"
WINDOWS = [24, 48, 96, 168, 336]
ALPHAS = [0.0, 0.25, 0.5, 1.0]


# ------------------------------------------------------------------ helpers
def log_file(dataset, alpha, window):
    """Deterministic log file name for a single experiment."""
    fname = (f"fig3_{dataset}_autoformer_dishts_w{window}_alpha{alpha:g}.txt")
    return os.path.join(LOGS, fname)


def mse_from_log(path):
    """Read MSE from a log produced by train.py (last block prints MSE=...)."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    # The last "MSE=XXXX" line wins (aggregated over the whole test set).
    for line in reversed(text.splitlines()):
        if "MSE=" in line and "=" in line:
            # e.g. "  MSE=0.123456  MAE=..."
            for token in line.split():
                if token.startswith("MSE="):
                    try:
                        return float(token.split("=", 1)[1])
                    except ValueError:
                        continue
    return None


def run_one(dataset, alpha, window, gpu=0):
    """Run train.py for one (dataset, alpha, window) cell; log to LOGS/."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = log_file(dataset, alpha, window)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        sys.executable, TRAIN,
        "--data", dataset,
        "--model", BACKBONE,
        "--norm", "dishts",
        "--seq_len", str(window),
        "--pred_len", str(window),
        # label_len clamped in train.py; pass a sensible default:
        "--label_len", str(max(24, window // 2)),
        "--features", "M",
        "--batch_size", "0",
        "--alpha", f"{alpha:g}",
        "--seed", "2021",
        "--gpu", "0",
        "--train_epochs", "10",
        "--patience", "3",
    ]
    print(f"[fig3] RUN {' '.join(cmd)}")
    with open(out, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                           env=env)
    print(f"[fig3]   -> exit={p.returncode}  log={out}")


def gather_results():
    """Build a DataFrame by scanning LOGS/ for figure3 outputs."""
    rows = []
    for dataset in DATASETS:
        for alpha in ALPHAS:
            for window in WINDOWS:
                mse = mse_from_log(log_file(dataset, alpha, window))
                rows.append({"dataset": dataset, "alpha": alpha,
                             "window": window, "mse": mse})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ plotting
def zscore_by_group(df, group_col="dataset", value_col="mse"):
    out = df.copy()
    out["zscore"] = np.nan
    for key, g in out.groupby(group_col):
        mu, sd = g[value_col].mean(), g[value_col].std()
        if sd is not None and sd > 0:
            out.loc[g.index, "zscore"] = (g[value_col] - mu) / sd
    return out


def plot_figure3(df):
    """Re-produce the style of Figure 3: 1x4 subplots, lineplot per alpha."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)
    palette = {0.0: "#1f77b4", 0.25: "#ff2d2d", 0.5: "#2ca02c", 1.0: "#d4a017"}
    markers = {0.0: "o", 0.25: "o", 0.5: "s", 1.0: "^"}

    for ax, dataset in zip(axes, DATASETS):
        sub = df[(df.dataset == dataset) & (df.mse.notna())]
        if sub.empty:
            ax.set_title(dataset + " (no data)")
            continue
        sub_z = zscore_by_group(sub, group_col="dataset", value_col="mse")
        for alpha in ALPHAS:
            seg = sub_z[sub_z.alpha == alpha].sort_values("window")
            if seg.empty:
                continue
            ax.plot(seg["window"], seg["zscore"],
                    marker=markers[alpha], linewidth=2.2,
                    label=rf"$\alpha={alpha:g}$",
                    color=palette[alpha])
        ax.set_title(dataset, fontsize=12)
        ax.set_xlabel("lookback/horizon window length", fontsize=10)
        ax.set_xticks(WINDOWS)
        ax.set_ylabel("MSE z-score", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Figure 3 reproduction: impact of prior-guidance weight alpha",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="Run train.py for every (dataset, alpha, window) cell")
    ap.add_argument("--plot", action="store_true",
                    help="Aggregate existing logs into CSV + plot")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # default: run then plot
    if not args.run and not args.plot:
        args.run = args.plot = True

    if args.run:
        print(f"[fig3] Running experiments for "
              f"{len(DATASETS)} datasets x {len(ALPHAS)} alphas x "
              f"{len(WINDOWS)} windows = "
              f"{len(DATASETS) * len(ALPHAS) * len(WINDOWS)} runs. "
              f"This will take several hours.")
        for dataset in DATASETS:
            for alpha in ALPHAS:
                for window in WINDOWS:
                    if os.path.exists(log_file(dataset, alpha, window)) \
                            and mse_from_log(log_file(dataset, alpha, window)) is not None:
                        print(f"[fig3] SKIP {dataset} a={alpha} w={window} (log exists)")
                        continue
                    run_one(dataset, alpha, window, gpu=args.gpu)

    df = gather_results()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[fig3] Wrote {OUT_CSV}  ({df.mse.notna().sum()} cells with MSE)")

    if args.plot and not df.empty and df.mse.notna().any():
        fig = plot_figure3(df)
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        print(f"[fig3] Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
