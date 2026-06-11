"""Reproduce Figure 1 from the Dish-TS paper.

Figure 1 is a *conceptual* plot showing what a time series, its lookback window
and its horizon window look like, along with the empirical distributions
(histograms / KDEs) of lookback vs. horizon.

It is *not* a plot that can be reproduced exactly from the paper numbers (the
paper uses a particular, unspecified Weather slice). We therefore draw the
same *type* of plot from any chosen slice of a dataset, and allow the user to
pick the sample.

Usage
-----
    python3 repro_figures/figure1_distribution_shift.py \
        --data Weather --sample_idx 0 --seq_len 96 --pred_len 96

Outputs
-------
    paper_results/figure1_distribution_shift.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "paper_results", "figure1_distribution_shift.png")


def _load_series(data):
    """Load a data CSV from ``data/{data}/{data}.csv`` and pick a numeric column."""
    path = os.path.join(ROOT, "data", data, f"{data}.csv")
    df = pd.read_csv(path)
    # pick a numeric column - either 'OT' (target), otherwise pick the 2nd column
    numeric_cols = [c for c in df.columns if c not in ("date", "Date", "time", "Time")]
    col = "OT" if "OT" in numeric_cols else numeric_cols[-1]
    return df[col].values.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Weather")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--sample_idx", type=int, default=0,
                    help="Which (lookback, horizon) slice to plot. 0 = first valid slice.")
    args = ap.parse_args()

    series = _load_series(args.data)
    start = args.sample_idx
    end = start + args.seq_len + args.pred_len
    if end > len(series):
        raise ValueError(f"sample_idx too large; series has {len(series)} points, "
                         f"requested {end}")
    lookback = series[start:start + args.seq_len]
    horizon = series[start + args.seq_len:end]

    # ---------- plotting ----------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    # top: time series plot with lookback / horizon highlighted
    ax_ts = fig.add_subplot(gs[0, :])
    t = np.arange(start, end)
    ax_ts.plot(t, series[start:end], color="#c0392b", linewidth=1.6, label=f"{args.data} series")
    ax_ts.axvspan(start, start + args.seq_len, color="#8ab4f8", alpha=0.2,
                  label=f"lookback ({args.seq_len})")
    ax_ts.axvspan(start + args.seq_len, end, color="#f9cb9c", alpha=0.25,
                  label=f"horizon ({args.pred_len})")
    ax_ts.set_title("(a) A time series and its lookback / horizon windows", fontsize=12)
    ax_ts.set_xlabel("time")
    ax_ts.set_ylabel("value")
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.grid(True, alpha=0.3)

    # bottom-left: distribution of lookback vs horizon (KDE)
    ax_kde = fig.add_subplot(gs[1, 0])
    # use histograms with density=True to approximate the KDE
    bins = np.linspace(min(lookback.min(), horizon.min()),
                       max(lookback.max(), horizon.max()), 40)
    ax_kde.hist(lookback, bins=bins, density=True, alpha=0.5,
                color="#1f77b4", label="lookback", edgecolor="black")
    ax_kde.hist(horizon, bins=bins, density=True, alpha=0.5,
                color="#ff7f0e", label="horizon", edgecolor="black")
    ax_kde.axvline(lookback.mean(), color="#1f77b4", linestyle="--",
                   label=f"lookback mean = {lookback.mean():.2f}")
    ax_kde.axvline(horizon.mean(), color="#ff7f0e", linestyle="--",
                   label=f"horizon mean = {horizon.mean():.2f}")
    ax_kde.set_title("(b) Empirical distribution of lookback vs horizon", fontsize=12)
    ax_kde.set_xlabel("value")
    ax_kde.set_ylabel("density")
    ax_kde.legend(fontsize=9)
    ax_kde.grid(True, alpha=0.3)

    # bottom-right: boxplots for a side-by-side comparison
    ax_box = fig.add_subplot(gs[1, 1])
    ax_box.boxplot([lookback, horizon], labels=["lookback", "horizon"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#a8d1ff", color="black"),
                   medianprops=dict(color="red", linewidth=2))
    ax_box.set_title("(c) Box plot comparison", fontsize=12)
    ax_box.set_ylabel("value")
    ax_box.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Figure 1 reproduction: distribution shift between "
                 f"lookback ({args.seq_len}) and horizon ({args.pred_len}) on {args.data}",
                 fontsize=13, y=1.0)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[fig1] Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
