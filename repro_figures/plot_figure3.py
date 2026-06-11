"""Plot a "Figure 3 -- MSE vs lookback/pred_len per alpha" style plot.

Reads a CSV produced by train.py (results/figure3_runs.csv), which has columns:

    dataset, seq_len, pred_len, model, norm, alpha, MSE, MAE, timestamp

The script aggregates the rows, picks a dataset / model / norm group, and plots
MSE (or MSE z-score) against the prediction horizon for each alpha.

Usage
-----
    # Use defaults (dataset=ETTm2, model=Autoformer, norm=dists)
    python3 repro_figures/plot_figure3.py --input results/figure3_runs.csv

    # Fully specified:
    python3 repro_figures/plot_figure3.py \
        --input results/figure3_runs.csv \
        --dataset ETTm2 --model Autoformer --norm dishts \
        --output results/figures/figure3_ETTm2.png \
        --zscore

    # Overlay paper reference from `paper_results/table3_revin_comparison.csv`
    # or similar reference CSVs:
    python3 repro_figures/plot_figure3.py \
        --input results/figure3_runs.csv \
        --paper paper_results/table3_revin_comparison.csv \
        --output results/figures/figure3_ETTm2_with_paper.png

Outputs
-------
    <output>.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names: allow either "pred_len" or "horizon"
    rename_map = {}
    for col in df.columns:
        if col.lower() in ("horizon", "window", "length") and "pred_len" not in df.columns:
            rename_map[col] = "pred_len"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def plot_alpha_sensitivity(df: pd.DataFrame, dataset: str, model: str, norm: str,
                           zscore: bool, title: str) -> plt.Figure:
    sub = df[(df["dataset"] == dataset) & (df["model"] == model) & (df["norm"] == norm)].copy()
    if sub.empty:
        raise ValueError(
            f"No rows for dataset={dataset} / model={model} / norm={norm}. "
            f"Available: {df[['dataset', 'model', 'norm']].drop_duplicates().to_string(index=False)}"
        )
    sub = sub.sort_values("alpha")
    alphas = sorted(sub["alpha"].unique())

    # One line per alpha: x = pred_len, y = MSE (or z-score).
    # To handle multiple runs at same (alpha, pred_len), take the mean.
    grouped = sub.groupby(["alpha", "pred_len"], as_index=False)["MSE"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = DEFAULT_PALETTE[:max(1, len(alphas))]

    y_col = "MSE"
    if zscore:
        mu, sd = grouped["MSE"].mean(), grouped["MSE"].std()
        if sd > 0:
            grouped["MSE_z"] = (grouped["MSE"] - mu) / sd
            y_col = "MSE_z"

    for idx, alpha in enumerate(alphas):
        seg = grouped[grouped["alpha"] == alpha].sort_values("pred_len")
        ax.plot(seg["pred_len"], seg[y_col], marker="o", linewidth=2.0,
                color=palette[idx], label=rf"$\alpha$ = {alpha:g}")

    ax.set_xlabel("pred_len (lookback = pred_len, typically)")
    ax.set_ylabel("MSE (z-score)" if zscore else "MSE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="alpha", loc="best", fontsize=10)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(ROOT, "results", "figure3_runs.csv"),
                    help="CSV of runs, one row per (dataset, model, norm, alpha, pred_len, MSE).")
    ap.add_argument("--dataset", default="ETTm2", help="Dataset to plot.")
    ap.add_argument("--model", default="Autoformer", help="Backbone model.")
    ap.add_argument("--norm", default="dishts", help="Norm method to plot (typically dishts).")
    ap.add_argument("--output", default=None,
                    help="Output PNG path (default: results/figures/figure3_<dataset>.png).")
    ap.add_argument("--zscore", action="store_true",
                    help="Normalise MSE to z-score (original paper Figure 3 uses z-score).")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    df = _load(args.input)
    title = (args.title
             or f"Figure 3 style -- {args.dataset} / {args.model} / norm={args.norm}")
    fig = plot_alpha_sensitivity(df, args.dataset, args.model, args.norm,
                                 args.zscore, title)

    if args.output is None:
        out_dir = os.path.join(ROOT, "results", "figures")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"figure3_{args.dataset}.png")
    else:
        out_png = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_figure3] saved -> {out_png}")


if __name__ == "__main__":
    main()
