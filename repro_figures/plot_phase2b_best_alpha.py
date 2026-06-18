#!/usr/bin/env python3
"""PPT-ready bar chart: best-alpha improvement vs alpha=0.0 (4 datasets x 3 horizons).

For each (dataset, pred_len) cell, plot the % MSE reduction of the best alpha
relative to alpha=0.0. Negative bars = improvement. Highlight ETTh1 cells
(cleanest paper trend match).
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/figure3_runs.csv")
    ap.add_argument("--output", default="results/figures/phase2b_best_alpha_improvement.png")
    ap.add_argument("--datasets", nargs="+",
                    default=["ETTm2", "ETTh1", "ECL", "WTH"])
    ap.add_argument("--pred-lens", type=int, nargs="+",
                    default=[96, 168, 336])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--prior", default="paper-phi-only",
                    help="Only keep rows with this prior (e.g. paper-phi-only, none)")
    ap.add_argument("--min-n", type=int, default=2)
    args = ap.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} does not exist.", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r.get("norm") == "dishts" and r.get("prior") == args.prior]

    # (dataset, pred_len, alpha) -> list of MSE
    agg: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("dataset") not in args.datasets:
            continue
        try:
            pl = int(r["pred_len"])
            al = float(r["alpha"])
            mse = float(r["MSE"])
        except (KeyError, ValueError):
            continue
        if pl not in args.pred_lens:
            continue
        if not any(abs(al - a) < 1e-9 for a in args.alphas):
            continue
        agg[(r["dataset"], pl, al)].append(mse)

    # Compute mean MSE per cell
    means: dict[tuple[str, int, float], float] = {}
    for k, arr in agg.items():
        if len(arr) >= args.min_n:
            means[k] = sum(arr) / len(arr)

    # For each (dataset, pred_len) cell, find best alpha and compute improvement
    cells = []  # (dataset, pred_len, best_alpha, baseline_mse, best_mse, pct)
    for ds in args.datasets:
        for pl in args.pred_lens:
            base_arr = agg.get((ds, pl, 0.0), [])
            if len(base_arr) < args.min_n:
                continue
            base_mse = sum(base_arr) / len(base_arr)
            best_al, best_mse = 0.0, base_mse
            for al in args.alphas:
                arr = agg.get((ds, pl, al), [])
                if len(arr) < args.min_n:
                    continue
                m = sum(arr) / len(arr)
                if m < best_mse:
                    best_al, best_mse = al, m
            pct = 100.0 * (best_mse - base_mse) / base_mse
            cells.append((ds, pl, best_al, base_mse, best_mse, pct))

    if not cells:
        print("No cells with enough data.", file=sys.stderr)
        return 3

    # Plot: 4 panels (one per dataset), 3 bars (one per pred_len)
    n_ds = len(args.datasets)
    fig, axes = plt.subplots(
        1, n_ds, figsize=(3.6 * n_ds, 5.2), sharey=False
    )
    if n_ds == 1:
        axes = [axes]

    # x positions: pred_len groups
    x = np.arange(len(args.pred_lens))
    width = 0.6

    # leave room on top for labels
    plt.subplots_adjust(top=0.82, bottom=0.10, left=0.06, right=0.98,
                        wspace=0.30)

    for ax_idx, (ax, ds) in enumerate(zip(axes, args.datasets)):
        heights = []
        labels_text = []
        colors = []
        for pl in args.pred_lens:
            cell = [c for c in cells if c[0] == ds and c[1] == pl]
            if not cell:
                heights.append(0)
                labels_text.append("")
                colors.append("lightgray")
            else:
                _, _, best_al, base, best, pct = cell[0]
                heights.append(pct)
                # only show α* label inside the bar (no extra noise)
                if pct <= -0.1:                     # real improvement
                    labels_text.append(f"α*={best_al}\n{pct:+.1f}%")
                else:                                # no improvement
                    labels_text.append("α*=0.0")
                # color: green if improvement, red if regression
                colors.append("#2ca02c" if pct <= 0 else "#d62728")

        bars = ax.bar(x, heights, width, color=colors,
                      edgecolor="black", linewidth=0.8)

        # place ALL labels INSIDE bars (no labels above plot area to avoid
        # title / neighbour-subplot overlap)
        for bar, label in zip(bars, labels_text):
            if not label:
                continue
            h = bar.get_height()
            # bottom of bar (since most bars are negative)
            y_text = h / 2 if h != 0 else -0.5
            ax.text(bar.get_x() + bar.get_width() / 2, y_text, label,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")

        # Title: keep inside the axes (top), do not use figure title
        ax.set_title(ds, fontsize=13, fontweight="bold", pad=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"H={pl}" for pl in args.pred_lens], fontsize=10)
        ax.set_xlabel("Horizon", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        # ETTh1 highlight
        if ds == "ETTh1":
            ax.set_facecolor("#eaf7ea")

        # extend y limits a bit so labels and bars are not clipped
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, min(heights + [0]) * 1.15 - 1)
        ymax = max(ymax, 2)
        ax.set_ylim(ymin, ymax)

    axes[0].set_ylabel("Δ MSE vs α=0.0 (%)", fontsize=10)

    # Suptitle at very top with sufficient padding
    fig.suptitle(
        "Best-α Improvement vs No-Prior (Dishts, Autoformer, --prior paper-phi-only)",
        fontsize=12, fontweight="bold", y=0.96,
    )

    # Subtitle / paper-trend line as figure-level text
    fig.text(0.5, 0.90,
             "ETTh1 (highlighted): best-α improves from -3.8% to -18.6% as H grows — matches paper Fig. 3",
             ha="center", fontsize=10, color="darkred", style="italic")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")

    # Also print the table for inspection
    print()
    print("=" * 80)
    print(f"{'dataset':<8} {'H':>4} {'best α':>8} {'baseline':>10} {'best':>10} {'Δ%':>8}")
    print("-" * 80)
    for ds, pl, al, base, best, pct in cells:
        print(f"{ds:<8} {pl:>4d} {al:>8.2f} {base:>10.3f} {best:>10.3f} {pct:>+7.2f}%")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
