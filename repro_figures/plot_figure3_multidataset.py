#!/usr/bin/env python3
"""Plot Figure 3 across 4 datasets in 2x2 subplots, matching paper layout.

For each dataset, plot MSE vs pred_len with 4 alpha curves.
Paper Figure 3 layout: 4 subplots, X=horizon, 4 alpha lines per subplot.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/figure3_runs.csv")
    ap.add_argument("--output", default="results/figures/figure3_multidataset.png")
    ap.add_argument("--datasets", nargs="+",
                    default=["ETTm2", "ETTh1", "ECL", "WTH"])
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--pred-lens", type=int, nargs="+",
                    default=[96, 168, 336])
    ap.add_argument("--min-n", type=int, default=2,
                    help="min seeds to include a (dataset, pred_len, alpha) cell")
    ap.add_argument("--prior", default="paper-phi-only",
                    help="Only keep rows with this prior (e.g. paper-phi-only, none)")
    args = ap.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} does not exist.", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))
    # Keep dishts only, and require a specific prior to avoid mixing experiment phases
    rows = [r for r in rows if r.get("norm") == "dishts" and r.get("prior") == args.prior]

    # aggregate: (dataset, pred_len, alpha) -> list of MSE
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

    # 2x2 layout matching paper Figure 3
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = {0.0: "tab:blue", 0.25: "tab:orange", 0.5: "tab:green",
              0.75: "tab:red", 1.0: "tab:purple"}
    markers = {0.0: "o", 0.25: "s", 0.5: "^", 0.75: "D", 1.0: "v"}

    for ax, ds in zip(axes.flat, args.datasets):
        for al in args.alphas:
            ys, errs = [], []
            for pl in args.pred_lens:
                arr = agg.get((ds, pl, al), [])
                if len(arr) >= args.min_n:
                    m = sum(arr) / len(arr)
                    v = sum((x - m) ** 2 for x in arr) / len(arr)
                    ys.append(m)
                    errs.append(math.sqrt(v))
                else:
                    ys.append(None)
                    errs.append(None)

            # filter to non-None
            xs_plot, ys_plot, errs_plot = [], [], []
            for pl, y, e in zip(args.pred_lens, ys, errs):
                if y is not None:
                    xs_plot.append(pl)
                    ys_plot.append(y)
                    errs_plot.append(e)

            if not xs_plot:
                continue

            ax.errorbar(xs_plot, ys_plot, yerr=errs_plot,
                        label=f"α={al}",
                        color=colors.get(al, "gray"),
                        marker=markers.get(al, "x"),
                        linewidth=2, markersize=7, capsize=3)

        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.set_xlabel("Horizon length", fontsize=10)
        ax.set_ylabel("MSE", fontsize=10)
        ax.set_xticks(args.pred_lens)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Figure 3 Reproduction — Alpha Sensitivity (4 datasets, Autoformer, dishts)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
