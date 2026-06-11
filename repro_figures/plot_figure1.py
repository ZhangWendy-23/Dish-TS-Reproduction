"""Plot a "Figure 1 -- distribution shift" style figure from raw time-series data.

Usage
-----
    # From a dataset CSV (e.g. data/Weather/Weather.csv):
    python3 repro_figures/plot_figure1.py \
        --input  data/Weather/Weather.csv \
        --output results/figures/figure1_weather.png \
        --seq_len 96 --pred_len 96 --sample_idx 0 \
        --column OT

    # --paper-mode additionally saves a "reference lookback/horizon mean and std"
    # text snapshot in a sibling .txt so you can reproduce the same window later.

Outputs
-------
    <output>.png   (3-panel plot: time-series + histograms + boxplot)
    <output>.json  (sample_idx, lookback / horizon mean & std)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_series(path: str, column: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    # Drop non-numeric columns (date / datetime)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if column is None:
        column = "OT" if "OT" in numeric_cols else numeric_cols[-1]
    return df[column].values.astype(float), column


def plot_figure1(series: np.ndarray, seq_len: int, pred_len: int,
                 sample_idx: int, title: str) -> tuple[plt.Figure, dict]:
    start = int(sample_idx)
    end = start + seq_len + pred_len
    if end > len(series):
        raise ValueError(
            f"sample_idx={sample_idx} too large for series of length {len(series)} "
            f"(need {end} points)."
        )
    lookback = series[start:start + seq_len]
    horizon = series[start + seq_len:end]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    # (a) Time-series with lookback / horizon highlighted
    ax_ts = fig.add_subplot(gs[0, :])
    t = np.arange(start, end)
    ax_ts.plot(t, series[start:end], color="#c0392b", linewidth=1.6,
               label="series")
    ax_ts.axvspan(start, start + seq_len, color="#8ab4f8", alpha=0.22,
                  label=f"lookback ({seq_len})")
    ax_ts.axvspan(start + seq_len, end, color="#f9cb9c", alpha=0.30,
                  label=f"horizon ({pred_len})")
    ax_ts.set_title("(a) time series / lookback window / horizon window", fontsize=12)
    ax_ts.set_xlabel("timestep")
    ax_ts.set_ylabel("value")
    ax_ts.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax_ts.grid(True, alpha=0.3)

    # (b) Histogram comparison
    ax_hist = fig.add_subplot(gs[1, 0])
    bins = np.linspace(min(lookback.min(), horizon.min()),
                       max(lookback.max(), horizon.max()), 40)
    ax_hist.hist(lookback, bins=bins, density=True, alpha=0.55,
                 color="#1f77b4", label="lookback", edgecolor="black")
    ax_hist.hist(horizon, bins=bins, density=True, alpha=0.55,
                 color="#ff7f0e", label="horizon", edgecolor="black")
    ax_hist.axvline(lookback.mean(), color="#1f77b4", linestyle="--", linewidth=1.5,
                    label=f"lookback mean={lookback.mean():.2f}")
    ax_hist.axvline(horizon.mean(), color="#ff7f0e", linestyle="--", linewidth=1.5,
                    label=f"horizon mean={horizon.mean():.2f}")
    ax_hist.set_title("(b) empirical distribution: lookback vs horizon", fontsize=12)
    ax_hist.set_xlabel("value")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # (c) Boxplot comparison
    ax_box = fig.add_subplot(gs[1, 1])
    ax_box.boxplot([lookback, horizon], labels=["lookback", "horizon"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#a8d1ff", edgecolor="black"),
                   medianprops=dict(color="red", linewidth=2))
    ax_box.set_title("(c) boxplot comparison", fontsize=12)
    ax_box.set_ylabel("value")
    ax_box.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()

    stats = {
        "sample_idx": sample_idx,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "lookback": {"mean": float(lookback.mean()),
                     "std": float(lookback.std()),
                     "min": float(lookback.min()),
                     "max": float(lookback.max())},
        "horizon": {"mean": float(horizon.mean()),
                    "std": float(horizon.std()),
                    "min": float(horizon.min()),
                    "max": float(horizon.max())},
    }
    return fig, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Dataset CSV path, e.g. data/Weather/Weather.csv")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Defaults to results/figures/figure1_<name>.png")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--column", default=None,
                    help="Which column to plot (default: OT or last numeric column).")
    args = ap.parse_args()

    series, column = _load_series(args.input, args.column)
    dataset_name = os.path.basename(args.input).replace(".csv", "")
    title = f"Figure 1 style -- {dataset_name} (column: {column})"

    fig, stats = plot_figure1(series, args.seq_len, args.pred_len,
                              args.sample_idx, title)

    if args.output is None:
        out_dir = os.path.join(ROOT, "results", "figures")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"figure1_{dataset_name}.png")
    else:
        out_png = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_figure1] saved PNG -> {out_png}")

    stats_file = os.path.splitext(out_png)[0] + ".json"
    with open(stats_file, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[plot_figure1] saved stats -> {stats_file}")


if __name__ == "__main__":
    main()
