"""Plot a "Figure 4 -- qualitative forecast comparison" from saved CSV samples.

Train.py, at the end of each run, writes a file like:

    results/figures/figure4_ETTm2_Autoformer_dishts_sq96_pd96_alpha0.5_seed2021.csv

with columns: timestep, dataset, model, norm, seq_len, pred_len, alpha, seed,
gt_0, gt_1, ..., pred_0, pred_1, ...

This script takes 2 such files (backbone + Dish-TS) and produces the two-row
Figure-4-style plot. Optionally overlays a RevIN run as a third line.

Usage
-----
    python3 repro_figures/plot_figure4.py \
        --backbone results/figures/figure4_ETTm2_Autoformer_none_*.csv \
        --dishts  results/figures/figure4_ETTm2_Autoformer_dishts_*.csv \
        --revin   results/figures/figure4_ETTm2_Autoformer_revin_*.csv \
        --channel 0 --output results/figures/figure4_ETTm2.png

If you only have backbone and dishts, simply omit --revin.

Outputs
-------
    <output>.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _resolve(path: str | None) -> str | None:
    if path is None or not path:
        return None
    if "*" in path:
        matches = sorted(glob.glob(path))
        if not matches:
            raise FileNotFoundError(f"no matches for glob: {path}")
        if len(matches) > 1:
            print(f"[plot_figure4] info: {path} matched {len(matches)} files, "
                  f"using latest: {matches[-1]}")
        return matches[-1]
    return path


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_row(ax: plt.Axes, gt: np.ndarray, pred: np.ndarray, seq_len: int,
             pred_len: int, method_label: str, method_color: str,
             sample_label: str):
    full_len = seq_len + pred_len
    if len(gt) != full_len or len(pred) != full_len:
        raise ValueError(
            f"expected length {full_len}, got gt={len(gt)} pred={len(pred)}"
        )
    t = np.arange(full_len)
    ax.plot(t, gt, color="#c0392b", linestyle="-.", linewidth=1.8,
            label="ground truth" if ax.get_legend() is None else None)
    # prediction starts at seq_len, so we plot from seq_len onwards; earlier
    # timesteps are NaN and will show as a gap.
    mask = np.arange(full_len) >= seq_len
    ax.plot(t[mask], pred[mask], color=method_color, linewidth=2.0,
            label=method_label)
    ax.axvline(seq_len - 0.5, color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(sample_label)
    ax.set_ylabel("value")
    ax.legend(loc="upper right", fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", required=True,
                    help="CSV path (or glob) for the backbone-only run (norm=none).")
    ap.add_argument("--dishts", required=True,
                    help="CSV path (or glob) for the Dish-TS run.")
    ap.add_argument("--revin", default=None,
                    help="Optional CSV path (or glob) for the RevIN run.")
    ap.add_argument("--channel", type=int, default=0,
                    help="Which column to plot (0 = first feature, useful for multivariate data).")
    ap.add_argument("--output", default=None,
                    help="Output PNG path (default: results/figures/figure4.png).")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    backbone_path = _resolve(args.backbone)
    dishts_path = _resolve(args.dishts)
    revin_path = _resolve(args.revin)

    df_bb = load(backbone_path)
    df_ds = load(dishts_path)
    df_rev = load(revin_path) if revin_path is not None else None

    # Validate shape consistency
    seq_len = int(df_bb["seq_len"].iloc[0])
    pred_len = int(df_bb["pred_len"].iloc[0])
    dataset = df_bb["dataset"].iloc[0]
    col = f"gt_{args.channel}"
    pred_col = f"pred_{args.channel}"
    if col not in df_bb.columns or pred_col not in df_bb.columns:
        raise KeyError(
            f"expected columns '{col}' and '{pred_col}'. "
            f"Available: {list(df_bb.columns)}"
        )

    gt_bb = df_bb[col].values.astype(float)
    # For Figure 4, all three rows share the SAME ground-truth (same dataset,
    # same sample_index=0), so we just use the backbone run's gt for both rows.
    pred_bb = df_bb[pred_col].values.astype(float)
    pred_ds = df_ds[pred_col].values.astype(float)
    pred_rev = df_rev[pred_col].values.astype(float) if df_rev is not None else None

    # Layout: row (a) backbone + dishts vs gt   row (b) revin + dishts vs gt
    nrows = 2 if df_rev is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(15, 4.5 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    # Row (a): backbone vs dishts vs gt
    ax = axes[0]
    t = np.arange(seq_len + pred_len)
    mask_horizon = t >= seq_len
    ax.plot(t, gt_bb, color="#c0392b", linestyle="-.", linewidth=1.8,
            label="ground truth")
    ax.plot(t[mask_horizon], pred_bb[mask_horizon], color="#b5a017",
            linewidth=2.0, label="backbone (Autoformer)")
    ax.plot(t[mask_horizon], pred_ds[mask_horizon], color="#1f77b4",
            linewidth=2.2, label="Dish-TS (Autoformer)")
    ax.axvline(seq_len - 0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_title(
        args.title or f"(a) {dataset}  (seq_len={seq_len}, pred_len={pred_len})"
    )
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if df_rev is not None:
        ax = axes[1]
        ax.plot(t, gt_bb, color="#c0392b", linestyle="-.", linewidth=1.8,
                label="ground truth")
        ax.plot(t[mask_horizon], pred_rev[mask_horizon], color="#ff9933",
                linewidth=2.0, label="RevIN (Autoformer)")
        ax.plot(t[mask_horizon], pred_ds[mask_horizon], color="#1f77b4",
                linewidth=2.2, label="Dish-TS (Autoformer)")
        ax.axvline(seq_len - 0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_title("(b) RevIN vs Dish-TS vs ground truth")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("timestep")
    fig.suptitle("Figure 4 reproduction -- qualitative forecast comparison",
                 y=1.02, fontsize=13)
    fig.tight_layout()

    if args.output is None:
        out_dir = os.path.join(ROOT, "results", "figures")
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"figure4_{dataset}.png")
    else:
        out_png = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_figure4] saved -> {out_png}")


if __name__ == "__main__":
    main()
