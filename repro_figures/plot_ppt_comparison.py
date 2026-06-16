"""Generate PPT-ready comparison figures: our reproduction vs paper.

Produces two PNGs in results/figures/:
  1. ppt_figure3_alpha_ETTm2.png  — alpha-sensitivity curve (our Figure 3 reproduction)
  2. ppt_dishts_vs_paper.png      — bar chart: our dishts MSE vs paper dishts MSE

Usage:
    python3 repro_figures/plot_ppt_comparison.py
"""

from __future__ import annotations
import csv, math, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "figure3_runs.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- paper reference data (Table 2) ----
PAPER_TABLE2 = {  # Autoformer, multivariate, dish-TS, paper-scale
    "ETTm2":  {24: 0.676, 48: 0.823, 96: 0.929, 168: 1.308, 336: 1.603},
    "ETTh1":  {24: 1.019, 48: 1.240, 96: 1.199, 168: 1.148, 336: 1.147},
    "WTH":    {24: 2.481, 48: 4.299, 96: 3.280, 168: 3.309, 336: 3.314},
    "ECL":    {24: 0.040, 48: 0.051, 96: 0.064, 168: 0.080, 336: 0.104},
}

# Per-dataset paper scale factors (multivariate), fitted by compare_paper.py
# raw MSE  ->  paper-scale MSE  =  raw_MSE * mse_factor
SCALE_MULTI = {
    "ETTm2":   9.4e-2,
    "ETTh1":   3.0e-2,
    "WTH":     9.3e-4,
    "ECL":     4.0e-2,
    "Electricity": 4.0e-2,
    "Weather":     9.3e-4,
}


def load_rows(path: Path) -> list[dict]:
    """Load ALL dishts rows (any alpha, any prior). Returns list with raw MSE."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["pred_len"] = int(r["pred_len"])
                r["alpha"] = float(r["alpha"])
                r["MSE"] = float(r["MSE"])
            except (KeyError, ValueError):
                continue
            if r["norm"] != "dishts" or math.isnan(r["MSE"]) or math.isinf(r["MSE"]):
                continue
            rows.append(r)
    return rows


# =============================================================
# Figure A: Alpha sensitivity curve (ETTm2)
# =============================================================
def plot_alpha_sensitivity(rows: list[dict]) -> Path:
    """One subplot per pred_len in Phase 2a target (96, 168, 336)."""
    target_ds = "ETTm2"
    preds = [96, 168, 336]

    fig, axes = plt.subplots(1, len(preds), figsize=(5.0 * len(preds), 3.8))
    colors = ["#2c7bb6", "#fdae61", "#d7191c"]

    for ax, pl, color in zip(axes, preds, colors):
        # Group by alpha, average over seeds
        alpha_vals = defaultdict(list)
        for r in rows:
            if r["dataset"] == target_ds and r["pred_len"] == pl:
                alpha_vals[r["alpha"]].append(r["MSE"])
        if not alpha_vals:
            ax.text(0.5, 0.5, f"no data for pred_len={pl}", ha="center")
            continue

        alphas = sorted(alpha_vals)
        means = [np.mean(alpha_vals[a]) for a in alphas]
        stds  = [np.std(alpha_vals[a]) for a in alphas]

        ax.errorbar(alphas, means, yerr=stds, fmt="o", color=color,
                    capsize=4, markersize=6, elinewidth=1.2)
        ax.plot(alphas, means, "-", color=color, linewidth=2, alpha=0.8)

        best_idx = int(np.argmin(means))
        best_a, best_m = alphas[best_idx], means[best_idx]
        ax.annotate(f"best a={best_a}",
                    xy=(best_a, best_m),
                    xytext=(best_a + 0.1, best_m + 0.025 * best_m),
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        # Compare with alpha=0.0
        if 0.0 in alpha_vals and best_a != 0.0:
            base_m = np.mean(alpha_vals[0.0])
            gain = (base_m - best_m) / base_m * 100
            ax.axhline(base_m, ls=":", color="gray", lw=0.8, alpha=0.6)
            ax.text(0.99, 0.97, f"a=0 MSE={base_m:.2f}\nbest a={best_a}: {gain:+.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="lightgray", alpha=0.9))

        ax.set_xlabel("alpha (prior weight)", fontsize=10)
        ax.set_ylabel("MSE (mean over seeds)", fontsize=10)
        ax.set_title(f"pred_len = {pl}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Figure 3 Reproduction - Alpha Sensitivity (ETTm2, Autoformer, dishts)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "ppt_figure3_alpha_ETTm2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  v {out}")
    return out


# =============================================================
# Figure B: Bar chart - our BEST Dish-TS vs paper Dish-TS MSE
# =============================================================
def plot_dishts_vs_paper(rows: list[dict]) -> Path:
    """For each (dataset, pred_len), use OUR BEST alpha's mean MSE (scaled),
    vs PAPER's Autoformer+DishTS MSE. Green/Red color encodes ratio.
    """
    # 1. Compute our best (dataset, pred_len) -> (best_alpha, best_mse_raw, n_seeds)
    cell_best = {}
    for ds in ["ETTm2", "WTH", "ETTh1", "ECL"]:
        for pl in [24, 96, 168, 336]:
            cell = defaultdict(list)
            for r in rows:
                if r["dataset"] == ds and r["pred_len"] == pl:
                    cell[r["alpha"]].append(r["MSE"])
            if not cell:
                continue
            # Best alpha for this (ds, pl)
            best_a = min(cell, key=lambda a: np.mean(cell[a]))
            best_mse_raw = float(np.mean(cell[best_a]))
            n_seeds = len(cell[best_a])
            cell_best[(ds, pl)] = (best_a, best_mse_raw, n_seeds)

    # 2. Build plot
    datasets = ["ETTm2", "WTH", "ETTh1"]
    common_lens = [24, 96, 168, 336]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4.2))

    for ax, ds in zip(axes, datasets):
        x_labels, ours_vals, paper_vals, ratios, annotations = [], [], [], [], []

        for pl in common_lens:
            paper_val = PAPER_TABLE2[ds].get(pl)
            cell = cell_best.get((ds, pl))
            if paper_val is None or cell is None:
                continue
            best_a, best_mse_raw, n_seeds = cell
            scale = SCALE_MULTI.get(ds, 1.0)
            our_val_scaled = best_mse_raw * scale

            x_labels.append(f"H={pl}")
            ours_vals.append(our_val_scaled)
            paper_vals.append(paper_val)
            ratio = our_val_scaled / paper_val
            ratios.append(ratio)
            annotations.append((best_a, n_seeds, ratio))

        x = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, ours_vals, width, label="Our Dish-TS (best alpha)",
                       color="#2c7bb6", edgecolor="white", linewidth=0.8)
        bars2 = ax.bar(x + width/2, paper_vals, width, label="Paper Dish-TS",
                       color="#bababa", edgecolor="gray", linewidth=0.8, hatch="//")

        # ratio + best alpha labels
        for i, (xi, ov, pv, (best_a, n, rat)) in enumerate(zip(x, ours_vals, paper_vals, annotations)):
            color = "green" if 0.95 <= rat <= 1.05 else ("darkorange" if 0.85 <= rat <= 1.15 else "red")
            ax.text(xi, max(ov, pv) * 1.07, f"{rat:.2f}x",
                    ha="center", fontsize=8, fontweight="bold", color=color)
            ax.text(xi, max(ov, pv) * 0.04, f"alpha={best_a}\n(n={n})",
                    ha="center", fontsize=7, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(f"{ds}  (best alpha per horizon)", fontsize=11, fontweight="bold")
        ax.set_ylabel("MSE (paper-scale)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if ds == datasets[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Dish-TS Reproduction vs Paper (Autoformer, multivariate, paper-scale)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "ppt_dishts_vs_paper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  v {out}")
    return out


# =============================================================
def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        return 2

    rows = load_rows(CSV_PATH)
    print(f"Loaded {len(rows)} dishts rows from {CSV_PATH}")

    # print best per (dataset, pred_len) for transparency
    print("\nBest (alpha) per (dataset, pred_len):")
    print(f"  {'dataset':>8s} {'pred_len':>9s} {'best_a':>7s} {'raw_MSE':>10s} {'scaled_MSE':>11s} {'paper':>10s} {'ratio':>6s}")
    for ds in ["ETTm2", "WTH", "ETTh1", "ECL"]:
        for pl in [24, 96, 168, 336]:
            cell = defaultdict(list)
            for r in rows:
                if r["dataset"] == ds and r["pred_len"] == pl:
                    cell[r["alpha"]].append(r["MSE"])
            if not cell:
                continue
            best_a = min(cell, key=lambda a: np.mean(cell[a]))
            best_raw = float(np.mean(cell[best_a]))
            scaled = best_raw * SCALE_MULTI.get(ds, 1.0)
            paper_v = PAPER_TABLE2.get(ds, {}).get(pl)
            ratio = scaled / paper_v if paper_v else None
            ratio_s = f"{ratio:.2f}x" if ratio else "-"
            paper_s = f"{paper_v:.3f}" if paper_v else "-"
            print(f"  {ds:>8s} {pl:>9d} {best_a:>7.2f} {best_raw:>10.4f} {scaled:>11.4f} {paper_s:>10s} {ratio_s:>6s}")

    # ---- Alpha sensitivity ----
    plot_alpha_sensitivity(rows)

    # ---- Bar chart (best alpha per cell, scaled to paper space) ----
    plot_dishts_vs_paper(rows)

    print("\nDone. Files in", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
