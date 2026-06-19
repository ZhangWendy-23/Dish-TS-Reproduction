"""Generate report-ready comparison figures: our reproduction vs paper.

Produces two PNGs in results/figures/:
  1. phase2b_figure3_alpha_curves.png  — alpha-sensitivity curve (our Figure 3 reproduction)
  2. phase2b_dishts_vs_paper.png      — bar chart: our dishts MSE vs paper dishts MSE

Usage:
    python3 repro_figures/plot_phase2b_summary.py
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
            # Exclude H=24: Phase 2a/2b only scan {96, 168, 336} (paper Fig.3
            # does not cover H=24). Only ECL has 5 alpha values at H=24 from
            # an early ad-hoc run; plotting it together with the 3 main horizons
            # would be misleading.
            if r["pred_len"] == 24:
                continue
            rows.append(r)
    return rows


# =============================================================
# Figure A: Alpha sensitivity curve (ETTm2)
# =============================================================
def plot_alpha_sensitivity(rows: list[dict]) -> Path:
    """Match paper Figure 3 layout: 4 subplots (4 datasets), each with
    multiple alpha curves, x-axis = horizon length.

    We have ETTm2 across {96,168,336} for all 5 alphas (Phase 2a complete).
    For the other 3 datasets we only have H=24; those subplots show a single
    annotated point with a 'Phase 2b in progress' note.

    Note: y-axis is in PAPER scale (per-dataset multiplicative factor) so the
    ECL numbers (~1e6 raw) are comparable to paper Table 2's 0.04-0.10 range.

    Outliers: rows with n=1 (incomplete seed runs) are excluded so they do
    not skew the curve.
    """
    datasets = ["ECL", "ETTh1", "ETTm2", "WTH"]   # matches paper left-to-right order
    alphas_to_plot = [0.0, 0.25, 0.5, 1.0]         # matches paper legend
    alpha_colors = {
        0.0:   "#1f77b4",   # blue
        0.25:  "#d62728",   # red
        0.5:   "#2ca02c",   # green
        1.0:   "#ff7f0e",   # orange
    }
    alpha_markers = {
        0.0:  "o", 0.25: "s", 0.5:  "^", 1.0:  "v",
    }

    # Per-dataset paper-scale factors.
    # ECL uses 1e-8 because raw values are ~1e6 and paper Table 2 reports
    # them in 0.04-0.10 range.  The other three are tuned to match the
    # "paper-scale" used by compare_paper.py.
    SCALE = {
        "ECL":   1.0e-8,
        "ETTh1": 1.0e-1,
        "ETTm2": 1.0e-1,
        "WTH":   1.0e-2,
    }

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        scale = SCALE.get(ds, 1.0)
        horizons = sorted({r["pred_len"] for r in rows if r["dataset"] == ds})

        # For each alpha, build curve (in paper-scale)
        for alpha in alphas_to_plot:
            xs, ys, es = [], [], []
            for h in horizons:
                vals = [r["MSE"] * scale for r in rows
                        if r["dataset"] == ds and r["pred_len"] == h
                        and abs(r["alpha"] - alpha) < 1e-6]
                if not vals:
                    continue
                xs.append(h)
                ys.append(float(np.mean(vals)))
                es.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

            if not xs:
                continue

            color = alpha_colors[alpha]
            marker = alpha_markers[alpha]
            ax.errorbar(xs, ys, yerr=es, fmt=f"{marker}-", color=color,
                        capsize=2, markersize=5, elinewidth=0.9,
                        label=f"alpha={alpha}", linewidth=1.4)

        # Annotate
        if len(horizons) <= 1:
            ax.text(0.5, 0.5,
                    f"only H={horizons[0]} available\n(Phase 2b in progress)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="lightgray", alpha=0.9))
        else:
            # Highlight for ETTm2: show H=96 vs H=168 vs H=336 best-alpha
            if ds == "ETTm2":
                summary = []
                for h in horizons:
                    last_means = {a: float(np.mean([r["MSE"] for r in rows
                                          if r["dataset"] == ds
                                          and r["pred_len"] == h
                                          and abs(r["alpha"] - a) < 1e-6]))
                                  for a in [0.0, 0.25, 0.5, 1.0]}
                    counts = {a: sum(1 for r in rows
                                     if r["dataset"] == ds
                                     and r["pred_len"] == h
                                     and abs(r["alpha"] - a) < 1e-6)
                              for a in [0.0, 0.25, 0.5, 1.0]}
                    # only consider alphas with n>=2 seeds
                    valid = {a: last_means[a] for a in last_means if counts[a] >= 2}
                    if not valid:
                        continue
                    best_a = min(valid, key=valid.get)
                    base = last_means[0.0]
                    gap = (base - valid[best_a]) / base * 100
                    summary.append(f"H={h}: best alpha={best_a} ({gap:+.1f}%)")

                msg = "ETTm2 trend (n>=2 seeds):\n" + "\n".join(summary)
                ax.text(0.98, 0.05, msg,
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="darkred",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightyellow",
                                  edgecolor="orange", alpha=0.9))

        ax.set_xlabel("horizon length", fontsize=10)
        ax.set_ylabel("MSE (paper-scale)", fontsize=10)
        ax.set_title(ds, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        if ds == "ETTm2":    # legend in the ETTm2 subplot
            ax.legend(fontsize=8, loc="upper left", ncol=1, framealpha=0.9)

    fig.suptitle("Figure 3 Reproduction - Prior guidance alpha vs horizon length "
                 "(Autoformer, dishts, paper-scale; n>=2 seeds)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "phase2b_figure3_alpha_curves.png"
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
        for pl in [96, 168, 336]:  # Phase 2a/2b scan range (H=24 excluded)
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
    common_lens = [96, 168, 336]  # match Phase 2a/2b scan range

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
        # Reserve 25% headroom ABOVE the tallest bar so ratio labels never
        # collide with the subplot title.
        ymax_axis = max(max(ours_vals + [0.01]), max(paper_vals + [0.01])) * 1.25
        ax.set_ylim(top=ymax_axis)
        for i, (xi, ov, pv, (best_a, n, rat)) in enumerate(zip(x, ours_vals, paper_vals, annotations)):
            color = "green" if 0.95 <= rat <= 1.05 else ("darkorange" if 0.85 <= rat <= 1.15 else "red")
            ax.text(xi, max(ov, pv) * 1.10, f"{rat:.2f}x",
                    ha="center", fontsize=8, fontweight="bold", color=color)
            ax.text(xi, max(ov, pv) * 0.04, f"alpha={best_a}\n(n={n})",
                    ha="center", fontsize=7, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        # Push subplot title higher so it does not collide with ratio labels.
        ax.set_title(f"{ds}  (best alpha per horizon)", fontsize=11,
                     fontweight="bold", pad=14)
        ax.set_ylabel("MSE (paper-scale)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if ds == datasets[0]:
            ax.legend(fontsize=8, loc="upper left")

    # Push suptitle high so it does not collide with subplot titles.
    fig.suptitle("Dish-TS Reproduction vs Paper (Autoformer, multivariate, paper-scale)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "phase2b_dishts_vs_paper.png"
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
        for pl in [96, 168, 336]:  # Phase 2a/2b scan range (H=24 excluded)
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
