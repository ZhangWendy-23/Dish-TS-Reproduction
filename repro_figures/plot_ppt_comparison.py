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

# ---- paper reference data (Table 2 + Table 3) ----
PAPER_TABLE2 = {  # Autoformer, multivariate, dish-TS
    "ETTm2":  {24: 0.676, 48: 0.823, 96: 0.929, 168: 1.308, 336: 1.603},
    "ETTh1":  {24: 1.019, 48: 1.240, 96: 1.199, 168: 1.148, 336: 1.147},
    "WTH":    {24: 2.481, 48: 4.299, 96: 3.280, 168: 3.309, 336: 3.314},
    "ECL":    {24: 0.040, 48: 0.051, 96: 0.064, 168: 0.080, 336: 0.104},
}

PAPER_REVIN = {  # Table 3 — RevIN baseline
    "ETTm2":  {24: 0.794, 168: 1.501, 336: 1.827},
    "ETTh1":  {24: 1.245, 168: 1.462, 336: 1.920},
    "WTH":    {24: 3.523, 168: 3.658, 336: 3.501},
    "ECL":    {24: 0.044, 168: 0.091, 336: 0.109},
}


def load_figure3_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["pred_len"] = int(r["pred_len"])
                r["alpha"] = float(r["alpha"])
                r["MSE"] = float(r["MSE"])
            except (KeyError, ValueError):
                continue
            if r["norm"] == "dishts" and not math.isnan(r["MSE"]):
                rows.append(r)
    return rows


def aggregate_alpha_sweep(rows: list[dict]) -> dict:
    """{(dataset, pred_len, alpha): {'mean':..., 'std':..., 'n':...}}"""
    raw = defaultdict(list)
    for r in rows:
        ds = r["dataset"]
        pl = r["pred_len"]
        al = r["alpha"]
        raw[(ds, pl, al)].append(r["MSE"])

    agg = {}
    for (ds, pl, al), vals in raw.items():
        arr = np.array(vals)
        agg[(ds, pl, al)] = {"mean": arr.mean(), "std": arr.std(), "n": len(vals)}
    return agg


def aggregate_dishts(rows: list[dict]) -> dict:
    """{(dataset, pred_len): mean_dishts_MSE} for alpha=0.0, prior=none rows."""
    raw = defaultdict(list)
    for r in rows:
        if abs(r["alpha"]) > 0.001:
            continue
        ds = r["dataset"]
        pl = r["pred_len"]
        raw[(ds, pl)].append(r["MSE"])
    agg = {}
    for (ds, pl), vals in raw.items():
        agg[(ds, pl)] = float(np.mean(vals))
    return agg


# =============================================================
# Figure A: Alpha sensitivity curve (ETTm2)
# =============================================================
def plot_alpha_sensitivity(agg: dict) -> Path:
    ds = "ETTm2"
    preds = sorted({k[1] for k in agg if k[0] == ds and k[1] != 24})
    # Only show 96, 168, 336 (Phase 2a target — 24 was Phase 1)
    if not preds:
        preds = sorted({k[1] for k in agg if k[0] == ds})

    fig, axes = plt.subplots(1, len(preds), figsize=(5.2 * len(preds), 3.8), sharey=False)
    if len(preds) == 1:
        axes = [axes]

    colors = ["#2c7bb6", "#fdae61", "#d7191c"]
    for ax, pl, color in zip(axes, preds, colors[:len(preds)]):
        alphas = sorted({k[2] for k in agg if k[0] == ds and k[1] == pl})
        means = [agg[(ds, pl, a)]["mean"] for a in alphas]
        stds  = [agg[(ds, pl, a)]["std"] for a in alphas]

        ax.errorbar(alphas, means, yerr=stds, fmt="o", color=color,
                    capsize=4, markersize=6, elinewidth=1.2, label="Our repro")
        ax.plot(alphas, means, "-", color=color, linewidth=2, alpha=0.8)

        # annotate best alpha
        best_a = alphas[int(np.argmin(means))]
        best_m = min(means)
        ax.annotate(f"best α={best_a}",
                    xy=(best_a, best_m), xytext=(best_a + 0.08, best_m + 0.02 * best_m),
                    fontsize=9, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("α (prior weight)", fontsize=10)
        ax.set_ylabel("MSE (mean over seeds)", fontsize=10)
        ax.set_title(f"pred_len = {pl}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Figure 3 Reproduction — Alpha Sensitivity (ETTm2, Autoformer)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "ppt_figure3_alpha_ETTm2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# =============================================================
# Figure B: Bar chart — our dishts vs paper dishts MSE
# =============================================================
def plot_dishts_vs_paper(our_mse: dict) -> Path:
    datasets = ["ETTm2", "WTH", "ETTh1"]
    # pred_lens we have data for from Phase 2a (alpha=0.0 rows)
    common_lens = [24, 96, 168, 336]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 4), sharey=False)

    colors_ours = ["#2c7bb6", "#2c7bb6", "#2c7bb6"]
    colors_paper = ["#bababa", "#bababa", "#bababa"]

    for ax, ds, co, cp in zip(axes, datasets, colors_ours, colors_paper):
        x_labels = []
        ours_vals = []
        paper_vals = []
        ratios = []

        for pl in common_lens:
            key = (ds, pl)
            if key in our_mse and pl in PAPER_TABLE2.get(ds, {}):
                x_labels.append(f"H={pl}")
                ours_vals.append(our_mse[key])
                paper_vals.append(PAPER_TABLE2[ds][pl])
                ratios.append(our_mse[key] / PAPER_TABLE2[ds][pl])

        x = np.arange(len(x_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, ours_vals, width, label="Our Dish-TS",
                       color=co, edgecolor="white", linewidth=0.8)
        bars2 = ax.bar(x + width/2, paper_vals, width, label="Paper Dish-TS",
                       color=cp, edgecolor="gray", linewidth=0.8, hatch="//")

        # ratio labels
        for i, (xi, ov, pv, rat) in enumerate(zip(x, ours_vals, paper_vals, ratios)):
            ax.text(xi, max(ov, pv) * 1.06, f"{rat:.2f}×",
                    ha="center", fontsize=8, fontweight="bold",
                    color="green" if 0.95 <= rat <= 1.05 else "darkorange")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(f"{ds}", fontsize=11, fontweight="bold")
        ax.set_ylabel("MSE", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if ds == datasets[0]:
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Dish-TS Reproduction vs Paper (Autoformer, multivariate, α=0)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = OUT_DIR / "ppt_dishts_vs_paper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# =============================================================
def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        return 2

    rows = load_figure3_rows(CSV_PATH)
    print(f"Loaded {len(rows)} dishts rows from {CSV_PATH}")

    # ---- Alpha sensitivity ----
    agg_alpha = aggregate_alpha_sweep(rows)
    plot_alpha_sensitivity(agg_alpha)

    # ---- Bar chart (alpha=0.0 only, for table comparison) ----
    agg_dishts = aggregate_dishts(rows)
    plot_dishts_vs_paper(agg_dishts)

    print("\nDone. Files in", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
