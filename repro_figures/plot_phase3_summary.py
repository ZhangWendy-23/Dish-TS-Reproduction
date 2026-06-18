"""Disht-TS vs RevIN vs none comparison (Phase 3) — 4-subplot bar chart for the report.

Generates results/figures/phase3_dishts_vs_revin_vs_none.png with 4 subplots
(one per dataset), each showing 3-norm comparison (dishts-α=0, revin, none)
at H ∈ {24, 96, 168, 336}.

For each (dataset, H) we report:
- Bar height: mean MSE (3 seeds)
- Error bar: ±1 std
- Color: blue=dishts, orange=revin, gray=none
- Numeric label on top of each bar
- Best-of-3 (fair, all α=0) marked with green checkmark
"""
import csv
from collections import defaultdict
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Load all runs
cells = defaultdict(lambda: {"none": [], "revin": [], "dishts": []})
with open("results/figure3_runs.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        ds = r["dataset"]
        norm = r["norm"]
        pl = int(r["pred_len"])
        try:
            mse = float(r["MSE"])
        except (KeyError, ValueError):
            continue
        if mse != mse or mse > 1e8:  # NaN or absurd
            continue
        if norm not in cells[(ds, pl)]:
            continue
        cells[(ds, pl)][norm].append(mse)

# Order
datasets = ["ETTm2", "WTH", "ETTh1", "ECL"]
horizons = [24, 96, 168, 336]
NORM_COLORS = {"dishts": "#4A90D9", "revin": "#E89B5A", "none": "#999999"}
NORM_LABELS = {"dishts": "Disht-TS (α=0)", "revin": "RevIN", "none": "None"}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=False)
fig.suptitle("Disht-TS vs RevIN vs None (Autoformer, Multivariate, Phase 3)",
             fontsize=13, fontweight="bold", y=1.02)

x = np.arange(len(horizons))
width = 0.26

wins = {"dishts": 0, "revin": 0, "none": 0, "tie": 0}

for ax, ds in zip(axes, datasets):
    means = {n: [] for n in NORM_LABELS}
    stds = {n: [] for n in NORM_LABELS}
    ns = {n: [] for n in NORM_LABELS}
    for pl in horizons:
        for n in NORM_LABELS:
            arr = cells.get((ds, pl), {}).get(n, [])
            means[n].append(mean(arr) if arr else 0)
            stds[n].append(stdev(arr) if len(arr) > 1 else 0)
            ns[n].append(len(arr))
    # Plot
    for i, n in enumerate(NORM_LABELS):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means[n], width, yerr=stds[n],
                       color=NORM_COLORS[n], label=NORM_LABELS[n],
                       edgecolor="black", linewidth=0.4, capsize=2,
                       error_kw={"lw": 0.6})
        for j, (b, m) in enumerate(zip(bars, means[n])):
            if m > 0:
                ax.text(b.get_x() + b.get_width()/2, m * 1.05, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=6, rotation=0)
    # Mark best (only when 3 norms have data)
    for j, pl in enumerate(horizons):
        fair = []
        for n in NORM_LABELS:
            v = means[n][j]
            if v > 0:
                fair.append((n, v))
        if len(fair) == 3:
            m_min = min(v for _, v in fair)
            tied = [n for n, v in fair if abs(v - m_min) < 1e-3]
            if len(tied) == 1:
                wins[tied[0]] += 1
            else:
                wins["tie"] += 1
            winner = tied[0]
            color_winner = NORM_COLORS[winner]
            ax.text(j, 0.96, "★", ha="center", va="top", fontsize=14,
                    color=color_winner, transform=ax.get_xaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([f"H={pl}" for pl in horizons], fontsize=9)
    ax.set_title(ds, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("MSE" if ds == datasets[0] else "", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    # Reserve 30% top headroom for numeric labels
    ymax = max(max(m) for m in means.values() if any(m)) if any(any(m) for m in means.values()) else 1
    ax.set_ylim(top=ymax * 1.30)
    if ds == datasets[0]:
        ax.legend(fontsize=8, loc="upper left")

# Total cells
total = sum(wins.values())
win_str = f"Fair wins: dishts={wins['dishts']}/{total}, revin={wins['revin']}, none={wins['none']}, ties={wins['tie']}"
fig.text(0.5, -0.02, win_str, ha="center", fontsize=10, style="italic")

fig.tight_layout()
out = "results/figures/phase3_dishts_vs_revin_vs_none.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(win_str)
