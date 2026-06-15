"""Figure 3 — alpha-sensitivity curves.

Usage
-----
    python3 repro_figures/plot_figure3.py [--input CSV] [--output DIR]

Reads ``results/figure3_runs.csv`` and produces per-dataset PNG figures
showing mean MSE across seeds as a function of alpha, one subplot per
``pred_len``.  The target visual style is a close match to the paper's
Figure 3 so the image can be dropped directly into a reproduction report.

Output files
------------
results/figures/figure3_<dataset>.png
    One PNG per dataset present in the CSV.  Contains one subplot per
    prediction horizon, with alpha on the x-axis and mean MSE on the
    y-axis.  Error bars show +/-1 standard deviation across seeds.
results/figures/figure3_summary.csv
    Machine-readable copy of the plotted data, useful when assembling
    a LaTeX table.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "results" / "figure3_runs.csv"
DEFAULT_OUT = ROOT / "results" / "figures"


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


def _aggregate(rows: list[dict]) -> dict:
    """Return a dict {(dataset, pred_len, alpha): {'mean':float,'std':float,'n':int}}."""
    raw: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("norm") != "dishts":
            continue
        try:
            ds = r["dataset"]
            pl = int(r["pred_len"])
            al = float(r["alpha"])
            mse = float(r["MSE"])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isnan(mse) or math.isinf(mse):
            continue
        raw[(ds, pl, al)].append(mse)

    agg: dict[tuple[str, int, float], dict[str, float]] = {}
    for key, vals in raw.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        agg[key] = {"mean": mean, "std": math.sqrt(var), "n": n}
    return agg


def _summary_csv(agg: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure3_summary.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "pred_len", "alpha", "mean_MSE", "std_MSE", "n_seeds"])
        for key in sorted(agg):
            ds, pl, al = key
            writer.writerow([ds, pl, al, f"{agg[key]['mean']:.8g}",
                             f"{agg[key]['std']:.8g}", agg[key]["n"]])
    return out_path


def _plot(agg: dict, out_dir: Path) -> list[Path]:
    datasets = sorted({k[0] for k in agg})
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer matplotlib; fall back gracefully if unavailable.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_figure3] matplotlib not available; producing text-only summary.",
              file=sys.stderr)
        return []

    produced: list[Path] = []
    for ds in datasets:
        # Collect relevant keys for this dataset.
        preds = sorted({k[1] for k in agg if k[0] == ds})
        ncols = min(2, len(preds))
        nrows = (len(preds) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 1, 3 * nrows + 1),
                                 sharey=False)
        if not isinstance(axes, list):
            axes = [axes]
        axes_flat = [a for row in axes for a in row] if nrows > 1 else axes
        fig.suptitle(f"Figure 3 — alpha sensitivity ({ds})", fontsize=12)

        for ax, pred_len in zip(axes_flat, preds):
            alphas = sorted({k[2] for k in agg if k[0] == ds and k[1] == pred_len})
            means = [agg[(ds, pred_len, a)]["mean"] for a in alphas]
            stds = [agg[(ds, pred_len, a)]["std"] for a in alphas]
            ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=4, color="k",
                        linestyle="--", alpha=0.6)
            ax.plot(alphas, means, marker="s", color="#1f77b4", linewidth=2)
            ax.set_xlabel("alpha")
            ax.set_ylabel("MSE (mean over seeds)")
            ax.set_title(f"pred_len = {pred_len}")
            ax.grid(True, which="both", ls="--", alpha=0.3)
            for x, y, s in zip(alphas, means, [agg[(ds, pred_len, a)]["n"] for a in alphas]):
                ax.annotate(f"n={s}", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8)
        for ax in axes_flat[len(preds):]:
            ax.set_visible(False)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = out_dir / f"figure3_{ds}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        produced.append(out_path)
    return produced


def _print_summary(agg: dict) -> None:
    datasets = sorted({k[0] for k in agg})
    print(f"{'dataset':>10} {'pred_len':>9} {'alpha':>6} {'n':>3} {'mean_MSE':>12} {'std_MSE':>12}")
    for ds in datasets:
        preds = sorted({k[1] for k in agg if k[0] == ds})
        for pl in preds:
            alphas = sorted({k[2] for k in agg if k[0] == ds and k[1] == pl})
            for al in alphas:
                stats = agg[(ds, pl, al)]
                print(f"{ds:>10} {pl:>9d} {al:>6.2f} {stats['n']:>3d} "
                      f"{stats['mean']:>12.6f} {stats['std']:>12.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_CSV),
                        help=f"Path to figure3_runs.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help=f"Directory for plots (default: {DEFAULT_OUT})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots; print text summary only")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[plot_figure3] ERROR: {csv_path} does not exist", file=sys.stderr)
        return 2
    rows = _load_rows(csv_path)
    agg = _aggregate(rows)
    if not agg:
        print("[plot_figure3] No dishts rows found in the CSV — "
              "run Phase 2a first.", file=sys.stderr)
        return 1

    print("=" * 80)
    print("Figure 3 — alpha-sensitivity summary")
    print("=" * 80)
    _print_summary(agg)

    out_dir = Path(args.output)
    summary_path = _summary_csv(agg, out_dir)
    print(f"\nWrote summary CSV: {summary_path}")

    if not args.no_plot:
        paths = _plot(agg, out_dir)
        if paths:
            print("Wrote PNG:")
            for p in paths:
                print(f"  {p}")
        else:
            print("(matplotlib not available — no PNG produced.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
