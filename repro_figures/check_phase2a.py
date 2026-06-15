"""Phase 2a / Figure 3 — Quality check + best-alpha analysis.

Usage
-----
    python3 repro_figures/check_phase2a.py

Reads ``results/figure3_runs.csv``, filters the rows that belong to the
ETTm2 dishts alpha-sweep experiment (i.e. Phase 2a), and prints

1. a per-config job / NaN / out-of-range report,
2. a (pred_len x alpha) table with mean, std, min MSE across seeds,
3. a "best alpha per prediction horizon" summary, and
4. a plain-text recommendation on how to proceed.

The script is intentionally dependency-light (stdlib only) so that it
can be run on any machine with Python 3.8+.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "figure3_runs.csv"

TARGET_DATASET = "ETTm2"
TARGET_NORM = "dishts"
TARGET_ALPHAS = {0.0, 0.5, 1.0}
TARGET_PREDS = {24, 96, 168, 336}

# MSE above this value is flagged as "unreasonably large" for the
# ETTm2 dataset at the default raw scale.
MSE_ABSURD = 1.0e4


def main() -> int:
    if not CSV.exists():
        print(f"[check_phase2a] ERROR: {CSV} does not exist.")
        print("  Run Phase 2a first. Example:")
        print("    bash repro_figures/run_phase2a.sh")
        return 2

    with open(CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # ---- filter rows that look like Phase 2a ----
    p2a_rows = []
    for r in rows:
        if r.get("dataset") != TARGET_DATASET or r.get("norm") != TARGET_NORM:
            continue
        try:
            alpha = float(r.get("alpha"))
            pred = int(r.get("pred_len"))
        except (TypeError, ValueError):
            continue
        if alpha not in TARGET_ALPHAS or pred not in TARGET_PREDS:
            continue
        p2a_rows.append(r)

    total_expected = len(TARGET_PREDS) * len(TARGET_ALPHAS) * 3  # 3 seeds
    print("=" * 80)
    print("Phase 2a — ETTm2 dishts alpha-sweep quality check")
    print("=" * 80)
    print(f"CSV file         : {CSV}")
    print(f"Rows in file     : {len(rows)}")
    print(f"Phase-2a rows    : {len(p2a_rows)} (expected <= {total_expected})")
    print(f"Target horizons  : {sorted(TARGET_PREDS)}")
    print(f"Target alphas    : {sorted(TARGET_ALPHAS)}")
    print()

    # ---- aggregate by (pred_len, alpha) ----
    buckets: dict[tuple[int, float], list[tuple[float, int]]] = defaultdict(list)
    bad_rows: list[str] = []
    for r in p2a_rows:
        key = (int(r["pred_len"]), float(r["alpha"]))
        try:
            mse = float(r["MSE"])
        except (KeyError, ValueError):
            bad_rows.append(f"  non-numeric MSE: {r}")
            continue
        if math.isnan(mse) or math.isinf(mse):
            bad_rows.append(f"  NaN/Inf MSE for {key} seed={r.get('seed')}")
            continue
        seed = int(r.get("seed") or -1)
        buckets[key].append((mse, seed))

    print("1. NaN / out-of-range check")
    print("-" * 80)
    if bad_rows:
        print(f"  BAD rows: {len(bad_rows)}")
        for b in bad_rows[:10]:
            print(b)
        if len(bad_rows) > 10:
            print(f"  ... ({len(bad_rows) - 10} more; see CSV)")
    else:
        print("  ✅  No NaN / Inf / non-numeric MSE rows found.")
    print()

    # ---- per-config job count + mean table ----
    print("2. Per-config job count")
    print("-" * 80)
    header = (f"{'pred_len':>8} {'alpha':>7} {'jobs':>5} "
              f"{'mean_MSE':>12} {'std':>10} {'min':>12}")
    print(header)
    print("-" * len(header))
    for pred_len in sorted(TARGET_PREDS):
        for alpha in sorted(TARGET_ALPHAS):
            key = (pred_len, alpha)
            arr = buckets.get(key, [])
            if not arr:
                print(f"{pred_len:>8d} {alpha:>7.2f} {0:>5d} "
                      f"{'—':>12} {'—':>10} {'—':>12}  ⚠️ missing")
                continue
            mses = [x[0] for x in arr]
            n = len(mses)
            mean = sum(mses) / n
            var = sum((m - mean) ** 2 for m in mses) / n
            std = math.sqrt(var)
            mn = min(mses)
            flag = ""
            if n < 3:
                flag = "  ⚠️ fewer than 3 seeds"
            if mean > MSE_ABSURD:
                flag += "  ❗ very large MSE"
            print(f"{pred_len:>8d} {alpha:>7.2f} {n:>5d} "
                  f"{mean:>12.6f} {std:>10.6f} {mn:>12.6f}{flag}")
    print()

    # ---- best alpha per horizon ----
    print("3. Best alpha per prediction horizon")
    print("-" * 80)
    header2 = f"{'pred_len':>10} {'best_alpha':>11} {'best_mean_MSE':>14} {'alpha_0_mean':>14} {'delta':>9} {'delta%':>9}"
    print(header2)
    print("-" * len(header2))
    recommendations: list[str] = []
    for pred_len in sorted(TARGET_PREDS):
        alpha_means: dict[float, float] = {}
        for alpha in sorted(TARGET_ALPHAS):
            arr = buckets.get((pred_len, alpha), [])
            if arr:
                alpha_means[alpha] = sum(x[0] for x in arr) / len(arr)
        if not alpha_means:
            continue
        best_alpha = min(alpha_means, key=alpha_means.get)
        best_mean = alpha_means[best_alpha]
        base_mean = alpha_means.get(0.0)
        if base_mean is None:
            continue
        delta = best_mean - base_mean
        delta_pct = 100.0 * delta / base_mean if base_mean > 0 else float("nan")
        print(f"{pred_len:>10d} {best_alpha:>11.2f} {best_mean:>14.6f} {base_mean:>14.6f} "
              f"{delta:>+9.6f} {delta_pct:>+8.2f}%")
        if pred_len <= 96 and best_alpha == 0.0:
            pass  # consistent with short-horizon behaviour
        elif pred_len >= 168 and best_alpha == 1.0:
            recommendations.append(f"pred_len={pred_len}: alpha=1.0 is best "
                                   f"({delta_pct:+.2f}% vs alpha=0.0) — "
                                   f"论文 Figure-3 的长窗口趋势已复现。")
        elif pred_len >= 168 and best_alpha != 0.0:
            recommendations.append(f"pred_len={pred_len}: best alpha={best_alpha} "
                                   f"({delta_pct:+.2f}% vs alpha=0.0) — "
                                   f"方向正确，但需更多 seeds 来确认差异是否显著。")
        elif pred_len >= 168 and best_alpha == 0.0:
            recommendations.append(f"pred_len={pred_len}: alpha=0.0 仍是最优 — "
                                   f"该数据上 prior term 未改善 MSE。")

    print()
    print("4. Summary / recommendation")
    print("-" * 80)
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  (no data — cannot produce a recommendation)")
    print()
    print("  Next-step suggestions:")
    print("    ① If every cell has 3 seeds and no NaN — proceed to Phase 2b")
    print("      (bash repro_figures/run_phase2b.sh)")
    print("    ② If some cell has fewer than 3 seeds — re-run missing seeds")
    print("      with the same bash command; train.py appends to the CSV.")
    print("    ③ For visual Figure-3 output — run:")
    print("      python3 repro_figures/plot_figure3.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
