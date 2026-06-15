#!/usr/bin/env python3
"""Phase 2a quality-check script.

Reads ``results/figure3_runs.csv`` and prints:
  * how many ETTm2 dishts rows were logged (expected 45)
  * any NaN / out-of-scale MSE rows
  * a (pred_len x alpha) mean +/- std MSE table (raw scale)
  * the best alpha per pred_len, with % diff vs alpha=0.0 baseline

Run it *after* ``repro_figures/run_phase2a.sh``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/figure3_runs.csv",
                    help="CSV produced by train.py --figure3 (default: %(default)s)")
    ap.add_argument("--dataset", default="ETTm2",
                    help="Dataset filter (default: %(default)s)")
    ap.add_argument("--norm", default="dishts",
                    help="Norm filter (default: %(default)s)")
    ap.add_argument("--mse-max", type=float, default=1e6,
                    help="Rows with MSE above this are flagged (default: %(default)s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2023, 2024, 2025],
                    help="Seeds to require (default: 2023 2024 2025)")
    ap.add_argument("--pred-lens", type=int, nargs="+", default=[96, 168, 336],
                    help="Prediction lengths to require (default: 96 168 336)")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="Alpha values to require (default: 0.0 0.25 0.5 0.75 1.0)")
    args = ap.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} does not exist. Run run_phase2a.sh first.",
              file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))
    # filter to Phase 2a scope
    def _row_match(r: dict) -> bool:
        if r.get("dataset") != args.dataset:
            return False
        if r.get("norm") != args.norm:
            return False
        try:
            pl = int(r["pred_len"])
            al = float(r["alpha"])
            seed = int(r["seed"])
        except (KeyError, ValueError):
            return False
        return (pl in args.pred_lens and al in args.alphas and seed in args.seeds)

    filtered = [r for r in rows if _row_match(r)]
    expected = len(args.pred_lens) * len(args.alphas) * len(args.seeds)

    print("=" * 80)
    print(f"Phase 2a quality check — dataset={args.dataset} norm={args.norm}")
    print(f"CSV={csv_path} total_rows={len(rows)}")
    print(f"Expected rows in scope = {expected}, found = {len(filtered)}")
    if len(filtered) == expected:
        print("  -> OK: every (pred_len, alpha, seed) cell present.")
    else:
        missing: list[tuple[int, float, int]] = []
        seen = {(int(r["pred_len"]), float(r["alpha"]), int(r["seed"])) for r in filtered}
        for pl in args.pred_lens:
            for al in args.alphas:
                for seed in args.seeds:
                    if (pl, al, seed) not in seen:
                        missing.append((pl, al, seed))
        print(f"  -> WARN: {len(missing)} cells missing:")
        for pl, al, seed in missing[:20]:
            print(f"       pred_len={pl} alpha={al} seed={seed}")
        if len(missing) > 20:
            print(f"       ... and {len(missing) - 20} more")

    # ---- bad MSE detection ----
    bad: list[tuple[int, float, int, str]] = []
    for r in filtered:
        mse_str = r.get("MSE", "")
        try:
            mse = float(mse_str)
        except ValueError:
            bad.append((int(r["pred_len"]), float(r["alpha"]),
                        int(r["seed"]), mse_str))
            continue
        if math.isnan(mse) or mse > args.mse_max:
            bad.append((int(r["pred_len"]), float(r["alpha"]),
                        int(r["seed"]), mse_str))
    print()
    print(f"Bad MSE rows (NaN or > {args.mse_max:.0e}): {len(bad)}")
    for pl, al, seed, mse in bad[:10]:
        print(f"  pred_len={pl} alpha={al} seed={seed} MSE={mse}")
    if len(bad) > 10:
        print(f"  ... and {len(bad) - 10} more")

    # ---- aggregate MSE per (pred_len, alpha) ----
    agg: dict[tuple[int, float], list[float]] = defaultdict(list)
    for r in filtered:
        try:
            mse = float(r["MSE"])
        except (KeyError, ValueError):
            continue
        key = (int(r["pred_len"]), float(r["alpha"]))
        agg[key].append(mse)

    print()
    print("=" * 80)
    print("mean +/- std MSE (raw scale)")
    print("=" * 80)
    header = f"{'pred_len':>9s} {'alpha':>6s} {'n':>4s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * 80)
    for pl in sorted({k[0] for k in agg}):
        for al in sorted(k[1] for k in agg if k[0] == pl):
            arr = agg[(pl, al)]
            m = sum(arr) / len(arr)
            v = sum((x - m) ** 2 for x in arr) / len(arr)
            s = math.sqrt(v)
            print(f"{pl:>9d} {al:>6.2f} {len(arr):>4d} "
                  f"{m:>10.3f} {s:>10.3f} "
                  f"{min(arr):>10.3f} {max(arr):>10.3f}")
        print()

    # ---- best alpha per pred_len ----
    print("Best alpha per pred_len (vs alpha=0.0 baseline):")
    any_better = False
    for pl in sorted({k[0] for k in agg}):
        arr_pl = [(al, agg[(pl, al)]) for al in sorted(k[1] for k in agg if k[0] == pl)]
        means = [(al, sum(a) / len(a)) for al, a in arr_pl]
        best = min(means, key=lambda x: x[1])
        base_rows = [x for x in means if abs(x[0] - 0.0) < 1e-9]
        if not base_rows:
            print(f"  pred_len={pl}: best_alpha={best[0]:.2f} MSE={best[1]:.3f} "
                  "(no alpha=0.0 baseline found)")
            continue
        base = base_rows[0][1]
        pct = 100.0 * (best[1] - base) / base
        marker = "  (baseline = alpha=0.0)" if abs(best[0]) < 1e-9 else "  (alpha>0 better than baseline!)"
        if abs(best[0]) >= 1e-9:
            any_better = True
        print(f"  pred_len={pl}: best_alpha={best[0]:.2f} MSE={best[1]:.3f} "
              f"(alpha=0.0 MSE={base:.3f}, diff={pct:+.2f}%){marker}")

    print()
    if any_better:
        print("DECISION: alpha>0 beats alpha=0.0 on at least one horizon. "
              "Proceed to Phase 2b (multi-dataset sweep).")
    else:
        print("DECISION: alpha=0.0 remains best. Consider whether to extend "
              "Phase 2b anyway to confirm this holds across datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
