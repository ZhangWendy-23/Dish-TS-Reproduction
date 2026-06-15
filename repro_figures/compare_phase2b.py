#!/usr/bin/env python3
"""Print a (dataset x pred_len x alpha) summary table of MSE (raw scale).

Useful after ``repro_figures/run_phase2b.sh``. Reads
``results/figure3_runs.csv`` and prints mean/std per
(dataset, pred_len, alpha) for norm='dishts'.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/figure3_runs.csv")
    ap.add_argument("--norm", default="dishts")
    ap.add_argument("--datasets", nargs="+",
                    default=["ECL", "ETTh1", "ETTm2", "WTH"])
    ap.add_argument("--pred-lens", type=int, nargs="+",
                    default=[96, 168, 336])
    args = ap.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} does not exist.", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(csv_path.open()))

    def _keep(r: dict) -> bool:
        if r.get("norm") != args.norm:
            return False
        if r.get("dataset") not in args.datasets:
            return False
        try:
            return int(r["pred_len"]) in args.pred_lens
        except (KeyError, ValueError):
            return False

    rows = [r for r in rows if _keep(r)]
    print(f"Loaded {len(rows)} rows from {csv_path} "
          f"(norm={args.norm}, datasets={args.datasets}).")

    # (dataset, pred_len, alpha) -> list of MSE
    agg: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for r in rows:
        try:
            mse = float(r["MSE"])
        except (KeyError, ValueError):
            continue
        key = (r["dataset"], int(r["pred_len"]), float(r["alpha"]))
        agg[key].append(mse)

    for dataset in args.datasets:
        print()
        print("=" * 80)
        print(f"dataset={dataset}  mean+/-std MSE (raw scale)")
        print("=" * 80)
        header = (f"{'pred_len':>9s} {'alpha':>6s} {'n':>4s} "
                  f"{'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}")
        print(header)
        print("-" * 80)
        for pl in args.pred_lens:
            rows_for_pl = [(al, agg[(dataset, pl, al)])
                            for al in sorted(k[2] for k in agg
                                              if k[0] == dataset and k[1] == pl)]
            if not rows_for_pl:
                print(f"{pl:>9d} (no data)")
                continue
            for al, arr in rows_for_pl:
                m = sum(arr) / len(arr)
                v = sum((x - m) ** 2 for x in arr) / len(arr)
                s = math.sqrt(v)
                print(f"{pl:>9d} {al:>6.2f} {len(arr):>4d} "
                      f"{m:>10.3f} {s:>10.3f} "
                      f"{min(arr):>10.3f} {max(arr):>10.3f}")

            # best alpha
            means = [(al, sum(arr) / len(arr)) for al, arr in rows_for_pl]
            best = min(means, key=lambda x: x[1])
            baseline = [x for x in means if abs(x[0] - 0.0) < 1e-9]
            if baseline:
                pct = 100.0 * (best[1] - baseline[0][1]) / baseline[0][1]
                print(f"           best_alpha={best[0]:.2f} MSE={best[1]:.3f} "
                      f"(alpha=0.0 MSE={baseline[0][1]:.3f}, "
                      f"diff={pct:+.2f}%)")
            else:
                print(f"           best_alpha={best[0]:.2f} MSE={best[1]:.3f} "
                      "(no alpha=0.0 baseline)")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
