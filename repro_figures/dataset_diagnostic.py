"""Diagnose a dataset CSV for scale, missing values, and time granularity.

Usage
-----
    python3 repro_figures/dataset_diagnostic.py --input data/ETTm2/ETTm2.csv

What it prints
--------------
* Number of rows, number of feature columns (excluding ``date``).
* Per-column mean / std / min / max / q1 / q50 / q99  — to spot features
  whose order-of-magnitude is anomalous compared with the rest.
* Number of NaNs / zeros / negative values per column.
* The minimum and median time-step inferred from the ``date`` column
  (useful to tell 15-minute from hourly data).
* A crude "anomaly score": columns whose std is more than
  ``threshold * median_std`` across columns are flagged, because they
  usually dominate MSE and make the aggregate look different from the paper.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the dataset CSV.")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="Flag a column if its std > threshold * (median std across columns).")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=[0])
    date_col = df.columns[0]
    features = list(df.columns[1:])
    print(f"Input : {args.input}")
    print(f"Rows  : {len(df):,}")
    print(f"Date column: {date_col}")
    print(f"Features ({len(features)}): {features}")
    print()

    # --- Time granularity ---
    try:
        dates = pd.to_datetime(df.iloc[:, 0])
        deltas = dates.diff().dropna()
        if len(deltas) > 0:
            print("Time-step (inferred from date column):")
            print(f"  min    = {deltas.min()}")
            print(f"  median = {deltas.median()}")
            print(f"  max    = {deltas.max()}")
        else:
            print("Time-step: not enough rows to infer.")
    except Exception as exc:  # noqa: BLE001
        print(f"Time-step: could not parse date column ({exc}).")
    print()

    # --- Per-column numeric summary ---
    stats = []
    for col in features:
        s = pd.to_numeric(df[col], errors="coerce")
        stats.append({
            "column": col,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q01": float(s.quantile(0.01)),
            "q50": float(s.quantile(0.50)),
            "q99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "n_nan": int(s.isna().sum()),
            "n_zero": int((s == 0).sum()),
            "n_neg": int((s < 0).sum()),
        })
    summary = pd.DataFrame(stats)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:>10.3f}"))
    print()

    # --- Flag anomalous columns ---
    stds = summary["std"].dropna()
    if len(stds) > 1:
        median_std = float(np.median(stds.values))
        flagged = summary[summary["std"] > args.threshold * median_std]
        print(f"Median std across columns = {median_std:.4f}; threshold = "
              f"{args.threshold * median_std:.4f}")
        if flagged.empty:
            print("No column flagged as outlier.")
        else:
            print("*** FLAGGED columns (std > threshold * median_std) — these may dominate MSE:")
            print(flagged[["column", "mean", "std", "min", "q50", "max"]]
                  .to_string(index=False, float_format=lambda v: f"{v:>10.3f}"))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
