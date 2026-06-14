"""Repair figure3_runs.csv: normalize all rows to the 16-column schema.

The current figure3_runs.csv contains rows from two train.py versions:

- OLD (13 columns): dataset, seq_len, pred_len, model, norm, alpha, seed,
  MSE, MAE, RMSE, MAPE, MSPE, timestamp
- NEW (16 columns): dataset, seq_len, pred_len, model, norm, alpha, seed,
  MSE, MAE, RMSE, MAPE, MSPE, dish_init, scaled, prior, timestamp

This script rewrites the file so every row uses the 16-column schema.
For the old 13-column rows we fill the missing three columns with safe
defaults:
    dish_init = "avg"   (paper default)
    scaled    = True    (old train.py defaulted to StandardScaler)
    prior     = "paper" (old train.py used the paper-style prior loss)

A backup of the original file is written to figure3_runs.csv.bak.<TIMESTAMP>.
"""

from __future__ import annotations

import datetime
import os
import shutil

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "results", "figure3_runs.csv")

NEW_COLS = [
    "dataset", "seq_len", "pred_len", "model", "norm", "alpha", "seed",
    "MSE", "MAE", "RMSE", "MAPE", "MSPE",
    "dish_init", "scaled", "prior", "timestamp",
]


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found; nothing to repair.")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CSV_PATH}.bak.{ts}"
    shutil.copy2(CSV_PATH, backup_path)
    print(f"[repair] backed up original -> {backup_path}")

    with open(CSV_PATH, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    if not lines:
        raise SystemExit("empty CSV")

    header_fields = [c.strip() for c in lines[0].split(",")]
    n_header = len(header_fields)
    print(f"[repair] original header: {n_header} fields -> {header_fields}")

    old_rows: list[list[str]] = []
    new_rows: list[list[str]] = []
    bad_rows = 0

    for line in lines[1:]:
        fields = [c.strip() for c in line.split(",")]
        if len(fields) == 13:
            old_rows.append(fields)
        elif len(fields) == 16:
            new_rows.append(fields)
        else:
            bad_rows += 1

    print(f"[repair] {len(old_rows)} old rows (13 fields); "
          f"{len(new_rows)} new rows (16 fields); {bad_rows} bad rows dropped")

    rebuilt_rows = []
    # Old 13-field rows: insert dish_init, scaled, prior BEFORE timestamp
    for row in old_rows:
        rebuilt_rows.append(row[:12] + ["avg", "True", "paper"] + row[12:13])
    for row in new_rows:
        rebuilt_rows.append(row[:16])

    rebuilt = pd.DataFrame(rebuilt_rows, columns=NEW_COLS)

    for col in ("seq_len", "pred_len", "seed"):
        rebuilt[col] = pd.to_numeric(
            rebuilt[col], errors="coerce"
        ).astype("Int64")
    for col in ("alpha", "MSE", "MAE", "RMSE", "MAPE", "MSPE"):
        rebuilt[col] = pd.to_numeric(rebuilt[col], errors="coerce")

    rebuilt = rebuilt.sort_values(
        ["dataset", "pred_len", "norm", "prior", "alpha", "seed"]
    ).reset_index(drop=True)

    rebuilt.to_csv(CSV_PATH, index=False)
    print(f"[repair] wrote {len(rebuilt)} rows to {CSV_PATH}")

    print()
    print("========== SUMMARY ==========")
    for (ds, pl, norm, prior), g in rebuilt.groupby(
            ["dataset", "pred_len", "norm", "prior"], dropna=False):
        alphas = sorted(g["alpha"].dropna().unique().tolist())
        seeds = sorted(g["seed"].dropna().unique().tolist())
        print(f"- {ds}  pl={pl}  norm={norm}  prior={prior}: "
              f"seeds={seeds}  alphas={alphas}  mean_MSE={g['MSE'].mean():.4f}")


if __name__ == "__main__":
    main()
