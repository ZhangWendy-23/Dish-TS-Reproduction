"""Generate a table comparing my experimental results against the paper.

Reads `results/figure3_runs.csv` (written by train.py after every run)
and compares with the paper's reported values from `paper_results/table*.csv`.

Produces `results/paper_vs_mine.csv` and a console summary with direction-consistency flags.

Usage
-----
    python3 repro_figures/compare_paper.py                          # uses figure3_runs.csv
    python3 repro_figures/compare_paper.py --input results/paper_summary.csv
    python3 repro_figures/compare_paper.py --alpha 0.5              # filter to specific alpha
"""
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAME_MAP = {"ECL": "Electricity", "WTH": "Weather", "ETTh1": "ETTh1", "ETTm2": "ETTm2"}

# Columns in figure3_runs.csv vs paper_summary.csv — both are supported.
F3_COLS  = ["dataset", "seq_len", "pred_len", "model", "norm", "alpha",
            "MSE", "MAE", "RMSE", "MAPE", "MSPE", "timestamp"]
PS_COLS  = ["data", "model", "norm", "seed", "seq_len", "label_len",
            "pred_len", "batch_size", "alpha", "MSE", "MAE", "RMSE",
            "MAPE", "MSPE", "log"]


def _deduce_format(path: str) -> str:
    """Return 'f3' (figure3_runs) or 'ps' (paper_summary) based on column names."""
    with open(path) as f:
        header = f.readline().strip()
    if "dataset" in header:
        return "f3"
    return "ps"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(ROOT, "results", "figure3_runs.csv"),
                    help="Path to results CSV (default: results/figure3_runs.csv)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Filter to a specific alpha value (default: use all rows).")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run experiments first.")
        sys.exit(1)

    fmt = _deduce_format(args.input)
    mine = pd.read_csv(args.input)

    # Normalize column names to a common set
    if fmt == "ps":
        mine = mine.rename(columns={"data": "dataset"})

    # If user specified --alpha (e.g. 0.5 for the main Table 2/3 runs)
    if args.alpha is not None and "alpha" in mine.columns:
        mine = mine[mine["alpha"] == args.alpha].copy()
        print(f"[filtered] alpha = {args.alpha}  →  {len(mine)} rows")
    else:
        print(f"[input]   {os.path.basename(args.input)}  →  {len(mine)} rows")

    # Group by (dataset, model, norm, pred_len) and average across seeds/alphas
    if "seed" in mine.columns and mine["seed"].nunique() > 1:
        print(f"[seeds]   {mine['seed'].nunique()} unique seeds detected → averaging across seeds")

    mean = mine.groupby(["dataset", "model", "norm", "pred_len"], as_index=False)["MSE"].mean()

    # Load paper reference tables
    t2 = pd.read_csv(os.path.join(ROOT, "paper_results", "table2_multivariate.csv"))
    t3 = pd.read_csv(os.path.join(ROOT, "paper_results", "table3_revin_comparison.csv"))

    # Pivot: one row per (dataset, model, pred_len), columns = norm
    mean_pivot = mean.pivot_table(index=["dataset", "model", "pred_len"],
                                  columns="norm", values="MSE").reset_index()
    mean_pivot = mean_pivot[mean_pivot["model"] == "Autoformer"].reset_index(drop=True)

    compare_rows = []
    for _, row in mean_pivot.iterrows():
        data = row["dataset"]
        pl = int(row["pred_len"])
        paper_name = NAME_MAP.get(data, data)

        paper_row2 = t2[(t2["dataset"] == paper_name) & (t2["horizon"] == pl)]
        paper_row3 = t3[(t3["dataset"] == paper_name) & (t3["horizon"] == pl)]

        r = {
            "dataset": data,
            "pred_len": pl,
            "my_none":   _val(row, "none"),
            "my_revin":  _val(row, "revin"),
            "my_dishts": _val(row, "dishts"),
            "paper_autoformer":        _val2(paper_row2, "autoformer_mse"),
            "paper_autoformer_dishts": _val2(paper_row2, "autoformer_dishts_mse"),
            "paper_revin":  _val2(paper_row3, "revin_mse"),
            "paper_dishts": _val2(paper_row3, "dishts_mse"),
        }
        if "none" in row and not np.isnan(row["none"]) and \
           r["paper_autoformer"] is not None:
            r["ratio_none"] = float(row["none"]) / r["paper_autoformer"]
        if r["my_revin"] and r["paper_revin"]:
            r["ratio_revin"] = r["my_revin"] / r["paper_revin"]
        if r["my_dishts"] and r["paper_dishts"]:
            r["ratio_dishts"] = r["my_dishts"] / r["paper_dishts"]
        if r["my_dishts"] and r["my_revin"] and r["paper_dishts"] and r["paper_revin"]:
            my_winner    = "dishts" if r["my_dishts"] < r["my_revin"] else "revin"
            paper_winner = "dishts" if r["paper_dishts"] < r["paper_revin"] else "revin"
            r["consistent_dir_revin_vs_dishts"] = (my_winner == paper_winner)
            r["my_gap_pct"]    = (r["my_dishts"] - r["my_revin"]) / r["my_revin"] * 100
            r["paper_gap_pct"] = (r["paper_dishts"] - r["paper_revin"]) / r["paper_revin"] * 100
        compare_rows.append(r)

    comp = pd.DataFrame(compare_rows)
    out_path = os.path.join(ROOT, "results", "paper_vs_mine.csv")
    comp.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")

    # --- pretty print ---
    cols_show = ["dataset", "pred_len", "my_revin", "my_dishts",
                 "paper_revin", "paper_dishts", "consistent_dir_revin_vs_dishts"]
    print()
    print(comp[cols_show].round(4).to_string(index=False))

    # --- direction summary ---
    print("\nDish-TS vs RevIN direction consistency (per dataset):")
    for dset in comp["dataset"].unique():
        sub = comp[comp["dataset"] == dset].dropna(subset=["consistent_dir_revin_vs_dishts"])
        if sub.empty:
            continue
        n = len(sub)
        ok = int(sub["consistent_dir_revin_vs_dishts"].sum())
        status = "✓ PASS" if ok == n else f"❌ {n-ok} MISMATCH" + ("(ES)" if n-ok > 0 else "")
        print(f"  {dset:8s}: {ok}/{n} cells match paper direction  {status}")

    # --- scale summary ---
    print("\nScale factor (my MSE / paper MSE):")
    ratio_cols = [c for c in ["ratio_none", "ratio_revin", "ratio_dishts"] if c in comp.columns]
    scale = comp[["dataset", "pred_len"] + ratio_cols].dropna(how="all", subset=ratio_cols)
    if not scale.empty:
        print(scale.round(2).to_string(index=False))
    else:
        print("  (no overlap with paper reference — datasets or horizons may differ)")


def _val(row: pd.Series, col: str):
    if col in row.index and not np.isnan(row[col]):
        return float(row[col])
    return None


def _val2(df: pd.DataFrame, col: str):
    if len(df) and col in df.columns:
        return float(df[col].values[0])
    return None


if __name__ == "__main__":
    main()
