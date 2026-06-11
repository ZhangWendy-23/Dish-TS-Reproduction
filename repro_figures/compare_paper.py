"""Generate a table comparing my experimental results against the paper.

Produces a human-readable comparison saved to results/paper_vs_mine.csv
and prints a summary with consistency flags.
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAME_MAP = {"ECL": "Electricity", "WTH": "Weather", "ETTh1": "ETTh1", "ETTm2": "ETTm2"}


def load_my_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    mine = load_my_results(os.path.join(ROOT, "results", "paper_summary.csv"))
    mean = mine.groupby(["data", "model", "norm", "pred_len"], as_index=False)["MSE"].mean()
    t2 = pd.read_csv(os.path.join(ROOT, "paper_results", "table2_multivariate.csv"))
    t3 = pd.read_csv(os.path.join(ROOT, "paper_results", "table3_revin_comparison.csv"))

    # Pivot mine from long to wide: one row per (data, pred_len), columns by norm.
    mean_pivot = mean.pivot_table(index=["data", "model", "pred_len"], columns="norm",
                                   values="MSE").reset_index()
    # Only keep rows where model == Autoformer (the rest is covered by Table 2 in paper)
    mean_pivot = mean_pivot[mean_pivot["model"] == "Autoformer"].reset_index(drop=True)

    # Build wide comparison rows
    compare_rows = []
    for _, row in mean_pivot.iterrows():
        data = row["data"]
        pl = int(row["pred_len"])
        paper_name = NAME_MAP.get(data, data)
        # Paper Table 2: Autoformer vs Autoformer+Dish-TS
        paper_row2 = t2[(t2["dataset"] == paper_name) & (t2["horizon"] == pl)]
        # Paper Table 3: RevIN vs Dish-TS
        paper_row3 = t3[(t3["dataset"] == paper_name) & (t3["horizon"] == pl)]

        r = {
            "dataset": data,
            "pred_len": pl,
            "my_none": float(row["none"]) if "none" in row and not np.isnan(row["none"]) else None,
            "my_revin": float(row["revin"]) if "revin" in row and not np.isnan(row["revin"]) else None,
            "my_dishts": float(row["dishts"]) if "dishts" in row and not np.isnan(row["dishts"]) else None,
            "paper_autoformer": float(paper_row2["autoformer_mse"].values[0]) if len(paper_row2) else None,
            "paper_autoformer_dishts": float(paper_row2["autoformer_dishts_mse"].values[0]) if len(paper_row2) else None,
            "paper_revin": float(paper_row3["revin_mse"].values[0]) if len(paper_row3) else None,
            "paper_dishts": float(paper_row3["dishts_mse"].values[0]) if len(paper_row3) else None,
        }
        # Scale factors (my / paper)
        if r["my_revin"] and r["paper_revin"]:
            r["ratio_revin"] = r["my_revin"] / r["paper_revin"]
        if r["my_dishts"] and r["paper_dishts"]:
            r["ratio_dishts"] = r["my_dishts"] / r["paper_dishts"]
        # Direction consistency: Dish-TS vs RevIN
        if r["my_dishts"] and r["my_revin"] and r["paper_dishts"] and r["paper_revin"]:
            my_winner = "dishts" if r["my_dishts"] < r["my_revin"] else "revin"
            paper_winner = "dishts" if r["paper_dishts"] < r["paper_revin"] else "revin"
            r["consistent_dir_revin_vs_dishts"] = (my_winner == paper_winner)
            r["my_gap_pct"] = (r["my_dishts"] - r["my_revin"]) / r["my_revin"] * 100
            r["paper_gap_pct"] = (r["paper_dishts"] - r["paper_revin"]) / r["paper_revin"] * 100
        compare_rows.append(r)

    comp = pd.DataFrame(compare_rows)
    out_path = os.path.join(ROOT, "results", "paper_vs_mine.csv")
    comp.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print()
    # Pretty print
    cols_show = ["dataset", "pred_len", "my_revin", "my_dishts",
                 "paper_revin", "paper_dishts",
                 "consistent_dir_revin_vs_dishts"]
    print(comp[cols_show].round(4).to_string(index=False))
    print()
    # Summary of direction consistency
    for data in comp["dataset"].unique():
        sub = comp[comp["dataset"] == data].dropna(subset=["consistent_dir_revin_vs_dishts"])
        if sub.empty:
            continue
        n = len(sub)
        ok = int(sub["consistent_dir_revin_vs_dishts"].sum())
        print(f"  {data:8s}: {ok}/{n} cells have the same Dish-TS vs RevIN direction as the paper.")

    # Scale analysis
    print("\nScale factor analysis (\"my / paper\"):")
    cols = comp[["dataset", "pred_len", "ratio_revin", "ratio_dishts"]].dropna()
    print(cols.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
