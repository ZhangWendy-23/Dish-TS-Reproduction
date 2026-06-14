"""Print alpha-sensitivity curves from the repaired figure3_runs.csv."""

from __future__ import annotations

import pandas as pd

CSV_PATH = "results/figure3_runs.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")
    print()

    # For each (dataset, pred_len, norm, prior), print: alpha -> MSE
    groups = df.groupby(["dataset", "pred_len", "norm", "prior"], dropna=False)
    for (ds, pl, norm, prior), g in sorted(groups.groups.items()):
        sub = df.loc[g]
        alphas = sorted(sub["alpha"].dropna().unique().tolist())
        if len(alphas) < 2:
            # not an alpha sweep; skip direction printing
            continue
        print(f"===== {ds}  pred_len={pl}  norm={norm}  prior={prior} =====")
        by_alpha = sub.groupby("alpha").agg(
            n=("seed", "count"),
            mean=("MSE", "mean"),
            std=("MSE", "std"),
        ).reset_index()
        by_alpha = by_alpha.sort_values("alpha").reset_index(drop=True)
        print(by_alpha.round(4).to_string(index=False))
        first_mse = by_alpha["mean"].iloc[0]
        last_mse = by_alpha["mean"].iloc[-1]
        direction = "UP" if last_mse > first_mse else ("DOWN" if last_mse < first_mse else "FLAT")
        print(f"alpha {alphas[0]} -> {alphas[-1]}: MSE {first_mse:.4f} -> {last_mse:.4f}  ({direction})")
        print()


if __name__ == "__main__":
    main()
