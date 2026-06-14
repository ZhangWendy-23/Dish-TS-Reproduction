import pandas as pd
import numpy as np

df = pd.read_csv("results/figure3_runs.csv")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
print()

for ds in sorted(df.dataset.unique()):
    sub_ds = df[df.dataset == ds]
    for pl in sorted(sub_ds.pred_len.unique()):
        sub = sub_ds[sub_ds.pred_len == pl]
        scaled_val = sub.scaled.iloc[0] if "scaled" in sub.columns else "?"
        print(f"-- {ds} pred_len={pl}  ({len(sub)} rows)  scaled={scaled_val}")
        grp = sub.groupby(["norm", "prior"]).apply(
            lambda g: pd.Series({
                "alphas": sorted(set(round(float(a), 2) for a in g.alpha.values)),
                "n_seeds": len(set(int(s) for s in g.seed.values)),
                "MSE_mean": float(g.MSE.mean()),
            })
        ).reset_index()
        print(grp.to_string(index=False))
        print()
