import pandas as pd
import os

BASE = "paper_results"


def analyze_univariate_multivariate():
    """Compare Dish-TS improvement across univariate/multivariate settings."""
    uni = pd.read_csv(os.path.join(BASE, "table1_univariate.csv"))
    multi = pd.read_csv(os.path.join(BASE, "table2_multivariate.csv"))

    print("=" * 80)
    print("A. Table 1 (Univariate) vs Table 2 (Multivariate) -- Dish-TS improvement")
    print("=" * 80)

    for label, df in [("Univariate (Table 1)", uni), ("Multivariate (Table 2)", multi)]:
        print(f"\n--- {label} ---")
        # Average MSE reduction for each backbone model
        for backbone in ["informer", "autoformer", "nbeats"]:
            base_mse = f"{backbone}_mse"
            dishts_mse = f"{backbone}_dishts_mse"
            imp = (df[base_mse] - df[dishts_mse]) / df[base_mse] * 100
            print(f"  {backbone:>10s}: avg MSE reduction = {imp.mean():.1f}%  "
                  f"(min {imp.min():.1f}%, max {imp.max():.1f}%)")

        # Print by dataset
        for ds in df["dataset"].unique():
            sub = df[df["dataset"] == ds]
            print(f"  Dataset {ds:>12s}: horizons={list(sub['horizon'])}")
            for backbone in ["informer", "autoformer", "nbeats"]:
                base_mse = f"{backbone}_mse"
                dishts_mse = f"{backbone}_dishts_mse"
                imp = (sub[base_mse].mean() - sub[dishts_mse].mean()) / sub[base_mse].mean() * 100
                print(f"    {backbone:>10s}: avg MSE reduction = {imp:.1f}%")


def analyze_revin():
    """Table 3 -- Dish-TS vs RevIN."""
    df = pd.read_csv(os.path.join(BASE, "table3_revin_comparison.csv"))
    print("\n" + "=" * 80)
    print("B. Table 3: Dish-TS vs RevIN (Autoformer backbone, multivariate)")
    print("=" * 80)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n  Dataset {ds}:")
        for _, row in sub.iterrows():
            print(f"    horizon={int(row['horizon']):>4d}  RevIN MSE={row['revin_mse']:.4f}  "
                  f"Dish-TS MSE={row['dishts_mse']:.4f}  "
                  f"Improvement={row['improvement_pct']:.1f}%")

        avg_imp = sub["improvement_pct"].mean()
        print(f"    -> Average improvement over RevIN: {avg_imp:.1f}%")


def analyze_long_horizon():
    """Table 4 -- Impact of longer horizons (336-720)."""
    df = pd.read_csv(os.path.join(BASE, "table4_long_horizon.csv"))
    print("\n" + "=" * 80)
    print("C. Table 4: Long-horizon forecasting (lookback=96 fixed, N-BEATS backbone)")
    print("=" * 80)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n  Dataset {ds}:")
        for _, row in sub.iterrows():
            imp = (row["backbone_mse"] - row["dishts_mse"]) / row["backbone_mse"] * 100
            print(f"    horizon={int(row['horizon']):>4d}  baseline MSE={row['backbone_mse']:.4f}  "
                  f"Dish-TS MSE={row['dishts_mse']:.4f}  improvement={imp:.1f}%")


def analyze_lookback():
    """Table 5 -- Impact of lookback length."""
    df = pd.read_csv(os.path.join(BASE, "table5_lookback.csv"))
    print("\n" + "=" * 80)
    print("D. Table 5: Lookback length analysis (horizon=48 fixed, N-BEATS backbone)")
    print("=" * 80)

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n  Dataset {ds}:")
        for _, row in sub.iterrows():
            imp = (row["backbone_mse"] - row["dishts_mse"]) / row["backbone_mse"] * 100
            print(f"    lookback={int(row['lookback']):>4d}  baseline MSE={row['backbone_mse']:.4f}  "
                  f"Dish-TS MSE={row['dishts_mse']:.4f}  improvement={imp:.1f}%")


def cross_table_summary():
    """Cross-table summary."""
    print("\n" + "=" * 80)
    print("E. Cross-table summary of Dish-TS behavior")
    print("=" * 80)

    # Key observations:
    print("\n  Key observations from the 5 tables:")
    print("  1. Table 1 & 2: Dish-TS consistently improves MSE across all 3 backbones")
    print("     (Informer, Autoformer, N-BEATS) in both uni- and multivariate settings.")
    print("  2. Improvement is generally LARGER in multivariate (Table 2) than univariate")
    print("     (Table 1), suggesting Dish-TS handles distribution shift across channels.")
    print("  3. Table 3: Dish-TS consistently outperforms RevIN (7-36% MSE reduction).")
    print("     Improvement is largest on ETTh1 (up to 36.3%), smallest on Weather (~10%).")
    print("  4. Table 4: As horizon increases (336->720), Dish-TS continues to outperform")
    print("     baseline N-BEATS -- improvement holds/stabilizes, confirming long-horizon")
    print("     robustness.")
    print("  5. Table 5: As lookback window grows (48->240), Dish-TS improvement typically")
    print("     increases (more context -> Dish-TS gains more). Peak around lookback=192")
    print("     for Electricity, while ETTh1 improves steadily.")


if __name__ == "__main__":
    analyze_univariate_multivariate()
    analyze_revin()
    analyze_long_horizon()
    analyze_lookback()
    cross_table_summary()
