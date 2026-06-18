"""Compare dishts vs revin vs none (Phase 3) and paper."""
import csv
from collections import defaultdict
from statistics import mean, stdev

cells = defaultdict(list)
with open("results/figure3_runs.csv", newline="") as fh:
    rows = list(csv.DictReader(fh))

for r in rows:
    ds = r["dataset"]
    norm = r["norm"]
    pl = int(r["pred_len"])
    try:
        mse = float(r["MSE"])
    except (KeyError, ValueError):
        continue
    if mse != mse:
        continue
    cells[(ds, norm, pl, r["alpha"])].append(mse)

# Paper Table 2 numbers (Autoformer, Multivariate, Dish-TS)
PAPER = {
    ("ETTm2", 24): 0.676, ("ETTm2", 96): 0.929, ("ETTm2", 168): 1.308, ("ETTm2", 336): 1.603,
    ("WTH", 24): 2.481, ("WTH", 96): 3.28, ("WTH", 168): 3.31, ("WTH", 336): 3.31,
    ("ETTh1", 24): 1.019, ("ETTh1", 96): 1.199, ("ETTh1", 168): 1.148, ("ETTh1", 336): 1.147,
    ("ECL", 24): 0.04, ("ECL", 96): 0.064, ("ECL", 168): 0.08, ("ECL", 336): 0.104,
}


def stat(d, n, p, a):
    arr = cells.get((d, n, p, a), [])
    if not arr:
        return None, 0
    return mean(arr), len(arr)


print(f'{"Dataset":<8} {"H":>4} | {"none":>13} | {"revin":>13} | {"dishts(α=0)":>14} | {"dishts(best α)":>17} | paper | ratio | best(fair)')
print("-" * 140)

wins = {"dishts": 0, "revin": 0, "none": 0, "tie": 0}
for ds in ["ETTm2", "WTH", "ETTh1", "ECL"]:
    for pl in [24, 96, 168, 336]:
        nv, nn = stat(ds, "none", pl, "0.0")
        rv, rn = stat(ds, "revin", pl, "0.0")
        d0v, d0n = stat(ds, "dishts", pl, "0.0")
        # Best dishts
        best_mses = []
        for (d, n, p, a), arr in cells.items():
            if d == ds and n == "dishts" and p == pl:
                for x in arr:
                    best_mses.append((x, float(a)))
        if best_mses:
            best_mse, best_a = min(best_mses, key=lambda x: x[0])
            best_n = sum(1 for m, a in best_mses if m == best_mse and a == best_a)
        else:
            best_mse, best_a, best_n = (None, 0.0, 0)
        pap = PAPER.get((ds, pl), None)
        # Fair best (only none/revin/dishtsα=0)
        fair = []
        if nv: fair.append(("none", nv, nn))
        if rv: fair.append(("revin", rv, rn))
        if d0v: fair.append(("dishts", d0v, d0n))
        if len(fair) == 3:
            m = min(fair, key=lambda x: x[1])
            tied = [n for n, v, _ in fair if abs(v - m[1]) < 0.001]
            if len(tied) == 1:
                wins[tied[0]] += 1
            else:
                wins["tie"] += 1
        fair_best = min(fair, key=lambda x: x[1])[0] if fair else "-"
        # Ratio best/paper
        ratio_str = "-"
        if pap and best_mse:
            ratio = best_mse / pap
            if 0.85 <= ratio <= 1.15: tag = "OK"
            elif ratio < 0.85: tag = "better"
            else: tag = "worse"
            ratio_str = f"{ratio:.2f}x {tag}"
        def fmt(v, n):
            return f"{v:.3f}(n={n})" if v is not None else "-"
        best_str = f"{best_mse:.3f} a={best_a:.2f}(n={best_n})" if best_mse else "-"
        print(f'{ds:<8} {pl:>4} | {fmt(nv,nn):>13} | {fmt(rv,rn):>13} | {fmt(d0v,d0n):>14} | {best_str:>17} | {pap:>5.3f} | {ratio_str:>12} | {fair_best}')

print()
print(f"Fair-comparison wins (dishts/α=0 vs revin vs none, 16 cells): {wins}")
print(f"dishts(α=0) wins: {wins['dishts']}/{16}, revin wins: {wins['revin']}, none wins: {wins['none']}, ties: {wins['tie']}")
