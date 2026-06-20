#!/usr/bin/env python3
"""Analyze figure3_runs.csv — compute clean summaries per (dataset, pred_len, model, norm),
build paper comparison tables, flag anomalies."""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "results" / "figure3_runs.csv"

# Paper Table 2 reference MSE values (Autoformer backbone, dishts, paper-scale normalized)
# Keys: (dataset, pred_len) -> {"paper_autoformer": x, "paper_autoformer_dishts": y, "paper_revin": z, "paper_dishts": w}
PAPER_TABLE2 = {
    ("ETTm2", 24):  (2.443, 0.676, 0.794, 0.651),
    ("ETTm2", 48):  (2.781, 0.823, None,  None),
    ("ETTm2", 96):  (3.017, 0.929, None,  None),
    ("ETTm2", 168): (3.656, 1.308, 1.501, 1.325),
    ("ETTm2", 336): (3.638, 1.603, 1.827, 1.599),
    ("ETTh1", 24):  (1.794, 1.019, 1.245, 1.018),
    ("ETTh1", 48):  (2.127, 1.240, None,  None),
    ("ETTh1", 96):  (2.965, 1.199, None,  None),
    ("ETTh1", 168): (3.234, 1.148, 1.462, 1.148),
    ("ETTh1", 336): (3.335, 1.147, 1.920, 1.222),
    ("WTH",   24):  (2.120, 2.481, 3.523, 2.481),
    ("WTH",   96):  (2.042, 3.280, None,  None),
    ("WTH",   168): (2.106, 3.309, 3.658, 3.283),
    ("WTH",   336): (2.567, 3.314, 3.501, 3.232),
    ("ECL",   24):  (0.249, 0.040, 0.044, 0.039),
    ("ECL",   48):  (1.002, 0.051, None,  None),
    ("ECL",   96):  (1.046, 0.064, None,  None),
    ("ECL",   168): (1.013, 0.080, 0.091, 0.076),
    ("ECL",   336): (1.058, 0.104, 0.109, 0.086),
}

def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["pred_len"] = int(r["pred_len"])
                r["seq_len"] = int(r["seq_len"])
                r["alpha"] = float(r["alpha"])
                r["seed"] = int(r["seed"])
                r["MSE"] = float(r["MSE"])
            except (KeyError, ValueError):
                continue
            if math.isnan(r["MSE"]) or math.isinf(r["MSE"]):
                continue
            rows.append(r)
    return rows


def summarize_by_key(rows, key_fn):
    """Aggregate by key_fn(row), return dict key -> (mean, sd, n, vals)."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r["MSE"])
    out = {}
    for k, v in buckets.items():
        n = len(v)
        mean = sum(v) / n
        var = sum((x - mean) ** 2 for x in v) / n
        sd = math.sqrt(var)
        out[k] = (mean, sd, n, sorted(v))
    return out


def mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")


def main() -> int:
    rows = load_rows()
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    # ── 1. Summary by (dataset, pred_len, model, norm, alpha, prior)
    key_full = lambda r: (r["dataset"], r["pred_len"], r["model"], r["norm"], r["alpha"], r["prior"])
    agg_full = summarize_by_key(rows, key_full)

    print("\n" + "="*110)
    print("SECTION 1 — Per (dataset, pred_len, model, norm, alpha, prior) summary")
    print("="*110)
    for k in sorted(agg_full.keys()):
        m, sd, n, vals = agg_full[k]
        # only print the "alpha=0.0, prior=none" rows as they are the main comparison set
        if k[4] == 0.0 and k[5] == "none":
            print(f"  {k[0]:<6} H={k[1]:>4} model={k[2]:<11} norm={k[3]:<7} mean={m:>14.6g} sd={sd:>12.6g} n={n}")

    # ── 2. Paper Table 2 / 3 comparison: focus on (dataset, pred_len, Autoformer, norm ∈ {none, revin, dishts})
    print("\n" + "="*110)
    print("SECTION 2 — Fair paper comparison (α=0.0, prior=none, paper-scale applied)")
    print("="*110)

    # Compute per-dataset scale factors from the "Autoformer, none" MSE at each horizon using paper
    # We compute it as a sanity check: for our none-MSE on Autoformer we should roughly match paper_autoformer
    norm_none_key = lambda r: (r["dataset"], r["pred_len"]) if r["model"] == "Autoformer" and r["norm"] == "none" and r["alpha"] == 0.0 and r["prior"] == "none" else None

    none_agg = defaultdict(list)
    for r in rows:
        if r["model"] == "Autoformer" and r["norm"] == "none" and r["alpha"] == 0.0 and r["prior"] == "none":
            none_agg[(r["dataset"], r["pred_len"])].append(r["MSE"])

    # compute per-dataset scale factor using all available horizons' average
    ds_scales = {}
    for ds in set(k[0] for k in PAPER_TABLE2.keys()):
        ratios = []
        for (d, h), (paper_af, _, _, _) in PAPER_TABLE2.items():
            if d == ds and (d, h) in none_agg and len(none_agg[(d, h)]) >= 2:
                mine = mean(none_agg[(d, h)])
                if paper_af and mine > 0:
                    ratios.append(paper_af / mine)
        if ratios:
            ds_scales[ds] = mean(ratios)
            print(f"  dataset={ds:<6} scale = {ds_scales[ds]:.6g} (from {len(ratios)} horizons)")
        else:
            print(f"  dataset={ds:<6} — no valid scale (ECL raw MSE huge; skip scaling there)")

    # Print clean Table-2 comparison using α=0.0 only
    print("\n" + "-"*110)
    print("  Table 2-style: Autoformer backbone, 3 norms (α=0.0, prior=none, paper-scale)")
    print("  ratio < 1 means ours BETTER than paper; > 1 means ours WORSE")
    print("-"*110)

    header = f"{'dataset':<6} {'H':>4} {'mine_none':>14} {'mine_revin':>14} {'mine_dishts':>14} {'paper_af':>11} {'paper_af_ds':>13} {'paper_revin':>12} {'paper_ds':>11} {'ratio_none':>11} {'ratio_revin':>11} {'ratio_ds':>11}"
    print(header)
    # filter: only Autoformer α=0.0, prior=none
    ds_agg = defaultdict(dict)   # (ds, h) -> {norm: mean}
    for r in rows:
        if r["model"] != "Autoformer" or r["alpha"] != 0.0 or r["prior"] != "none":
            continue
        key = (r["dataset"], r["pred_len"])
        ds_agg[key].setdefault(r["norm"], []).append(r["MSE"])

    rows_out = []
    for key in sorted(ds_agg.keys()):
        ds, h = key
        mine_none = mean(ds_agg[key].get("none", []))
        mine_revin = mean(ds_agg[key].get("revin", []))
        mine_dishts = mean(ds_agg[key].get("dishts", []))
        paper_refs = PAPER_TABLE2.get((ds, h))
        if paper_refs is None:
            continue
        paper_af, paper_af_ds, paper_revin, paper_ds = paper_refs
        scale = ds_scales.get(ds, 1.0)
        # scale our raw MSE down to paper scale for direct comparison
        s_none = mine_none * scale if mine_none == mine_none else float("nan")
        s_revin = mine_revin * scale if mine_revin == mine_revin else float("nan")
        s_dishts = mine_dishts * scale if mine_dishts == mine_dishts else float("nan")
        r_none = s_none / paper_af if paper_af and s_none else float("nan")
        r_revin = s_revin / paper_revin if paper_revin and s_revin else float("nan")
        r_dishts = s_dishts / paper_ds if paper_ds and s_dishts else float("nan")
        row_tuple = (ds, h, s_none, s_revin, s_dishts,
                     paper_af if paper_af else 0, paper_af_ds if paper_af_ds else 0,
                     paper_revin if paper_revin else 0, paper_ds if paper_ds else 0,
                     r_none, r_revin, r_dishts)
        rows_out.append(row_tuple)
        print(f"  {ds:<6} {h:>4} {s_none:>14.6g} {s_revin:>14.6g} {s_dishts:>14.6g} "
              f"{paper_af if paper_af else '-':>11} {paper_af_ds if paper_af_ds else '-':>13} "
              f"{paper_revin if paper_revin else '-':>12} {paper_ds if paper_ds else '-':>11} "
              f"{r_none:>11.3f} {r_revin:>11.3f} {r_dishts:>11.3f}")

    # ── 3. Informer + NBEATS data check (for Table 2 / long-horizon)
    print("\n" + "="*110)
    print("SECTION 3 — Informer / N-BEATS coverage")
    print("="*110)
    for model in ("Informer", "NBEATS"):
        print(f"\n  Model: {model} (α=0.0, prior=none):")
        subset = [r for r in rows if r["model"] == model and r["alpha"] == 0.0 and r["prior"] == "none"]
        by = defaultdict(list)
        for r in subset:
            by[(r["dataset"], r["pred_len"], r["norm"])].append(r["MSE"])
        for k in sorted(by.keys()):
            mean_v = sum(by[k]) / len(by[k])
            print(f"    {k[0]:<6} H={k[1]:>4} norm={k[2]:<7} mean={mean_v:>12.6g}  n={len(by[k])}")

    # ── 4.  Flag anomaly: Informer-ECL huge variance
    print("\n" + "="*110)
    print("SECTION 4 — ANOMALY CHECK: Informer ECL huge MSE spread")
    print("="*110)
    eclip = [r for r in rows if r["dataset"] == "ECL" and r["model"] == "Informer" and r["norm"] == "none"]
    # group by pred_len
    eclip_by_h = defaultdict(list)
    for r in eclip:
        eclip_by_h[r["pred_len"]].append((r["seed"], r["MSE"], r["timestamp"]))
    for h in sorted(eclip_by_h.keys()):
        entries = eclip_by_h[h]
        vals = [x[1] for x in entries]
        print(f"  H={h:>4} n={len(entries):<3} min={min(vals):.6g} max={max(vals):.6g} mean={mean(vals):.6g} ratio_max/min={max(vals)/min(vals):.2f}")
        for seed_v, mse_v, ts_v in sorted(entries, key=lambda x: (x[0], x[1])):
            print(f"    seed={seed_v} MSE={mse_v:.6g} ts={ts_v}")

    # ── 5. Table 6 (CONET init ablation) summary
    print("\n" + "="*110)
    print("SECTION 5 — Table 6: CONET init (avg vs norm vs uni)")
    print("="*110)
    # dish_init values: "avg", "norm", "uni"
    t6_rows = [r for r in rows if r["norm"] == "dishts" and r["alpha"] == 0.0 and r["prior"] == "none" and r["dish_init"] in ("avg", "norm", "uni")]
    by_t6 = defaultdict(list)
    for r in t6_rows:
        by_t6[(r["dataset"], r["pred_len"], r["model"], r["dish_init"])].append(r["MSE"])
    # pivot: for each (dataset, pred_len, model) compare 3 init methods
    pivots = defaultdict(dict)
    for k, lst in by_t6.items():
        pivots[(k[0], k[1], k[2])][k[3]] = mean(lst)
    print(f"  {'dataset':<6} {'model':<11} {'H':>4} {'avg':>12} {'norm':>12} {'uni':>12} {'best':>5}")
    for key in sorted(pivots.keys()):
        ds, h, model = key
        vals = pivots[key]
        if len(vals) < 3:
            continue
        best_init = min(vals, key=lambda x: vals[x])
        print(f"  {ds:<6} {model:<11} {h:>4} {vals.get('avg', float('nan')):>12.6g} {vals.get('norm', float('nan')):>12.6g} {vals.get('uni', float('nan')):>12.6g} {best_init:>5}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
