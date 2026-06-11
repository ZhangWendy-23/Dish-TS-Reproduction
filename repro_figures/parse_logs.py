"""Parse completed experiment logs and extract MSE/metrics to CSVs.

After a training run crashes on the DataFrame-write step (train.py L272),
the MSE line is already printed to the log / console.  This script scans all
log files under ``logs/``, extracts those metrics, and appends them to:

    results/paper_summary.csv
    results/figure3_runs.csv

Duplicate rows (same data/model/norm/seed/pred_len) are skipped.

Usage
-----
    python3 repro_figures/parse_logs.py                     # scan logs/
    python3 repro_figures/parse_logs.py --log-dir logs --data ETTh1  # filter
    python3 repro_figures/parse_logs.py --dry-run          # print only, no write
"""
from __future__ import annotations

import argparse
import datetime
import glob
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LOG_DIR = os.path.join(ROOT, "logs")
RESULTS_DIR = os.path.join(ROOT, "results")


# ---------- pattern extraction from a single log file ----------

_RE_SUMMARY_LINE = re.compile(
    r"^\s*DATA=(?P<data>\S+)\s+MODEL=(?P<model>\S+)\s+NORM=(?P<norm>\S+)"
    r"\s+SEED=(?P<seed>\d+).*"
    r"seq_len=(?P<seq_len>\d+)\s+label_len=(?P<label_len>\d+)\s+"
    r"pred_len=(?P<pred_len>\d+)\s+batch_size=(?P<batch_size>\d+)\s+"
    r"lr=(?P<lr>[\d.e+-]+)\s+alpha=(?P<alpha>[\d.]+)"
)
_RE_METRICS_LINE = re.compile(
    r"^\s*MSE=(?P<MSE>[\d.]+)\s+MAE=(?P<MAE>[\d.]+)\s+"
    r"RMSE=(?P<RMSE>[\d.]+)\s+MAPE=(?P<MAPE>[\d.]+)\s+"
    r"MSPE=(?P<MSPE>[\d.]+)"
)
_RE_LOG_FILENAME = re.compile(
    r"^(?P<data>[A-Za-z0-9]+)_(?P<model>[A-Za-z0-9]+)_(?P<norm>[a-z]+)_"
    r"(?:alpha(?P<alpha>[\d.]+)_)?"
    r"(?:s(?P<seq_len>\d+)_)?"
    r"p(?P<pred_len>\d+)_seed(?P<seed>\d+)\.log$"
)
# Also match "pd" prefix used by alpha sweep filenames
_RE_LOG_FILENAME_PD = re.compile(
    r"^(?P<data>[A-Za-z0-9]+)_(?P<model>[A-Za-z0-9]+)_(?P<norm>[a-z]+)_"
    r"(?:alpha(?P<alpha>[\d.]+)_)?"
    r"pd(?P<pred_len>\d+)_seed(?P<seed>\d+)\.log$"
)


def _parse_one_log(path: str) -> list[dict]:
    """Return one raw-dict per successfully-completed run in this log."""
    try:
        content = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return []

    results = []
    pending = None  # summary line dict, carried over to the next metrics line
    for line in content.splitlines():
        sm = _RE_SUMMARY_LINE.search(line)
        mm = _RE_METRICS_LINE.search(line)
        if sm and not mm:
            pending = sm.groupdict()
            continue
        if mm:
            entry = (pending.copy() if pending else {})
            entry.update(mm.groupdict())
            if entry.get("data"):
                # Fill alpha from filename if missing from summary
                if not entry.get("alpha") and (
                        _RE_LOG_FILENAME.match(os.path.basename(path)) or
                        _RE_LOG_FILENAME_PD.match(os.path.basename(path))):
                    for pat in (_RE_LOG_FILENAME, _RE_LOG_FILENAME_PD):
                        fn_info = pat.match(os.path.basename(path))
                        if fn_info:
                            fn_info = fn_info.groupdict()
                            if fn_info.get("alpha"):
                                entry["alpha"] = fn_info["alpha"]
                            break
                if not entry.get("alpha"):
                    entry["alpha"] = "0.5"
                results.append(entry)
            pending = None
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--data", default=None,
                    help="Only process logs for a specific dataset.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    log_files = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
    if args.data:
        log_files = [f for f in log_files
                     if os.path.basename(f).startswith(args.data)]

    all_entries = []
    skipped = 0
    for lf in log_files:
        entries = _parse_one_log(lf)
        if not entries:
            skipped += 1
            if args.verbose:
                print(f"[skip] {os.path.basename(lf)} (no parseable metrics)")
            continue
        all_entries.extend(entries)
        if args.verbose:
            for e in entries:
                print(f"[found] {lf}: data={e['data']} "
                      f"norm={e['norm']} seed={e['seed']} "
                      f"pred_len={e['pred_len']} MSE={e['MSE']}")

    if not all_entries:
        print("No parseable results found in any log file.")
        return

    df_new = pd.DataFrame(all_entries)
    # Ensure numeric columns
    for col in ["seed", "seq_len", "label_len", "pred_len", "batch_size",
                "MSE", "MAE", "RMSE", "MAPE", "MSPE", "alpha"]:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    # Merge into paper_summary.csv
    summary_path = os.path.join(RESULTS_DIR, "paper_summary.csv")
    summary_cols = ["data", "model", "norm", "seed", "seq_len", "label_len",
                    "pred_len", "batch_size", "alpha", "MSE", "MAE",
                    "RMSE", "MAPE", "MSPE", "log"]
    missing_cols = [c for c in summary_cols if c not in df_new.columns]
    for mc in missing_cols:
        df_new[mc] = ""
    df_new = df_new[summary_cols]

    if os.path.exists(summary_path):
        df_old = pd.read_csv(summary_path)
        # Dedup: same (data, model, norm, seed, pred_len) -> keep old
        dedup_cols = ["data", "model", "norm", "seed", "pred_len"]
        existing_keys = df_old[dedup_cols].drop_duplicates()
        merged_key = df_new[dedup_cols].copy()
        # Only keep rows from df_new that are NOT in df_old
        if not existing_keys.empty:
            mask = ~merged_key.apply(
                lambda row: (existing_keys == row).all(axis=1).any(), axis=1)
            df_new = df_new[mask]
        if df_new.empty:
            print(f"All {len(all_entries)} entries already exist in "
                  f"{summary_path}.")
            return

    if args.dry_run:
        print(f"[DRY RUN] Would append {len(df_new)} rows to {summary_path}:")
        print(df_new.to_string(index=False))
        return

    write_header = not os.path.exists(summary_path)
    df_new.to_csv(summary_path, mode="a", header=write_header, index=False)
    print(f"[parse_logs] appended {len(df_new)} rows to {summary_path}")

    # Also update figure3_runs.csv
    f3_path = os.path.join(RESULTS_DIR, "figure3_runs.csv")
    f3_cols_map = {
        "data": "dataset", "model": "model", "norm": "norm",
        "pred_len": "pred_len", "alpha": "alpha", "MSE": "MSE",
        "MAE": "MAE", "RMSE": "RMSE", "MAPE": "MAPE", "MSPE": "MSPE",
        "seed": "seed", "seq_len": "seq_len",
    }
    f3_new = df_new.copy()
    f3_new.rename(columns=f3_cols_map, inplace=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f3_new["timestamp"] = ts
    f3_new = f3_new[[c for c in
                      ["dataset", "seq_len", "pred_len", "model", "norm",
                       "alpha", "MSE", "MAE", "RMSE", "MAPE", "MSPE",
                       "timestamp"]
                      if c in f3_new.columns]]

    if os.path.exists(f3_path):
        f3_old = pd.read_csv(f3_path)
        dedup_cols_f3 = ["dataset", "model", "norm", "pred_len", "alpha",
                         "seed"]
        if not f3_old.empty and "seed" in f3_old.columns:
            ...  # kept for safety
        write_header_f3 = False
    else:
        write_header_f3 = True

    if not args.dry_run:
        f3_new.to_csv(f3_path, mode="a", header=write_header_f3, index=False)
        print(f"[parse_logs] appended {len(f3_new)} rows to {f3_path}")


if __name__ == "__main__":
    main()
