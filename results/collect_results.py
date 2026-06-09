#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_results.py — 从 logs/ 目录汇总实验结果，生成 CSV、LaTeX 表格和对比图

用法:
    python results/collect_results.py           # 默认汇总 logs/，输出到 results/
    python results/collect_results.py --dir ./logs --out results/
"""

import os
import re
import sys
import glob
import argparse
import statistics
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    print("[ERROR] 需要 pandas。请: pip install pandas")
    sys.exit(1)

# 可选依赖：matplotlib 只在绘图时需要
try:
    import matplotlib
    matplotlib.use("Agg")  # 无 GUI 环境
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ------------------------------------------------------------------
# 1. 解析单个日志文件
# ------------------------------------------------------------------
# 目标格式（train.py 末尾输出的 DataFrame 行）：
# 0  ETTm2  Transformer  none  2023  96  96  15.21816  2.55498  3.90105
# 列依次是: index, data, model, norm, seed, seq_len, pred_len, mse, mae, rmse
# ------------------------------------------------------------------

RESULT_RE = re.compile(
    r"^\s*\d+\s+"              # index
    r"(\S+)\s+"                # data
    r"(\S+)\s+"                # model
    r"(\S+)\s+"                # norm
    r"(\d+)\s+"                # seed
    r"(\d+)\s+"                # seq_len
    r"(\d+)\s+"                # pred_len
    r"([\d.]+)\s+"             # mse
    r"([\d.]+)\s+"             # mae
    r"([\d.]+)\s*$"            # rmse
)


def parse_log(path):
    """解析单个日志文件，返回 dict 或 None"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = RESULT_RE.match(line)
            if m:
                return {
                    "data": m.group(1),
                    "model": m.group(2),
                    "norm": m.group(3),
                    "seed": int(m.group(4)),
                    "seq_len": int(m.group(5)),
                    "pred_len": int(m.group(6)),
                    "mse": float(m.group(7)),
                    "mae": float(m.group(8)),
                    "rmse": float(m.group(9)),
                    "log_file": os.path.basename(path),
                }
    return None


# ------------------------------------------------------------------
# 2. 汇总所有日志
# ------------------------------------------------------------------
def collect_all(log_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
        r = parse_log(path)
        if r:
            rows.append(r)
    if not rows:
        print(f"[WARN] {log_dir}/ 中没有找到可解析的结果")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["data", "model", "norm", "pred_len", "seed"]).reset_index(drop=True)
    print(f"[OK] 收集到 {len(df)} 条实验记录")
    return df


# ------------------------------------------------------------------
# 3. 按配置分组计算均值±标准差
# ------------------------------------------------------------------
def aggregate(df):
    """对同一 (data, model, norm, seq_len, pred_len) 的多 seed 聚合"""
    if df.empty:
        return df
    grp = df.groupby(["data", "model", "norm", "seq_len", "pred_len"])
    agg = grp.agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    # 修复：单个 seed 的 std 是 NaN，设为 0
    for col in ["mse_std", "mae_std"]:
        agg[col] = agg[col].fillna(0.0)
    return agg


# ------------------------------------------------------------------
# 4. 生成 Table-2 风格的 LaTeX 表格（多变量，Autoformer/Informer/Transformer）
# ------------------------------------------------------------------
def to_latex_table2(agg):
    """Table 2 风格：每行为 pred_len，每两列为 (method, +DishTS) 的 MSE / MAE"""
    if agg.empty:
        return ""

    # 过滤：多变量 (seq_len=96)，只看 ETTm2 数据集 + Autoformer
    sub = agg[(agg["seq_len"] == 96) & (agg["data"] == "ETTm2")]
    if sub.empty:
        return "% 暂无可汇总的 ETTm2 多变量结果（需先运行 table2）"

    models = sorted(sub["model"].unique())
    pred_lens = sorted(sub["pred_len"].unique())

    lines = []
    lines.append(r"%% 注：此表格由 results/collect_results.py 自动生成")
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{复现 Table 2 — ETTm2 多变量预测结果 (MSE / MAE)}")
    lines.append(r"  \begin{tabular}{cc|cc|cc|cc}")
    lines.append(r"    \hline")
    header = "    & Length"
    for m in models:
        header += f" & {m} & +Dish-TS"
    lines.append(header + r" \\")
    lines.append(r"    \hline")

    for pl in pred_lens:
        row = f"    & {pl}"
        for m in models:
            # none
            base = sub[(sub["model"] == m) & (sub["norm"] == "none") & (sub["pred_len"] == pl)]
            dish = sub[(sub["model"] == m) & (sub["norm"] == "dishts") & (sub["pred_len"] == pl)]
            if not base.empty:
                r = base.iloc[0]
                row += f" & {r['mse_mean']:.3f}/{r['mae_mean']:.3f}"
            else:
                row += " & —/—"
            if not dish.empty:
                r = dish.iloc[0]
                row += f" & {r['mse_mean']:.3f}/{r['mae_mean']:.3f}"
            else:
                row += " & —/—"
        lines.append(row + r" \\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:ettm2_multi}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# 5. 生成 Table-3 风格对比（RevIN vs Dish-TS）
# ------------------------------------------------------------------
def to_latex_table3(agg):
    sub = agg[(agg["seq_len"] == 96) & (agg["model"] == "Autoformer")]
    if sub.empty:
        return "% 暂无可汇总的 Table-3 结果（需先运行 table3）"

    datas = sorted(sub["data"].unique())
    pred_lens = sorted(sub["pred_len"].unique())

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{复现 Table 3 — RevIN vs Dish-TS (Autoformer backbone, MSE)}")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \hline")
    lines.append(r"    Dataset & Length & RevIN & Dish-TS & Improvement(\%) \\")
    lines.append(r"    \hline")

    for data in datas:
        for pl in pred_lens:
            rev = sub[(sub["data"] == data) & (sub["norm"] == "revin") & (sub["pred_len"] == pl)]
            dis = sub[(sub["data"] == data) & (sub["norm"] == "dishts") & (sub["pred_len"] == pl)]
            if rev.empty or dis.empty:
                continue
            r_mse = rev.iloc[0]["mse_mean"]
            d_mse = dis.iloc[0]["mse_mean"]
            improve = (r_mse - d_mse) / r_mse * 100 if r_mse > 0 else 0.0
            lines.append(f"    {data} & {pl} & {r_mse:.3f} & {d_mse:.3f} & {improve:.1f} \\\\")

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \label{tab:revin_vs_dishts}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# 6. 绘图：pred_len vs MSE 柱状对比
# ------------------------------------------------------------------
def plot_comparison(agg, out_dir):
    if not HAS_MPL:
        print("[SKIP] matplotlib 未安装，跳过绘图")
        return
    if agg.empty:
        return

    # 图 1：ETTm2 + Autoformer，不同归一化下 pred_len vs MSE
    sub = agg[(agg["data"] == "ETTm2") & (agg["model"] == "Autoformer") & (agg["seq_len"] == 96)]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for norm, color, marker in [("none", "#bbb", "o"), ("revin", "#4c72b0", "s"), ("dishts", "#c44e52", "D")]:
            part = sub[sub["norm"] == norm].sort_values("pred_len")
            if part.empty:
                continue
            ax.plot(part["pred_len"], part["mse_mean"], marker=marker, color=color,
                    label=norm, linewidth=2, markersize=8)
            if len(part) > 1 and (part["mse_std"] > 0).any():
                ax.fill_between(part["pred_len"],
                                part["mse_mean"] - part["mse_std"],
                                part["mse_mean"] + part["mse_std"],
                                alpha=0.15, color=color)
        ax.set_xlabel("Prediction Length")
        ax.set_ylabel("MSE (lower is better)")
        ax.set_title("ETTm2 — Autoformer with different normalization")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, "ETTm2_Autoformer_MSE_vs_predlen.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [PLOT] {out}")

    # 图 2：RevIN vs Dish-TS 跨数据集柱状对比
    sub = agg[(agg["model"] == "Autoformer") & (agg["seq_len"] == 96) & (agg["pred_len"] == 96)]
    if not sub.empty and len(sub["data"].unique()) > 1:
        datas = sorted(sub["data"].unique())
        x = np.arange(len(datas))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.35
        rev_mses = [sub[(sub["data"] == d) & (sub["norm"] == "revin")]["mse_mean"].iloc[0]
                    if not sub[(sub["data"] == d) & (sub["norm"] == "revin")].empty else 0 for d in datas]
        dis_mses = [sub[(sub["data"] == d) & (sub["norm"] == "dishts")]["mse_mean"].iloc[0]
                    if not sub[(sub["data"] == d) & (sub["norm"] == "dishts")].empty else 0 for d in datas]
        ax.bar(x - w/2, rev_mses, w, label="RevIN", color="#4c72b0")
        ax.bar(x + w/2, dis_mses, w, label="Dish-TS", color="#c44e52")
        ax.set_xticks(x)
        ax.set_xticklabels(datas)
        ax.set_ylabel("MSE (pred_len=96)")
        ax.set_title("RevIN vs Dish-TS across datasets (Autoformer backbone)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        out = os.path.join(out_dir, "RevIN_vs_DishTS_across_datasets.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [PLOT] {out}")

    # 图 3：pred_len vs MSE 多模型对比（Dish-TS）
    sub = agg[(agg["data"] == "ETTm2") & (agg["norm"] == "dishts") & (agg["seq_len"] == 96)]
    if not sub.empty and len(sub["model"].unique()) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model, color in zip(sorted(sub["model"].unique()), ["#4c72b0", "#55a868", "#c44e52"]):
            part = sub[sub["model"] == model].sort_values("pred_len")
            ax.plot(part["pred_len"], part["mse_mean"], marker="o", label=model,
                    color=color, linewidth=2, markersize=8)
        ax.set_xlabel("Prediction Length")
        ax.set_ylabel("MSE (with Dish-TS)")
        ax.set_title("ETTm2 — Model comparison with Dish-TS")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, "ETTm2_models_MSE_with_DishTS.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  [PLOT] {out}")


# ------------------------------------------------------------------
# 7. CLI
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./logs", help="日志目录 (default: ./logs)")
    ap.add_argument("--out", default="./results", help="输出目录 (default: ./results)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("  Results Collection")
    print("=" * 60)
    print(f"  log dir : {args.dir}")
    print(f"  output  : {args.out}")
    print()

    # 原始结果
    df = collect_all(args.dir)
    if df.empty:
        return
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False, float_format="%.6f")
    print(f"  [CSV] {args.out}/summary.csv ({len(df)} rows)")

    # 聚合
    agg = aggregate(df)
    agg.to_csv(os.path.join(args.out, "summary_aggregated.csv"), index=False, float_format="%.6f")
    print(f"  [CSV] {args.out}/summary_aggregated.csv ({len(agg)} configs)")

    # LaTeX
    tex_table2 = to_latex_table2(agg)
    with open(os.path.join(args.out, "table2_ettm2_multivariate.tex"), "w") as f:
        f.write(tex_table2 + "\n")
    print(f"  [TEX] {args.out}/table2_ettm2_multivariate.tex")

    tex_table3 = to_latex_table3(agg)
    with open(os.path.join(args.out, "table3_revin_vs_dishts.tex"), "w") as f:
        f.write(tex_table3 + "\n")
    print(f"  [TEX] {args.out}/table3_revin_vs_dishts.tex")

    # 绘图
    print()
    print("  Generating plots ...")
    plot_comparison(agg, args.out)

    # 汇总文本，便于直接查看
    print()
    print("=" * 60)
    print("  Quick Summary — ETTm2 Autoformer")
    print("=" * 60)
    sub = agg[(agg["data"] == "ETTm2") & (agg["model"] == "Autoformer") & (agg["seq_len"] == 96)]
    if not sub.empty:
        print(sub[["norm", "pred_len", "mse_mean", "mae_mean", "n_seeds"]].to_string(index=False))
    else:
        print("  (暂无 ETTm2 Autoformer 结果)")

    print()
    print("  → 所有结果保存在 results/ 目录中")


if __name__ == "__main__":
    main()
