#!/bin/bash
# ============================================================================
# run_paper_exps.sh — Dish-TS 论文完整复现脚本（100% 对齐论文参数）
# ============================================================================
# 参考: AAAI 2023 paper "Dish-TS: A General Paradigm for Alleviating
#        Distribution Shift in Time Series Forecasting"
#
# 论文参数（严格对齐）:
#   - optimizer: Adam, lr=1e-3
#   - patience=7, early stopping
#   - d_model=512, n_heads=8, e_layers=2, d_ff=2048
#   - batch_size: Informer=256, Autoformer/Transformer=128, ECL=64
#   - alpha=0.5 (paper: searched from 0 to 1; 0.5 is typical best)
#   - seed: single seed for Table1/2/4; 3 seeds (2023,2024,2025) for Table3
#   - data split: ETT/ILI=6:2:2, others=7:1:2
#   - No global normalization (raw values)
#   - Dish-TS init: standard (GELU activation, learned reduce_mlayer)
#
# 4 个 Table 的设计:
#   Table 1 — Univariate:    seq_len = pred_len ∈ {24,48,96,168,336}
#   Table 2 — Multivariate:  seq_len=96, pred_len ∈ {24,48,96,168,336}
#   Table 3 — RevIN对比:     seq_len=96, pred_len ∈ {24,168,336}, 3 seeds, Autoformer
#   Table 4 — 长窗口:        seq_len=96, pred_len ∈ {336,420,540,600,720}
#
# 用法:
#   ./run_paper_exps.sh table1      # Table 1
#   ./run_paper_exps.sh table2      # Table 2
#   ./run_paper_exps.sh table3      # Table 3 (多seed)
#   ./run_paper_exps.sh table4      # Table 4
#   ./run_paper_exps.sh quick       # 快速验证 (ETTm2, 3 pred_len, 3 norm, 1 model)
#   ./run_paper_exps.sh summarize   # 汇总已有结果 (skip if exist)
#   ./run_paper_exps.sh all         # 运行全部 4 个 tables
# ============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU=0
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" results

run_one() {
    # 参数: data model norm pred_len seq_len [seed] [features_flag] [extra]
    local data="$1"
    local model="$2"
    local norm="$3"
    local pred_len="$4"
    local seq_len="$5"
    local seed="${6:-2023}"
    local ft_flag="${7:-M}"
    local extra="${8:-}"

    local key="${data}_${model}_${norm}_s${seq_len}_p${pred_len}_seed${seed}"
    local LOG="$LOG_DIR/${key}.log"

    # 如果已有完整结果（包含 MSE= 行），跳过
    if [ -f "$LOG" ] && grep -q "DATA=$data  MODEL=$model  NORM=$norm" "$LOG" 2>/dev/null; then
        echo "  [SKIP] $model + $norm, seq=$seq_len pred=$pred_len (seed=$seed) — 已有结果"
        return 0
    fi

    echo "[$(date +'%H:%M:%S')] $data | $model + $norm | seq=$seq_len pred=$pred_len | seed=$seed | ft=$ft_flag"
    local t0=$(date +%s)

    python train.py \
        --data "$data" \
        --model "$model" \
        --norm "$norm" \
        --pred_len "$pred_len" \
        --seq_len "$seq_len" \
        --seed "$seed" \
        --features "$ft_flag" \
        --batch_size 0 \
        --alpha 0.5 \
        --gpu "$GPU" \
        $extra 2>&1 | tee "$LOG"

    local t1=$(date +%s)
    echo "  → 用时 $(( (t1 - t0) / 60 )) 分 $(( (t1 - t0) % 60 )) 秒"
}

# ============================================================================
# Table 1: Univariate Time Series Forecasting
# seq_len = pred_len ∈ {24, 48, 96, 168, 336}
# ============================================================================
table1() {
    echo ""
    echo "=================================================================="
    echo "  Table 1: Univariate (seq_len = pred_len)"
    echo "  pred_len ∈ {24, 48, 96, 168, 336}, datasets × 3 models × 3 norms"
    echo "=================================================================="
    local datasets=(ECL ETTh1 ETTm2 WTH ILI)
    local models=(Autoformer Informer Transformer)
    local norms=(none revin dishts)
    local pred_lens=(24 48 96 168 336)
    local total=$(( ${#datasets[@]} * ${#models[@]} * ${#norms[@]} * ${#pred_lens[@]} ))
    local count=0
    for data in "${datasets[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            for model in "${models[@]}"; do
                for norm in "${norms[@]}"; do
                    count=$((count + 1))
                    echo "[$count/$total]"
                    run_one "$data" "$model" "$norm" "$pred_len" "$pred_len" 2023 S
                done
            done
        done
    done
    echo "[Table 1] 完成 $total 个实验。"
}

# ============================================================================
# Table 2: Multivariate Time Series Forecasting
# seq_len = 96, pred_len ∈ {24, 48, 96, 168, 336}
# ============================================================================
table2() {
    echo ""
    echo "=================================================================="
    echo "  Table 2: Multivariate (seq_len=96, pred_len ∈ {24,48,96,168,336})"
    echo "=================================================================="
    local datasets=(ECL ETTh1 ETTm2 WTH ILI)
    local models=(Autoformer Informer Transformer)
    local norms=(none revin dishts)
    local pred_lens=(24 48 96 168 336)
    local total=$(( ${#datasets[@]} * ${#models[@]} * ${#norms[@]} * ${#pred_lens[@]} ))
    local count=0
    for data in "${datasets[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            for model in "${models[@]}"; do
                for norm in "${norms[@]}"; do
                    count=$((count + 1))
                    echo "[$count/$total]"
                    run_one "$data" "$model" "$norm" "$pred_len" 96 2023 M
                done
            done
        done
    done
    echo "[Table 2] 完成 $total 个实验。"
}

# ============================================================================
# Table 3: RevIN vs Dish-TS comparison (Autoformer backbone, 3 seeds)
# seq_len = 96, pred_len ∈ {24, 168, 336}, seeds ∈ {2023, 2024, 2025}
# ============================================================================
table3() {
    echo ""
    echo "=================================================================="
    echo "  Table 3: RevIN vs Dish-TS (Autoformer, 3 seeds, pred_len ∈ {24,168,336})"
    echo "=================================================================="
    local datasets=(ECL ETTh1 ETTm2 WTH)
    local norms=(revin dishts)
    local pred_lens=(24 168 336)
    local seeds=(2023 2024 2025)
    local total=$(( ${#datasets[@]} * ${#norms[@]} * ${#pred_lens[@]} * ${#seeds[@]} ))
    local count=0
    for data in "${datasets[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            for norm in "${norms[@]}"; do
                for seed in "${seeds[@]}"; do
                    count=$((count + 1))
                    echo "[$count/$total]"
                    run_one "$data" Autoformer "$norm" "$pred_len" 96 "$seed" M
                done
            done
        done
    done
    echo "[Table 3] 完成 $total 个实验。"
}

# ============================================================================
# Table 4: Impact of Larger Horizons (long TSF)
# seq_len = 96, pred_len ∈ {336, 420, 540, 600, 720}
# ============================================================================
table4() {
    echo ""
    echo "=================================================================="
    echo "  Table 4: Larger Horizons (seq_len=96, pred_len ∈ {336,420,540,600,720})"
    echo "  (paper uses N-BEATS; we use Autoformer as available backbone)"
    echo "=================================================================="
    local datasets=(ECL ETTh1)
    local models=(Autoformer)
    local norms=(none dishts)
    local pred_lens=(336 420 540 600 720)
    local total=$(( ${#datasets[@]} * ${#models[@]} * ${#norms[@]} * ${#pred_lens[@]} ))
    local count=0
    for data in "${datasets[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            for model in "${models[@]}"; do
                for norm in "${norms[@]}"; do
                    count=$((count + 1))
                    echo "[$count/$total]"
                    run_one "$data" "$model" "$norm" "$pred_len" 96 2023 M
                done
            done
        done
    done
    echo "[Table 4] 完成 $total 个实验。"
}

# ============================================================================
# Quick smoke test: 1 dataset × 1 model × 3 norms × 2 pred_len
# ============================================================================
quick() {
    echo ""
    echo "=================================================================="
    echo "  Quick smoke test: ETTm2 × Autoformer × {none, revin, dishts} × pred=24,96"
    echo "=================================================================="
    for pred_len in 24 96; do
        for norm in none revin dishts; do
            run_one ETTm2 Autoformer "$norm" "$pred_len" 96 2023 M
        done
    done
    summarize
    echo "[Quick] 快速验证完成。"
}

# ============================================================================
# 结果汇总
# ============================================================================
summarize() {
    echo ""
    echo "=================================================================="
    echo "  结果汇总（从 logs/ 提取所有 MSE/MAE）"
    echo "=================================================================="
    python - <<'PYEOF'
import os, re, glob, csv
log_dir = "./logs"
rows = []
for fpath in sorted(glob.glob(os.path.join(log_dir, "*.log"))):
    with open(fpath, "r", errors="replace") as f:
        text = f.read()
    m = re.search(r"DATA=(\S+).*MODEL=(\S+).*NORM=(\S+).*SEED=(\S+).*seq_len=(\d+).*label_len=(\d+).*pred_len=(\d+).*batch_size=(\d+).*alpha=(\S+)", text)
    if not m:
        continue
    data, model, norm, seed, seq_len, label_len, pred_len, bs, alpha = m.groups()
    mm = re.search(r"MSE=(\S+).*MAE=(\S+).*RMSE=(\S+).*MAPE=(\S+).*MSPE=(\S+)", text)
    if not mm:
        continue
    mse, mae, rmse, mape, mspe = mm.groups()
    rows.append({
        "data": data, "model": model, "norm": norm, "seed": seed,
        "seq_len": seq_len, "label_len": label_len, "pred_len": pred_len,
        "batch_size": bs, "alpha": alpha,
        "MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "MSPE": mspe,
        "log": os.path.basename(fpath)
    })

rows.sort(key=lambda r: (r["data"], r["model"], r["norm"], int(r["seq_len"]), int(r["pred_len"]), int(r["seed"])))

out_csv = "./results/paper_summary.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Total experiments: {len(rows)}")
print(f"CSV saved to: {out_csv}")
print("")
print("=== 每个数据集 × 模型 × norm 的 pred_len 汇总 ===")
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    key = (r["data"], r["model"], r["norm"])
    agg[key].append((int(r["pred_len"]), r["MSE"], r["MAE"], r["seed"]))
for key in sorted(agg.keys()):
    vals = sorted(agg[key])
    print(f"{key[0]:8s} {key[1]:12s} {key[2]:8s} -> " + "; ".join([f"p{p}:MSE={mse}" for p, mse, _, _ in vals[:5]]))
PYEOF
}

# ============================================================================
# 入口
# ============================================================================
case "${1:-all}" in
    table1)     table1; summarize ;;
    table2)     table2; summarize ;;
    table3)     table3; summarize ;;
    table4)     table4; summarize ;;
    quick)      quick ;;
    summarize)  summarize ;;
    all)
        table1
        table2
        table3
        table4
        summarize
        ;;
    *)
        echo "用法: $0 {table1|table2|table3|table4|quick|summarize|all}"
        echo ""
        echo "  table1   — Table 1: 单变量, seq_len=pred_len"
        echo "  table2   — Table 2: 多变量, seq_len=96"
        echo "  table3   — Table 3: RevIN vs Dish-TS, 3 seeds"
        echo "  table4   — Table 4: 长窗口 (336~720)"
        echo "  quick    — 快速验证 (9 个实验)"
        echo "  summarize — 只汇总已有结果"
        echo "  all      — 运行 4 个 tables + 汇总"
        exit 1
        ;;
esac

echo ""
echo "[DONE] 全部结束。"
