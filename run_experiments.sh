#!/bin/bash
# ============================================================================
# run_experiments.sh — Dish-TS 论文完整复现脚本
# ============================================================================
# 对照论文 4 个 Table 的参数设置：
#
#   Table 1 — Univariate:   seq_len = pred_len ∈ {24, 48, 96, 168, 336}
#   Table 2 — Multivariate: seq_len = 96, pred_len ∈ {24, 48, 96, 168, 336}
#   Table 3 — RevIN对比:    seq_len = 96, pred_len ∈ {24, 168, 336}, Autoformer
#   Table 4 — LongerHorizon: seq_len = 96, pred_len ∈ {336, 420, 540, 600, 720}
#
# 用法：
#   chmod +x run_experiments.sh
#   ./run_experiments.sh table1       # 单变量实验
#   ./run_experiments.sh table2       # 多变量实验
#   ./run_experiments.sh table3       # RevIN vs Dish-TS 对比
#   ./run_experiments.sh table4       # 长窗口预测
#   ./run_experiments.sh ablation     # dish_init 消融
#   ./run_experiments.sh all          # 运行全部（非常慢！）
#   ./run_experiments.sh all_paper    # 运行 Table 1-4（推荐）
# ============================================================================

set -e
GPU=0
SEED=2023
BASE_SEEDS=(2023 2024 2025)          # 论文用 3 个 seed 取均值±标准差
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" results

# ------------------------------------------------------------------
# 工具：执行一次训练
# 参数：data model norm pred_len seq_len [seed] [extra_args]
# ------------------------------------------------------------------
run_one() {
    local data="$1"
    local model="$2"
    local norm="$3"
    local pred_len="$4"
    local seq_len="$5"
    local seed="${6:-$SEED}"
    local extra="${7:-}"

    local key="${data}_${model}_${norm}_s${seq_len}_p${pred_len}_seed${seed}"
    local LOG="$LOG_DIR/${key}.log"

    # 如果已有完整结果（包含 MSE 行），跳过
    if [ -f "$LOG" ] && grep -q "^0.*$data.*$model" "$LOG" 2>/dev/null; then
        echo "  [SKIP] $model + $norm, seq=$seq_len pred=$pred_len (seed=$seed) — 已有结果"
        return 0
    fi

    echo "[$(date +%H:%M:%S)] $data | $model + $norm | seq=$seq_len pred=$pred_len | seed=$seed"
    python train.py \
        --data "$data" \
        --model "$model" \
        --norm "$norm" \
        --pred_len "$pred_len" \
        --seq_len "$seq_len" \
        --seed "$seed" \
        --gpu "$GPU" \
        $extra 2>&1 | tee "$LOG"
}

# ============================================================================
# Table 1 — Univariate（单变量，lookback = horizon）
# ============================================================================
# Length ∈ {24, 48, 96, 168, 336}，seq_len = pred_len
# 模型：Informer, Autoformer, Transformer（论文还包含 N-BEATS，本仓库未实现）
# 数据集：ECL, ETTh1, ETTm2, WTH
# ============================================================================
table1() {
    echo ""
    echo "=================================================================="
    echo "  Table 1: Univariate Time Series Forecasting"
    echo "  (seq_len = pred_len ∈ {24, 48, 96, 168, 336})"
    echo "=================================================================="
    for data in ECL ETTh1 ETTm2 WTH; do
        for model in Informer Autoformer Transformer; do
            for norm in none dishts revin; do
                for pred_len in 24 48 96 168 336; do
                    run_one "$data" "$model" "$norm" "$pred_len" "$pred_len"
                done
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Table 1 完成。"
}

# ============================================================================
# Table 2 — Multivariate（多变量，lookback 固定 96）
# ============================================================================
# seq_len = 96，pred_len ∈ {24, 48, 96, 168, 336}
# 模型：Informer, Autoformer, Transformer（论文还包含 N-BEATS*）
# 数据集：ECL, ETTh1, ETTm2, WTH
# ============================================================================
table2() {
    echo ""
    echo "=================================================================="
    echo "  Table 2: Multivariate Time Series Forecasting"
    echo "  (seq_len = 96, pred_len ∈ {24, 48, 96, 168, 336})"
    echo "=================================================================="
    for data in ECL ETTh1 ETTm2 WTH; do
        for model in Informer Autoformer Transformer; do
            for norm in none dishts revin; do
                for pred_len in 24 48 96 168 336; do
                    run_one "$data" "$model" "$norm" "$pred_len" 96
                done
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Table 2 完成。"
}

# ============================================================================
# Table 3 — RevIN vs Dish-TS（Autoformer backbone，多变量）
# ============================================================================
# seq_len = 96，pred_len ∈ {24, 168, 336}
# 模型：Autoformer（多 seed 取均值±标准差）
# 数据集：ECL, ETTh1, ETTm2, WTH
# ============================================================================
table3() {
    echo ""
    echo "=================================================================="
    echo "  Table 3: RevIN vs Dish-TS (Autoformer backbone, multivariate)"
    echo "  (seq_len = 96, pred_len ∈ {24, 168, 336}, 3 seeds)"
    echo "=================================================================="
    for data in ECL ETTh1 ETTm2 WTH; do
        for norm in revin dishts; do
            for pred_len in 24 168 336; do
                for seed in "${BASE_SEEDS[@]}"; do
                    run_one "$data" Autoformer "$norm" "$pred_len" 96 "$seed"
                done
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Table 3 完成。"
}

# ============================================================================
# Table 4 — Larger Horizon（长窗口，N-BEATS 或 Autoformer 替代）
# ============================================================================
# seq_len = 96，pred_len ∈ {336, 420, 540, 600, 720}
# 模型：论文用 N-BEATS，本仓库用 Autoformer 作为可用替代
# 数据集：ECL, ETTh1
# ============================================================================
table4() {
    echo ""
    echo "=================================================================="
    echo "  Table 4: Impact of Larger Horizons (Long TSF)"
    echo "  (seq_len = 96, pred_len ∈ {336, 420, 540, 600, 720})"
    echo "  (Note: paper uses N-BEATS; we use Autoformer as available backbone)"
    echo "=================================================================="
    for data in ECL ETTh1; do
        for model in Autoformer; do
            for norm in none dishts; do
                for pred_len in 336 420 540 600 720; do
                    run_one "$data" "$model" "$norm" "$pred_len" 96
                done
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Table 4 完成。"
}

# ============================================================================
# 消融实验：Dish-TS 初始化方式（standard / avg / uniform）
# ============================================================================
ablation() {
    echo ""
    echo "=================================================================="
    echo "  Ablation: DishTS initialization methods"
    echo "  (standard vs avg vs uniform, Autoformer + ETTm2)"
    echo "=================================================================="
    for init in standard avg uniform; do
        for pred_len in 24 48 96 168 336; do
            run_one ETTm2 Autoformer dishts "$pred_len" 96 $SEED "--dish_init $init"
        done
    done
    echo "[$(date +%H:%M:%S)] Ablation 完成。"
}

# ============================================================================
# 多 seed：对 Table 2 核心配置（ETTm2, Autoformer）跑 3 个 seed，取均值±标准差
# ============================================================================
multi_seed() {
    echo ""
    echo "=================================================================="
    echo "  Multi-seed: ETTm2 + Autoformer, 3 seeds, 3 norms × 5 pred_lens"
    echo "=================================================================="
    for seed in "${BASE_SEEDS[@]}"; do
        for norm in none revin dishts; do
            for pred_len in 24 48 96 168 336; do
                run_one ETTm2 Autoformer "$norm" "$pred_len" 96 "$seed"
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Multi-seed 完成。"
}

# ============================================================================
# 结果汇总 — 从 logs 提取所有 MSE / MAE，输出 CSV
# ============================================================================
summarize() {
    echo ""
    echo "=================================================================="
    echo "  Results Summary (extracting from logs/)"
    echo "=================================================================="
    python results/collect_results.py
    echo ""
    echo "→ 结果已写入 results/summary.csv"
}

# ============================================================================
# 调度
# ============================================================================
case "${1:-}" in
    table1)      table1     ;;
    table2)      table2     ;;
    table3)      table3     ;;
    table4)      table4     ;;
    ablation)    ablation   ;;
    multi_seed)  multi_seed ;;
    summarize)   summarize  ;;
    all_paper)
        table1
        table2
        table3
        table4
        summarize
        ;;
    all)
        table1
        table2
        table3
        table4
        ablation
        multi_seed
        summarize
        ;;
    *)
        echo "用法: $0 {table1|table2|table3|table4|ablation|multi_seed|summarize|all_paper|all}"
        echo ""
        echo "  table1      — Table 1: 单变量, seq_len=pred_len ∈ {24,48,96,168,336}"
        echo "  table2      — Table 2: 多变量, seq_len=96, pred_len ∈ {24,48,96,168,336}"
        echo "  table3      — Table 3: RevIN vs Dish-TS, Autoformer, 3 seeds"
        echo "  table4      — Table 4: 长窗口, pred_len ∈ {336,420,540,600,720}"
        echo "  ablation    — DishTS 初始化方式对比 (standard/avg/uniform)"
        echo "  multi_seed  — 多 seed 实验 (ETTm2, Autoformer, 3 seeds)"
        echo "  summarize   — 汇总所有日志到 results/summary.csv"
        echo "  all_paper   — 运行 Table 1~4（论文主要实验）"
        echo "  all         — 运行全部（含消融和多 seed）"
        exit 1
        ;;
esac
