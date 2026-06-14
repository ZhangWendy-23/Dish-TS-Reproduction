#!/usr/bin/env bash
#
# Paper Tables 2 & 3 — multivariate forecasting (L=H).
#
# Table 2 reports Informer / Autoformer / N-BEATS with and without Dish-TS.
# Table 3 reports RevIN vs Dish-TS on Autoformer backbone with MSE only.
#
# Datasets: Electricity (= ECL), ETTh1, ETTm2, Weather (4 datasets).
# L=H ∈ {24, 48, 96, 168, 336} (5 horizons).
# Backbones: Informer, Autoformer, N-BEATS.
# Normalization methods: none (plain), revin, dishts.
# Seeds: 2023, 2024, 2025.
#
# Total jobs per norm: 4 datasets × 5 horizons × 3 backbones × 3 seeds
#                    = 180  (per norm); so 3 norms × 180 = 540.
#
# Typical run time: ~5 min/job → ~45 h total on a single 3090.
# You can parallelise across GPUs by setting GPU=0/1/... and restricting
# DATA / NORM env vars.
#
#   # example 1: ECL only with Informer + Dish-TS, GPU 0
#   DATA=ECL MODEL=Informer NORMS="none dishts" GPU=0 bash repro_figures/run_table3.sh
#
#   # example 2: everything but ECL on GPU 1
#   DATASET_SUBSET="ETTh1 ETTm2 Weather" GPU=1 bash repro_figures/run_table3.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(24 48 96 168 336)
MODELS=("Informer" "Autoformer" "N-BEATS")
NORMS=(${NORMS:-"none revin dishts"})

# `--seq_len 0` is the repo convention for "auto-set to pred_len" so that
# the experiment automatically respects the paper's L=H rule.
SEQ_LEN=0

DATASETS=(${DATASET_SUBSET:-"ECL ETTh1 ETTm2 Weather"})

mkdir -p logs results

total=0
for DATA in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for norm in "${NORMS[@]}"; do
            for pred_len in "${PREDS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    total=$((total + 1))
                done
            done
        done
    done
done

i=0
for DATA in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        # Informer's default batch_size is 256 in our repo; ECL uses 64.
        if [[ "$DATA" == "ECL" ]]; then
            bs=64
        elif [[ "$MODEL" == "Informer" ]]; then
            bs=256
        else
            bs=128
        fi

        for norm in "${NORMS[@]}"; do
            for pred_len in "${PREDS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    i=$((i + 1))
                    log="logs/table2_${DATA}_${MODEL}_${norm}_L${pred_len}_H${pred_len}_seed${seed}.log"
                    echo "[$i/$total] data=$DATA model=$MODEL norm=$norm L=H=$pred_len seed=$seed bs=$bs -> $log"
                    python3 -u train.py \
                        --data "$DATA" --model "$MODEL" --norm "$norm" \
                        --seq_len "$SEQ_LEN" --pred_len "$pred_len" \
                        --batch_size "$bs" --lr 1e-3 \
                        --train_epochs 100 --patience 7 \
                        --features M --seed "$seed" --gpu "$GPU" \
                        --figure3 \
                        >"$log" 2>&1
                done
            done
        done
    done
done

echo
echo "Done. Now compare with paper reference numbers:"
echo "  python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
