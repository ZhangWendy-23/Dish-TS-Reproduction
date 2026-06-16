#!/bin/bash
# Phase 2a - ETTm2 deep alpha sweep (5 alphas x 3 preds x 3 seeds = 45 jobs)
# We already have alpha=0.0, 0.5, 1.0 from Phase 1.
# This script runs the missing alpha=0.25 and alpha=0.75.

set -u
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
DATA="ETTm2"
MODEL="Autoformer"
NORM="dishts"
PRIOR="paper-phi-only"
DI="avg"
PREDS=(96 168 336)
ALPHAS=(0.25 0.75)
SEEDS=(2023 2024 2025)

mkdir -p logs results

njobs=$(( ${#PREDS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} ))
echo "[Phase2a] ETTm2 paper-phi-only: 3 pred_len x 2 missing alphas x 3 seeds = $njobs NEW jobs"
echo "[Phase2a] Already have alpha=0.0, 0.5, 1.0 from Phase 1"
echo

for H in "${PREDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            log="logs/fig3_${DATA}_${MODEL}_${PRIOR//-/_}_PL${H}_alpha${alpha}_seed${seed}.log"
            echo "[Phase2a] pred_len=$H alpha=$alpha seed=$seed"
            python3 -u train.py \
                --data "$DATA" --model "$MODEL" --norm "$NORM" \
                --alpha "$alpha" --prior "$PRIOR" --dish_init "$DI" \
                --seq_len "$H" --pred_len "$H" --label_len 48 \
                --batch_size 128 --lr 1e-3 --train_epochs 100 --patience 10 \
                --features M --seed "$seed" --gpu "$GPU" --figure3 \
                > "$log" 2>&1
        done
    done
done

echo
echo "[Phase2a] All jobs done. Now run quality checks:"
echo "  python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
echo "  python3 repro_figures/plot_figure3.py --input results/figure3_runs.csv --dataset ETTm2"
