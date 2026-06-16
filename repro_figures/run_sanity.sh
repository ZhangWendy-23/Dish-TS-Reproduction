#!/usr/bin/env bash
#
# Phase 1 — single-dataset sanity check on the CURRENT code, BEFORE the
# full Figure 3 sweep and the full comparison tables.
#
# Target grid (3090, ~30–60 minutes single card):
#   dataset:  ETTm2
#   backbone: Autoformer
#   norm:     dishts (paper-phi-only)
#   horizons: 96, 168, 336
#   alphas:   0.0, 0.5, 1.0
#   seeds:    2023, 2024, 2025
# Total: 1 dataset x 3 horizons x 3 alphas x 3 seeds = 27 jobs.
#
# We keep the grid small on purpose — this is a gate check: if alpha=0.0
# is not the best (or very close to it) after these 27 runs, something
# in the current code has regressed vs. the earlier local experiments
# and we should NOT start the multi-hour Phase 2 sweep.
#
# Usage (on 3090):
#   cd ~/Dish-TS
#   git pull
#   screen -U -S sanity                              # detachable session
#   bash repro_figures/run_sanity.sh
#
# Inspect results while running:
#   tail -f logs/sanity_ETTm2_Autoformer_paper-phi-only_PL96_alpha0.0_seed2023.log
#   python3 repro_figures/compare_paper.py --apply-paper-scale multivariate

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(96 168 336)
ALPHAS=(0.0 0.5 1.0)
DATA="ETTm2"
MODEL="Autoformer"
NORM="dishts"
PRIOR="paper-phi-only"

mkdir -p logs results

njobs=$(( ${#PREDS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} ))
echo "[sanity] dataset=$DATA  model=$MODEL  norm=$NORM  prior=$PRIOR"
echo "[sanity] horizons=${PREDS[*]}   alphas=${ALPHAS[*]}   seeds=${SEEDS[*]}"
echo "[sanity] TOTAL=$njobs jobs (should take ~30-60 min on 1x3090)"
echo

for H in "${PREDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            log="logs/sanity_${DATA}_${MODEL}_${PRIOR//-/_}_PL${H}_alpha${alpha}_seed${seed}.log"
            echo "[sanity] H=$H alpha=$alpha seed=$seed -> $log"
            python3 -u train.py \
                --data "$DATA" --model "$MODEL" --norm "$NORM" \
                --alpha "$alpha" --prior "$PRIOR" --dish_init avg \
                --seq_len "$H" --pred_len "$H" --label_len 48 \
                --batch_size 128 --lr 1e-3 --train_epochs 100 --patience 7 \
                --features M --seed "$seed" --gpu "$GPU" --figure3 \
                > "$log" 2>&1
        done
    done
done

echo
echo "Phase 1 sanity-check sweep done."
echo "  Post-process and compare: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
echo "  Then decide whether to proceed to Phase 2 (full Fig 3) and Phase 3 (baselines)."
