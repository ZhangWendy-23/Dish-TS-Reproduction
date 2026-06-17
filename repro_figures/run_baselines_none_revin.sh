#!/usr/bin/env bash
#
# Phase 3 — none / revin baselines (alpha=0, prior=none).
#
# These are the TABLE 2 / 3 comparison baselines: plain Autoformer, without
# Dish-TS or RevIN normalisation, with the same data scale and the same
# hyper-parameters as the dishts runs of Phase 2.
#
# Grid:
#   datasets:     ECL, ETTh1, ETTm2, WTH        (overridable via DATASET_SUBSET)
#   backbone:     Autoformer                    (paper default for Tables 2/3)
#   norm:         none, revin
#   horizons:     24, 96, 168, 336
#   seeds:        2023, 2024, 2025
#   alpha / prior:0.0 / none                    (matching the OFFICIAL Dish-TS repo)
# Total: 4 datasets x 2 norms x 4 horizons x 3 seeds = 96 jobs.
#        At ~5 min/job on a 3090 this is ~8 hours.
#
# All results are APPENDED to results/figure3_runs.csv (one row per job),
# and the reported alpha column is clamped to 0.0 for norm != 'dishts',
# so the baselines are naturally separable from the Phase 2 alpha-sweep
# rows.
#
# Usage (on 3090, from a detached screen):
#   cd ~/Dish-TS
#   git pull
#   screen -U -S baselines
#   bash repro_figures/run_baselines_none_revin.sh
#
# Run only a single dataset (useful for overnight):
#   DATASET_SUBSET="ETTm2" bash repro_figures/run_baselines_none_revin.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
NORMS=(none revin)
PREDS=(24 96 168 336)
DATASETS=(${DATASET_SUBSET:-ECL ETTh1 ETTm2 WTH})

mkdir -p logs results

njobs=0
for _ in "${DATASETS[@]}";  do njobs=$((njobs + 1)); done
total=$((njobs * ${#NORMS[@]} * ${#PREDS[@]} * ${#SEEDS[@]}))
echo "[baselines] datasets=${DATASETS[*]}"
echo "[baselines] norms=${NORMS[*]}    horizons=${PREDS[*]}   seeds=${SEEDS[*]}"
echo "[baselines] TOTAL=$total jobs (~8 h on 1x3090; ~5 min/job)"
echo

for DATA in "${DATASETS[@]}"; do
    # ECL / WTH use a smaller batch size; ETT datasets are fine at 128.
    [[ "$DATA" == "ECL" || "$DATA" == "WTH" ]] && bs=64 || bs=128
    for NORM in "${NORMS[@]}"; do
        for H in "${PREDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                log="logs/baseline_${DATA}_Autoformer_${NORM}_L${H}_H${H}_seed${seed}.log"
                echo "[baselines] data=$DATA norm=$NORM H=$H seed=$seed bs=$bs -> $log"
                python3 -u train.py \
                    --data "$DATA" --model Autoformer --norm "$NORM" \
                    --alpha 0.0 --prior none --dish_init avg \
                    --seq_len "$H" --pred_len "$H" --label_len 48 \
                    --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 7 \
                    --features M --seed "$seed" --gpu "$GPU" --figure3 \
                    > "$log" 2>&1
            done
        done
    done
done

echo
echo "Phase 3 baselines sweep done."
echo "  Post-process: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
echo "  Plot:         python3 repro_figures/plot_figure3.py --input results/figure3_runs.csv"
