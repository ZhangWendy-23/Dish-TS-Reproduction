#!/usr/bin/env bash
#
# Paper Table 6 — CONET initialization ablation.
#
# We compare three initialisation strategies for Dish-TS's CONET weights:
#   - avg   (constant 1.0)
#   - norm  (standard normal)
#   - uni   (uniform [0,1])
#
# The comparison is done on ETTh1 and Weather datasets, at three horizons
# (24, 96, 168), with both Autoformer and N-BEATS as backbones.
#
# Typical total size: 2 datasets × 3 horizons × 2 backbones × 3 inits × 3 seeds
#                    = 108 jobs.
#
# Runtime estimate (single 3090): ~5 min/job → ~9 hours.
#
# Usage:  bash repro_figures/run_table6.sh
#         DATASET_SUBSET="ETTh1" GPU=0 bash repro_figures/run_table6.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(24 96 168)
MODELS=(Autoformer NBEATS)
DATASETS=(${DATASET_SUBSET:-"ETTh1 WTH"})

mkdir -p logs results

for DATA in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        [[ "$MODEL" == "NBEATS" ]] && bs=256 || bs=128
        [[ "$DATA" == "ECL" ]] && bs=64

        for pred_len in "${PREDS[@]}"; do
            for init in avg norm uni; do
                for seed in "${SEEDS[@]}"; do
                    log="logs/t6_${DATA}_${MODEL}_dishts_L${pred_len}_H${pred_len}_init_${init}_seed${seed}.log"
                    echo "[t6] data=$DATA model=$model H=$pred_len init=$init seed=$seed bs=$bs -> $log"
                    python3 -u train.py \
                        --data "$DATA" --model "$MODEL" --norm dishts \
                        --dish_init "$init" \
                        --seq_len "$pred_len" --pred_len "$pred_len" --label_len 48 \
                        --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 10 \
                        --alpha 0.0 --prior none --features M \
                        --seed "$seed" --gpu "$GPU" --figure3 \
                        >"$log" 2>&1
                done
            done
        done
    done
done

echo
echo "Table 6 sweep done.  Run: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
