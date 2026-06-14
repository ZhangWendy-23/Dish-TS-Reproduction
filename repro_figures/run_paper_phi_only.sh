#!/usr/bin/env bash
#
# ETTm2 × 4 pred_len × 4 alpha × 3 seeds × (prior=paper-phi-only).
#
# This is a focused matrix designed to reproduce the Dish-TS paper's
# alpha-vs-MSE curve exactly.  The key experimental decision is:
#   --prior paper-phi-only   (prior gradient reaches HoriCoNet only;
#                             BackCoNet is trained purely by the base MSE)
#
# Total jobs: 4 * 4 * 3 = 48.
# Runtime estimate:  ~4 min/job × 48 = ~3.2 h on a single 3090.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DATA=ETTm2
MODEL=Autoformer
NORM=dishts
PRIOR=paper-phi-only
DISH_INIT=avg
SEQ_LEN=96
BATCH_SIZE=128
TRAIN_EPOCHS=50
PATIENCE=7
GPU=0

mkdir -p logs results

PREDS=(24 96 168 336)
ALPHAS=(0.0 0.25 0.5 1.0)
SEEDS=(2023 2024 2025)

total=0
for p in "${PREDS[@]}"; do
    for a in "${ALPHAS[@]}"; do
        for s in "${SEEDS[@]}"; do
            total=$((total + 1))
        done
    done
done

i=0
for p in "${PREDS[@]}"; do
    for a in "${ALPHAS[@]}"; do
        for s in "${SEEDS[@]}"; do
            i=$((i + 1))
            log="logs/${DATA}_${MODEL}_${NORM}_s${SEQ_LEN}_p${p}_seed${s}_alpha${a}_prior${PRIOR}_raw.log"
            echo "[$i/$total] pred_len=$p alpha=$a seed=$s  -> $log"
            python3 -u train.py \
                --data "$DATA" --model "$MODEL" --norm "$NORM" \
                --alpha "$a" --prior "$PRIOR" --dish_init "$DISH_INIT" \
                --seq_len "$SEQ_LEN" --pred_len "$p" \
                --label_len $((p > 48 ? 48 : p)) \
                --batch_size "$BATCH_SIZE" \
                --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
                --seed "$s" --gpu "$GPU" \
                --figure3 \
                >"$log" 2>&1
        done
    done
done

echo "All 48 jobs done.  Now run: python3 repro_figures/compare_paper.py --prior paper-phi-only"
