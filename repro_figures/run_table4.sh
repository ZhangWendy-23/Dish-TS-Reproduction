#!/usr/bin/env bash
#
# Paper Table 4 — Long horizon forecasting ablation.
#
# Fixed lookback L = 96.  Predicted horizon H ∈ {336, 420, 540, 600, 720}.
# Paper backbone: N-BEATS.  Paper datasets: Electricity, ETTh1.
#
# Runtime estimate (per dataset): ~3 min/job × 5 preds × 3 seeds = ~45 min.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(336 420 540 600 720)
DATASETS=(ETTh1 ECL)  # ECL ≡ Electricity

# sweep both plain-norm and Dish-TS for easy comparison; RevIN is optional.
NORMS=(none dishts)

mkdir -p logs results

for DATA in "${DATASETS[@]}"; do
    for H in "${PREDS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for norm in "${NORMS[@]}"; do
                log="logs/t4_${DATA}_NBEATS_${norm}_L96_H${H}_seed${seed}.log"
                echo "[t4] DATA=$DATA L=96 H=$H norm=$norm seed=$seed -> $log"
                python3 -u train.py \
                    --data "$DATA" --model NBEATS --norm "$norm" \
                    --seq_len 96 --pred_len "$H" --label_len 48 \
                    --batch_size 256 --lr 1e-3 --train_epochs 100 --patience 10 \
                    --alpha 0.0 --prior none --features M \
                    --seed "$seed" --gpu "$GPU" --figure3 \
                    >"$log" 2>&1
            done
        done
    done
done

echo
echo "Table 4 sweep done.  Run: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
