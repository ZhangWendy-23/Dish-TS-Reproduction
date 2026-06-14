#!/usr/bin/env bash
#
# Paper Table 5 — Lookback length ablation.
#
# Fixed horizon H = 48.  Lookback L ∈ {48, 96, 144, 192, 240}.
# Paper backbone: N-BEATS.  Paper datasets: Electricity, ETTh1.
#
# Runtime estimate (per dataset): ~2 min/job × 5 lookbacks × 3 seeds = ~30 min.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
LOOKBACKS=(48 96 144 192 240)
DATASETS=(ETTh1 ECL)
NORMS=(none dishts)

mkdir -p logs results

for DATA in "${DATASETS[@]}"; do
    for L in "${LOOKBACKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for norm in "${NORMS[@]}"; do
                log="logs/t5_${DATA}_NBEATS_${norm}_L${L}_H48_seed${seed}.log"
                echo "[t5] DATA=$DATA L=$L H=48 norm=$norm seed=$seed -> $log"
                python3 -u train.py \
                    --data "$DATA" --model NBEATS --norm "$norm" \
                    --seq_len "$L" --pred_len 48 --label_len 48 \
                    --batch_size 256 --lr 1e-3 --train_epochs 100 --patience 10 \
                    --alpha 0.0 --prior none --features M \
                    --seed "$seed" --gpu "$GPU" --figure3 \
                    >"$log" 2>&1
            done
        done
    done
done

echo
echo "Table 5 sweep done.  Run: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
