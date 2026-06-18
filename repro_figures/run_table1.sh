#!/usr/bin/env bash
#
# Paper Table 1 — Univariate forecasting (L=H).
#
# Recipe (per paper):
#   * L = H ∈ {24, 96, 168, 336}
#   * backbones: Informer, Autoformer, N-BEATS
#   * features: S (univariate)
#   * normalisation: none / revin / dishts
#   * alpha / prior: OFF (alpha=0, --prior none)
#
# Total jobs: 4 datasets × 3 backbones × 3 norms × 4 horizons × 3 seeds
#           = 432 jobs.
# Single 3090 runtime estimate: ~3-5 min/job → ~25-40 h.
#
# Usage (on 3090):
#   cd ~/Dish-TS
#   git pull
#   bash repro_figures/run_table1.sh
#   # Or restrict to one dataset / one backbone for quick test:
#   DATASET_SUBSET="ETTm2" MODEL=Autoformer bash repro_figures/run_table1.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
MODELS=(${MODEL:-Informer Autoformer NBEATS})
NORMS=(none revin dishts)
PREDS=(24 96 168 336)
DATASETS=(${DATASET_SUBSET:-ECL ETTh1 ETTm2 WTH})

mkdir -p logs results

total=$(( ${#DATASETS[@]} * ${#MODELS[@]} * ${#NORMS[@]} * ${#PREDS[@]} * ${#SEEDS[@]} ))
echo "[run_table1] datasets=${DATASETS[*]}"
echo "[run_table1] models=${MODELS[*]}"
echo "[run_table1] norms=${NORMS[*]}"
echo "[run_table1] horizons=${PREDS[*]}"
echo "[run_table1] seeds=${SEEDS[*]}"
echo "[run_table1] TOTAL=$total jobs"

for DATA in "${DATASETS[@]}"; do
    [[ "$DATA" == "ECL" ]] && bs=64 || bs=128
    for MODEL in "${MODELS[@]}"; do
        [[ "$MODEL" == "NBEATS" ]] && bs=256
        for NORM in "${NORMS[@]}"; do
            for H in "${PREDS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    log="logs/table1_${DATA}_${MODEL}_${NORM}_L${H}_H${H}_seed${seed}.log"
                    echo "[table1] data=$DATA model=$MODEL norm=$NORM H=$H seed=$seed bs=$bs -> $log"
                    python3 -u train.py \
                        --data "$DATA" --model "$MODEL" --norm "$NORM" \
                        --alpha 0.0 --prior none --dish_init avg \
                        --seq_len "$H" --pred_len "$H" --label_len 48 \
                        --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 10 \
                        --features S --seed "$seed" --gpu "$GPU" --figure3 \
                        > "$log" 2>&1
                done
            done
        done
    done
done

echo
echo "Table 1 (Univariate) sweep done."
echo "  Post-process: python3 repro_figures/compare_paper.py --apply-paper-scale univariate"
