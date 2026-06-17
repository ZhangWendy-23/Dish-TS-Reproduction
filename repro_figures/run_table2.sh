#!/usr/bin/env bash
#
# Paper Tables 2 & 3 — the core reproducing experiments.
#
# Recipe (per paper, and matching the official Dish-TS open-source repo):
#   * L = H ∈ {24, 96, 168, 336}
#   * backbones: Informer, Autoformer, N-BEATS
#   * normalisation: none / revin / dishts
#   * alpha / prior: OFF (alpha=0, --prior none).  The alpha-ablation
#     prior discussed in Figure 3 is kept for Figure 3 ONLY; every
#     comparison table in the paper uses the plain Dish-TS structure
#     without a weighted prior term.  Using alpha>0 here would
#     introduce a training-vs-inference distribution gap and make
#     numbers not comparable with the paper.
#
# Typical scale: 4 datasets x 3 models x 3 norms x 4 horizons x 3 seeds
#              = 432 jobs (single seed would be 144).
# With ~5 min/job this is ~36 hours per seed on a single 3090.
#
# Usage (on 3090):
#   cd ~/Dish-TS
#   git pull                                 # always pick up latest scripts
#   DATASET_SUBSET="ECL ETTh1 ETTm2 WTH" bash repro_figures/run_table2.sh
# or just one dataset for test:
#   DATASET_SUBSET="ETTm2" bash repro_figures/run_table2.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
MODELS=(Informer Autoformer NBEATS)
NORMS=(none revin dishts)
PREDS=(24 96 168 336)
DATASETS=(${DATASET_SUBSET:-ECL ETTh1 ETTm2 WTH})

mkdir -p logs results

njobs=0
for D in "${DATASETS[@]}";          do njobs=$((njobs + 1)); done
total=$((njobs * ${#MODELS[@]} * ${#NORMS[@]} * ${#PREDS[@]} * ${#SEEDS[@]}))
echo "[run_table2] datasets=${DATASETS[*]}"
echo "[run_table2] models=${MODELS[*]}"
echo "[run_table2] norms=${NORMS[*]}"
echo "[run_table2] horizons=${PREDS[*]}"
echo "[run_table2] seeds=${SEEDS[*]}"
echo "[run_table2] TOTAL=$total jobs"

for DATA in "${DATASETS[@]}"; do
    [[ "$DATA" == "ECL" ]] && bs=64 || bs=128
    for MODEL in "${MODELS[@]}"; do
        for NORM in "${NORMS[@]}"; do
            for H in "${PREDS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    log="logs/table2_${DATA}_${MODEL}_${NORM}_L${H}_H${H}_seed${seed}.log"
                    echo "[table2] data=$DATA model=$MODEL norm=$NORM H=$H seed=$seed bs=$bs -> $log"
                    python3 -u train.py \
                        --data "$DATA" --model "$MODEL" --norm "$NORM" \
                        --alpha 0.0 --prior none --dish_init avg \
                        --seq_len "$H" --pred_len "$H" --label_len 48 \
                        --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 10 \
                        --features M --seed "$seed" --gpu "$GPU" --figure3 \
                        > "$log" 2>&1
                done
            done
        done
    done
done

echo
echo "Table 2 / 3 sweep done."
echo "  Post-process: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
