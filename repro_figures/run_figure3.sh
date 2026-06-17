#!/usr/bin/env bash
#
# Paper Figure 3 — alpha (prior loss weight) ablation.
#
# Values: alpha ∈ {0.0, 0.25, 0.5, 1.0}.  The paper shows:
#   - Short horizons (≤168): curves are nearly flat (alpha doesn't matter).
#   - Long horizons (>168): alpha=0 error sharply rises while alpha>0 stays low.
#
# IMPORTANT — reproducing this figure is NOT the same thing as reproducing
# the paper's main comparison tables (Table 1/2/3/4/5/6).  The alpha prior
# uses the TRUE FUTURE window mean as a soft supervision signal for the
# HORICONET coefficient φ_h, which is a form of look-ahead data leakage.
# The paper authors themselves removed the alpha term from their open-source
# implementation; the baseline Dish-TS (Tables 1–6) always trains with
# alpha=0 and no prior term.
#
# Use this script ONLY when you want to:
#   (a) draw a Figure-3-style alpha-sweep plot;
#   (b) demonstrate that the alpha term HURTS generalisation under
#       standard inference rules (no access to future data).
# For everything else use repro_figures/run_table2.sh (alpha=0, prior=none).
#
# We use Autoformer backbone (the paper's default for this figure),
# with the paper's MSE-vs-horizon shape.  L=H ∈ {24, 96, 168, 336} to cover
# both sides of the ~168 inflection point.
#
# Total jobs: 4 datasets × 4 horizons × 4 alphas × 3 seeds = 192.
#
# Usage:  bash repro_figures/run_figure3.sh
#         DATASET_SUBSET="ETTm2" GPU=0 bash repro_figures/run_figure3.sh

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(24 96 168 336)
ALPHAS=(0.0 0.25 0.5 1.0)
DATASETS=(${DATASET_SUBSET:-ECL ETTh1 ETTm2 WTH})

mkdir -p logs results

for DATA in "${DATASETS[@]}"; do
    [[ "$DATA" == "ECL" ]] && bs=64 || bs=128
    for H in "${PREDS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                log="logs/fig3_${DATA}_Autoformer_dishts_L${H}_H${H}_alpha${alpha}_seed${seed}.log"
                echo "[fig3] data=$DATA H=$H alpha=$alpha seed=$seed bs=$bs -> $log"
                python3 -u train.py \
                    --data "$DATA" --model Autoformer --norm dishts \
                    --alpha "$alpha" --prior paper-phi-only --dish_init avg \
                    --seq_len "$H" --pred_len "$H" --label_len 48 \
                    --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 10 \
                    --features M --seed "$seed" --gpu "$GPU" --figure3 \
                    >"$log" 2>&1
            done
        done
    done
done

echo
echo "Figure 3 sweep done."
echo "  Inspect: python3 repro_figures/compare_paper.py --apply-paper-scale multivariate"
