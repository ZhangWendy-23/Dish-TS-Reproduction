#!/usr/bin/env bash
#
# Paper Table 3 "RevIN vs Dish-TS" full sweep.
#
# Matrix:   4 datasets × (plain Autoformer + RevIN + Dish-TS)
#           × 4 prediction lengths × 3 seeds
#           + additional alpha sweep for Dish-TS.
#
# Paper defaults:
#   raw-data (no --scale) / L = H (--seq_len 0, auto == pred_len)
#   label_len = 48              (decoder warm-start)
#   batch_size 128              (Autoformer default; ECL is forced to 64)
#   train_epochs 100 / patience 7
#   dish_init = avg             (Table 6 ablation default)
#   prior = paper-phi-only      (prior signal reaches HoriCoNet only)
#   alpha ∈ {0.0, 0.25, 0.5, 1.0}  (Figure 3 sweep, Table 3 main result)
#
# Total jobs:
#   dishts: 4 datasets × 4 preds × 4 alphas × 3 seeds   = 192
#   revin : 4 datasets × 4 preds × 3 seeds              = 48
#   none  : 4 datasets × 4 preds × 3 seeds              = 48
#   total : 288 jobs
#
# At ~4 min/job on a 3090 this is roughly 19 h of wall time.
#
# NOTE: the official Dish-TS repo does NOT use an alpha prior loss at all
# (equivalent to --prior none / alpha 0).  So both of the following are
# legitimate paper baselines and we keep both:
#   * dishts with alpha 0 / prior none         -> official repo equiv
#   * dishts with prior paper-phi-only / alpha>0 -> with the paper's claimed
#                                                    horizon guidance signal

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

MODEL=Autoformer
DISH_INIT=avg
SEQ_LEN=0            # 0 means "auto-align to pred_len" (paper L=H)
TRAIN_EPOCHS=100
PATIENCE=7
GPU=0

mkdir -p logs results

DATASETS=(ETTm2 ETTh1 ECL WTH)
PREDS=(24 96 168 336)
SEEDS=(2023 2024 2025)
ALPHAS=(0.0 0.25 0.5 1.0)

# --- job counting ---------------------------------------------------------
total=0
for _ in "${DATASETS[@]}"; do
    for _ in "${PREDS[@]}"; do
        for _ in "${SEEDS[@]}"; do
            # dishts × 4 alphas + revin + none = 6 lines
            for _ in 1 2 3 4 5 6; do total=$((total + 1)); done
        done
    done
done
# dishts only: 4 d × 4 p × 3 s × 4 a = 192
# revin + none: 4 d × 4 p × 3 s × 2 = 96
# but the loop above counts per "(d,p,s)" 6 lines which is wrong; just
# recompute explicitly:
total=$(( ${#DATASETS[@]} * ${#PREDS[@]} * ${#SEEDS[@]} * (${#ALPHAS[@]} + 2) ))

i=0
for DATA in "${DATASETS[@]}"; do
    # ECL (Electricity): many series × long sequences → paper uses bs=64
    if [ "$DATA" = "ECL" ]; then
        BATCH_SIZE=64
    else
        BATCH_SIZE=128
    fi

    for p in "${PREDS[@]}"; do
        for s in "${SEEDS[@]}"; do
            # --- plain Autoformer (norm = none) ---
            i=$((i + 1))
            log="logs/${DATA}_${MODEL}_none_seq${p}_pred${p}_seed${s}_raw.log"
            echo "[$i/$total] plain Autoformer  data=$DATA  pred=$p  seed=$s"
            python3 -u train.py \
                --data "$DATA" --model "$MODEL" --norm none \
                --seq_len "$SEQ_LEN" --pred_len "$p" --label_len 0 \
                --batch_size "$BATCH_SIZE" \
                --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
                --seed "$s" --gpu "$GPU" --figure3 \
                >"$log" 2>&1

            # --- RevIN baseline ---
            i=$((i + 1))
            log="logs/${DATA}_${MODEL}_revin_seq${p}_pred${p}_seed${s}_raw.log"
            echo "[$i/$total] RevIN  data=$DATA  pred=$p  seed=$s"
            python3 -u train.py \
                --data "$DATA" --model "$MODEL" --norm revin \
                --seq_len "$SEQ_LEN" --pred_len "$p" --label_len 0 \
                --batch_size "$BATCH_SIZE" \
                --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
                --seed "$s" --gpu "$GPU" --figure3 \
                >"$log" 2>&1

            # --- Dish-TS with alpha sweep (prior=paper-phi-only) ---
            for a in "${ALPHAS[@]}"; do
                i=$((i + 1))
                log="logs/${DATA}_${MODEL}_dishts_seq${p}_pred${p}_seed${s}_alpha${a}_priorPaperPhiOnly_raw.log"
                echo "[$i/$total] Dish-TS(paper-phi-only)  data=$DATA  pred=$p  alpha=$a  seed=$s"
                python3 -u train.py \
                    --data "$DATA" --model "$MODEL" --norm dishts \
                    --alpha "$a" --prior paper-phi-only --dish_init "$DISH_INIT" \
                    --seq_len "$SEQ_LEN" --pred_len "$p" --label_len 0 \
                    --batch_size "$BATCH_SIZE" \
                    --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
                    --seed "$s" --gpu "$GPU" --figure3 \
                    >"$log" 2>&1
            done
        done
    done
done

echo
echo "============================================================"
echo "All $total jobs finished."
echo "Now inspect results with:  python3 repro_figures/compare_paper.py"
echo "                           python3 show_alpha_curves.py"
