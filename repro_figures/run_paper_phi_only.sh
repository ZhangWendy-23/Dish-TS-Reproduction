#!/usr/bin/env bash
#
# Paper-matching sweep: ETTm2 × 4 pred_len × 4 alpha × 3 seeds ×
# prior=paper-phi-only.  The key experimental decisions:
#   --seq_len 0             →  the paper's "L = H" convention (lookback ==
#                              prediction length; this is the DEFAULT for
#                              Tables 1-3 in the paper).
#   --pred_len <24|96|168|336>
#   --label_len 0          →  auto-clamp to min(seq_len, pred_len) with
#                              floor 48 (paper default is 48).
#   --prior paper-phi-only →  prior loss gradient reaches HoriCoNet only;
#                              BackCoNet is trained purely by the base MSE.
#   --dish_init avg        →  paper Table 6 default.
#   --batch_size 128       →  paper Autoformer default (Table 2/3).
#   --train_epochs 100     →  paper default.
#   --patience 7           →  paper default.
#   NO --scale             →  the paper uses raw data ("directly take the
#                              original data").
#
# Total jobs: 4 pred × 4 alpha × 3 seeds = 48.
# Runtime estimate:  ~4 min/job × 48 = ~3.2 h on a single 3090.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DATA=ETTm2
MODEL=Autoformer
NORM=dishts
PRIOR=paper-phi-only
DISH_INIT=avg
SEQ_LEN=0           # 0 = auto-align to pred_len (paper L = H)
BATCH_SIZE=128
TRAIN_EPOCHS=100
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
                --seq_len "$SEQ_LEN" --pred_len "$p" --label_len 0 \
                --batch_size "$BATCH_SIZE" \
                --train_epochs "$TRAIN_EPOCHS" --patience "$PATIENCE" \
                --seed "$s" --gpu "$GPU" \
                --figure3 \
                >"$log" 2>&1
        done
    done
done

echo "All 48 jobs done.  Now run: python3 repro_figures/compare_paper.py --prior paper-phi-only"
