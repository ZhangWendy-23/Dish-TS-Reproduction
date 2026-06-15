#!/usr/bin/env bash
#
# Phase 2b — extend Figure-3 alpha sweep to ECL, ETTh1, WTH.
#
# Use it *after* Phase 2a shows a consistent trend on ETTm2.
#
# Grid per dataset
# ----
#   backbone : Autoformer
#   norm     : dishts
#   prior    : paper-phi-only
#   horizons : 96, 168, 336
#   alphas   : 0.0, 0.25, 0.5, 0.75, 1.0
#   seeds    : 2023, 2024, 2025
# Total per dataset : 3 horizons x 5 alphas x 3 seeds = 45 jobs.
# Total for 3 datasets: 135 jobs (~6-9 hours on a single 3090).
#
# Batch size:  64 for ECL / WTH (wider feature dim, more memory)
#              128 for ETTh1 / ETTm2
#
# Usage
# -----
#   screen -U -S dishts-phase2b
#   cd ~/autodl-tmp/Dish-TS
#   DATASETS="ECL ETTh1 WTH" bash repro_figures/run_phase2b.sh
#   # Ctrl+A, D

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
SEEDS=(2023 2024 2025)
PREDS=(96 168 336)
ALPHAS=(0.0 0.25 0.5 0.75 1.0)
DATASETS=(${DATASETS:-ECL ETTh1 WTH})

# --- speed-vs-fidelity knobs ---
# Phase 2b = confirmatory sweep on ECL/ETTh1/WTH. A bit less aggressive
# than Phase 2a, but still faster than the "patience=7 / max_epochs=100"
# comparison baselines used for Tables 2/3.
PATIENCE="${PATIENCE:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-70}"

mkdir -p logs results

MASTER_LOG="logs/phase2b_master.log"
: > "$MASTER_LOG"

total_per_ds=$(( ${#PREDS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} ))
total=$(( total_per_ds * ${#DATASETS[@]} ))
{
    echo "[phase2b] datasets=${DATASETS[*]}"
    echo "[phase2b] horizons=${PREDS[*]}   alphas=${ALPHAS[*]}   seeds=${SEEDS[*]}"
    echo "[phase2b] patience=$PATIENCE max_epochs=$MAX_EPOCHS"
    echo "[phase2b] TOTAL=$total jobs ($total_per_ds per dataset)"
    echo "[phase2b] GPU=$GPU  cwd=$PROJ_ROOT"
} | tee -a "$MASTER_LOG"

done=0
for DATA in "${DATASETS[@]}"; do
    case "$DATA" in
        ECL|WTH) bs=64 ;;
        *)       bs=128 ;;
    esac
    for PL in "${PREDS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                log="logs/fig3_${DATA}_Autoformer_paper-phi-only_PL${PL}_alpha${alpha}_seed${seed}.log"
                {
                    echo "[phase2b] ($((done+1))/$total) data=$DATA pl=$PL alpha=$alpha seed=$seed bs=$bs patience=$PATIENCE max_epochs=$MAX_EPOCHS -> $log"
                } | tee -a "$MASTER_LOG"

                python3 -u train.py \
                    --data "$DATA" --model Autoformer --norm dishts \
                    --alpha "$alpha" --prior paper-phi-only --dish_init avg \
                    --seq_len "$PL" --pred_len "$PL" --label_len 48 \
                    --batch_size "$bs" --lr 1e-3 --train_epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
                    --features M --seed "$seed" --gpu "$GPU" --figure3 \
                    >> "$log" 2>&1
                rc=$?
                if [[ $rc -ne 0 ]]; then
                    echo "[phase2b]  !! FAILED (rc=$rc) data=$DATA pl=$PL alpha=$alpha seed=$seed" \
                        | tee -a "$MASTER_LOG"
                fi
                done=$((done+1))
            done
        done
    done
done

echo "[phase2b] ALL DONE ($done jobs)" | tee -a "$MASTER_LOG"
