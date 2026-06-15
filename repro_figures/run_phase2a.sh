#!/usr/bin/env bash
#
# Phase 2a — ETTm2-only focused alpha sweep (denser grid than Phase 1).
#
# Purpose
# -------
# After Phase 1 confirms alpha=0.0 is not dominated by alpha>0 on ETTm2,
# we run a finer grid on ETTm2 only to decide whether a multi-dataset
# sweep (Phase 2b) is worth the GPU time.
#
# Grid (single dataset, single backbone, single norm, single prior)
# ----
#   dataset    : ETTm2
#   backbone   : Autoformer
#   norm       : dishts
#   prior      : paper-phi-only
#   horizons   : 96, 168, 336
#   alphas     : 0.0, 0.25, 0.5, 0.75, 1.0
#   seeds      : 2023, 2024, 2025
# Total jobs : 3 horizons x 5 alphas x 3 seeds = 45
# Runtime    : ~60-90 minutes on a single RTX 3090.
#
# Outputs
# -------
#   * Per-job log      : logs/fig3_ETTm2_Autoformer_paper-phi-only_PL*_alpha*_seed*.log
#   * Master log       : logs/phase2a_master.log
#   * Cumulative CSV   : results/figure3_runs.csv (1 row appended per job)
#
# Usage (on 3090)
# -------
#   screen -U -S dishts-phase2a
#   cd ~/autodl-tmp/Dish-TS
#   bash repro_figures/run_phase2a.sh
#   # Ctrl+A, D to detach; screen -r dishts-phase2a to reattach.
#
# After the sweep finishes, run:
#   python3 repro_figures/check_phase2a.py
# to print the alpha-vs-MSE summary table.

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
DATASET="${DATASET:-ETTm2}"           # kept as a variable so you can re-use
SEEDS=(2023 2024 2025)
PREDS=(96 168 336)
ALPHAS=(0.0 0.25 0.5 0.75 1.0)

# --- speed-vs-fidelity knobs ---
# Phase 2a defaults match the main comparison configuration used for
# Tables 2/3 (patience=7, max_epochs=100). You can still override at
# call time, e.g. `PATIENCE=3 MAX_EPOCHS=50 bash repro_figures/run_phase2a.sh`
# to do a shorter qualitative sweep.
PATIENCE="${PATIENCE:-7}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"

mkdir -p logs results

MASTER_LOG="logs/phase2a_master.log"
: > "$MASTER_LOG"

# -------- diagnostics header --------
njobs=$(( ${#PREDS[@]} * ${#ALPHAS[@]} * ${#SEEDS[@]} ))
{
    echo "[phase2a] dataset=$DATASET backbone=Autoformer norm=dishts prior=paper-phi-only"
    echo "[phase2a] horizons=${PREDS[*]}   alphas=${ALPHAS[*]}   seeds=${SEEDS[*]}"
    echo "[phase2a] patience=$PATIENCE max_epochs=$MAX_EPOCHS (faster sweep — see run_phase2b.sh for tighter config)"
    echo "[phase2a] TOTAL=$njobs jobs"
    echo "[phase2a] GPU=$GPU  cwd=$PROJ_ROOT"
} | tee -a "$MASTER_LOG"

nvidia_smi_line=$(nvidia-smi --format=csv,noheader --query-gpu=name,memory.total 2>/dev/null | head -1)
echo "[phase2a] GPU: $nvidia_smi_line" | tee -a "$MASTER_LOG"

# -------- sweep --------
done=0
for PL in "${PREDS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            log="logs/fig3_${DATASET}_Autoformer_paper-phi-only_PL${PL}_alpha${alpha}_seed${seed}.log"
            {
                echo "[phase2a] ($((done+1))/$njobs) pl=$PL alpha=$alpha seed=$seed bs=128 patience=$PATIENCE max_epochs=$MAX_EPOCHS -> $log"
            } | tee -a "$MASTER_LOG"

            python3 -u train.py \
                --data "$DATASET" --model Autoformer --norm dishts \
                --alpha "$alpha" --prior paper-phi-only --dish_init avg \
                --seq_len "$PL" --pred_len "$PL" --label_len 48 \
                --batch_size 128 --lr 1e-3 --train_epochs "$MAX_EPOCHS" --patience "$PATIENCE" \
                --features M --seed "$seed" --gpu "$GPU" --figure3 \
                >> "$log" 2>&1
            rc=$?

            if [[ $rc -ne 0 ]]; then
                echo "[phase2a]  !! FAILED (rc=$rc) for PL=$PL alpha=$alpha seed=$seed" | tee -a "$MASTER_LOG"
            fi
            done=$((done+1))
        done
    done
done

echo "[phase2a] ALL DONE ($done jobs)" | tee -a "$MASTER_LOG"
echo "[phase2a] next step: python3 repro_figures/check_phase2a.py" | tee -a "$MASTER_LOG"
