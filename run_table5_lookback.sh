#!/usr/bin/env bash
# ============================================================================
# run_table5_lookback.sh
# ============================================================================
# Table 5 - Lookback length sensitivity analysis (ETTm2)
#
# Tests the impact of input window length (seq_len) on prediction quality.
# This validates Dish-TS's claim of being more robust to lookback length
# than RevIN normalization.
#
# Configuration (paper-identical):
#   - Dataset:     ETTm2
#   - Model:       Autoformer
#   - Normalization: revin, dishts (alpha=0.1, from Figure 3 analysis)
#   - seq_len:     96, 192, 336, 720
#   - pred_len:    96, 168, 336
#   - label_len:   matching seq_len
#   - Seeds:       2023, 2024, 2025
#
# Total: 1 dataset x 2 norms x 4 seq_lens x 3 preds x 3 seeds = 72 experiments
#
# Each experiment logs to results/figure3_runs.csv automatically (--figure3)
# ============================================================================

set -u

DATA="ETTm2"
NORMS=("revin" "dishts")
SEQ_LENS=(96 192 336 720)
PRED_LENS=(96 168 336)
SEEDS=(2023 2024 2025)

BATCH_SIZE=0
PATIENCE=15
MODEL="Autoformer"
ALPHA=0.1   # for dishts - from Figure 3 analysis
GPU=0

LOG_DIR="logs/table5_lookback"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/_master.log"

TOTAL=$((${#NORMS[@]} * ${#SEQ_LENS[@]} * ${#PRED_LENS[@]} * ${#SEEDS[@]}))

echo "========================================" | tee -a "${MASTER_LOG}"
echo " Table 5 - Lookback Length Sensitivity  " | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
echo " dataset    : ${DATA}" | tee -a "${MASTER_LOG}"
echo " model      : ${MODEL}" | tee -a "${MASTER_LOG}"
echo " norms      : ${NORMS[*]}" | tee -a "${MASTER_LOG}"
echo " seq_len    : ${SEQ_LENS[*]}" | tee -a "${MASTER_LOG}"
echo " pred_len   : ${PRED_LENS[*]}" | tee -a "${MASTER_LOG}"
echo " seeds      : ${SEEDS[*]}" | tee -a "${MASTER_LOG}"
echo " alpha      : ${ALPHA} (for dishts)" | tee -a "${MASTER_LOG}"
echo " total      : ${TOTAL}" | tee -a "${MASTER_LOG}"
echo " gpu        : ${GPU}" | tee -a "${MASTER_LOG}"
echo "----------------------------------------" | tee -a "${MASTER_LOG}"

COUNTER=0
START_TIME=$(date +%s)

for NORM in "${NORMS[@]}"; do
    for SL in "${SEQ_LENS[@]}"; do
        for PL in "${PRED_LENS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                COUNTER=$((COUNTER + 1))
                LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_${NORM}_sl${SL}_pl${PL}_seed${SEED}.log"

                echo "[${COUNTER}/${TOTAL}] $(date '+%Y-%m-%d %H:%M:%S') ${NORM}/sl=${SL}/pl=${PL}/seed=${SEED} -> ${LOG_FILE}" | tee -a "${MASTER_LOG}"

                EXP_START=$(date +%s)

                python3 -u train.py \
                    --data "${DATA}" \
                    --model "${MODEL}" \
                    --norm "${NORM}" \
                    --seq_len "${SL}" \
                    --label_len "${SL}" \
                    --pred_len "${PL}" \
                    --batch_size "${BATCH_SIZE}" \
                    --alpha "${ALPHA}" \
                    --patience "${PATIENCE}" \
                    --seed "${SEED}" \
                    --gpu "${GPU}" \
                    --figure3 \
                    > "${LOG_FILE}" 2>&1

                EXP_END=$(date +%s)
                EXP_DUR=$((EXP_END - EXP_START))

                MSE_LINE=$(grep -E "^[[:space:]]*0[[:space:]]" "${LOG_FILE}" | tail -1)
                if [ -n "${MSE_LINE}" ]; then
                    MSE_VAL=$(echo "${MSE_LINE}" | awk '{print $NF}')
                    echo "    => done in ${EXP_DUR}s, MSE=${MSE_VAL}" | tee -a "${MASTER_LOG}"
                else
                    echo "    => done in ${EXP_DUR}s (no MSE found in log)" | tee -a "${MASTER_LOG}"
                fi
            done
        done
    done
done

END_TIME=$(date +%s)
TOTAL_DUR=$((END_TIME - START_TIME))

echo "========================================" | tee -a "${MASTER_LOG}"
echo " All ${TOTAL} experiments finished." | tee -a "${MASTER_LOG}"
echo " Total wall time: ${TOTAL_DUR}s ($((TOTAL_DUR/60)) min)" | tee -a "${MASTER_LOG}"
echo " Master log: ${MASTER_LOG}" | tee -a "${MASTER_LOG}"
echo " Individual logs: ${LOG_DIR}/*.log" | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
