#!/usr/bin/env bash
# ============================================================================
# run_figure3_extend.sh
# ============================================================================
# Extended Figure 3 alpha-sensitivity experiments for ECL / ETTh1 / WTH.
#
# Configuration (paper-identical):
#   - Datasets:    ECL, ETTh1, WTH
#   - Model:       Autoformer
#   - Normalization: dishts (with alpha sweep)
#   - alphas:      0.0, 0.1, 0.25, 0.5, 0.75, 1.0
#   - pred_len:    24, 96, 168, 336
#   - seq_len:     96
#   - label_len:   96
#   - Seeds:       2023, 2024, 2025
#
# Total: 3 datasets x 6 alphas x 4 preds x 3 seeds = 216 experiments
#
# Each experiment logs to results/figure3_runs.csv automatically (--figure3)
# ============================================================================

set -u

DATASETS=("ECL" "ETTh1" "WTH")
ALPHAS=(0.0 0.1 0.25 0.5 0.75 1.0)
PRED_LENS=(24 96 168 336)
SEEDS=(2023 2024 2025)

SEQ_LEN=96
LABEL_LEN=96
BATCH_SIZE=0
PATIENCE=15
MODEL="Autoformer"
NORM="dishts"
GPU=0

LOG_DIR="logs/figure3_extend"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/_master.log"

TOTAL=0
for D in "${DATASETS[@]}"; do
    for A in "${ALPHAS[@]}"; do
        for P in "${PRED_LENS[@]}"; do
            for S in "${SEEDS[@]}"; do
                TOTAL=$((TOTAL + 1))
            done
        done
    done
done

echo "========================================" | tee -a "${MASTER_LOG}"
echo " Figure 3 Extended - Alpha Sensitivity  " | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
echo " datasets   : ${DATASETS[*]}" | tee -a "${MASTER_LOG}"
echo " alphas     : ${ALPHAS[*]}" | tee -a "${MASTER_LOG}"
echo " pred_len   : ${PRED_LENS[*]}" | tee -a "${MASTER_LOG}"
echo " seeds      : ${SEEDS[*]}" | tee -a "${MASTER_LOG}"
echo " model      : ${MODEL}" | tee -a "${MASTER_LOG}"
echo " norm       : ${NORM}" | tee -a "${MASTER_LOG}"
echo " seq_len    : ${SEQ_LEN}" | tee -a "${MASTER_LOG}"
echo " total      : ${TOTAL}" | tee -a "${MASTER_LOG}"
echo " gpu        : ${GPU}" | tee -a "${MASTER_LOG}"
echo "----------------------------------------" | tee -a "${MASTER_LOG}"

COUNTER=0
START_TIME=$(date +%s)

for DATA in "${DATASETS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        for PL in "${PRED_LENS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                COUNTER=$((COUNTER + 1))
                LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_dishts_alpha${ALPHA}_pl${PL}_seed${SEED}.log"

                echo "[${COUNTER}/${TOTAL}] $(date '+%Y-%m-%d %H:%M:%S') ${DATA}/alpha=${ALPHA}/pl=${PL}/seed=${SEED} -> ${LOG_FILE}" | tee -a "${MASTER_LOG}"

                EXP_START=$(date +%s)

                python3 -u train.py \
                    --data "${DATA}" \
                    --model "${MODEL}" \
                    --norm "${NORM}" \
                    --seq_len "${SEQ_LEN}" \
                    --label_len "${LABEL_LEN}" \
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
