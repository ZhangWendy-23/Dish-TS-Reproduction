#!/usr/bin/env bash
# ============================================================================
# run_simplified_exps.sh
# ============================================================================
# Dish-TS - Simplified / Quick-Start experiment runner.
#
# This script runs a SUBSET of the paper experiments, targeting the most
# informative configurations:
#   - Primary dataset: ETTm2
#   - Models:        Autoformer (paper's main model) + Informer
#   - Normalization: none / revin / dishts
#   - pred_len:      24 (short) / 96 (medium) / 336 (long)
#
# Total: 2 models x 3 norms x 3 horizons = 18 experiments
# Expected runtime: ~1-3 hours on a single RTX 3080 Ti / A100.
#
# For FULL paper reproduction, use run_paper_exps.sh.
# ============================================================================

set -u

DATASETS=("ETTm2")
MODELS=("Autoformer" "Informer")
NORMS=("none" "revin" "dishts")
PRED_LENS=(24 96 336)

# -------- constants (all paper-identical) --------
SEQ_LEN=96
BATCH_SIZE=0        # 0 = use paper defaults (Autoformer=128, Informer=256)
ALPHA=0.5           # prior knowledge guidance weight (paper: 0.5 default)
PATIENCE=7          # early-stop patience (paper)
SEED=2023           # fixed seed for reproducibility
GPU=0

LOG_DIR="logs/simplified"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/_master.log"

echo "========================================" | tee -a "${MASTER_LOG}"
echo " Dish-TS  -  Simplified Experiments    " | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
echo " datasets   : ${DATASETS[*]}"          | tee -a "${MASTER_LOG}"
echo " models     : ${MODELS[*]}"            | tee -a "${MASTER_LOG}"
echo " norms      : ${NORMS[*]}"             | tee -a "${MASTER_LOG}"
echo " pred_len   : ${PRED_LENS[*]}"         | tee -a "${MASTER_LOG}"
echo " batch_size : ${BATCH_SIZE} (paper default)" | tee -a "${MASTER_LOG}"
echo " alpha      : ${ALPHA}"                | tee -a "${MASTER_LOG}"
echo " patience   : ${PATIENCE}"             | tee -a "${MASTER_LOG}"
echo " seed       : ${SEED}"                 | tee -a "${MASTER_LOG}"
echo " gpu        : ${GPU}"                  | tee -a "${MASTER_LOG}"
echo "----------------------------------------" | tee -a "${MASTER_LOG}"

TOTAL=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#NORMS[@]} * ${#PRED_LENS[@]}))
COUNTER=0

for DATA in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for NORM in "${NORMS[@]}"; do
            for PL in "${PRED_LENS[@]}"; do
                COUNTER=$((COUNTER + 1))
                LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_${NORM}_pred${PL}.log"

                echo "[${COUNTER}/${TOTAL}] $(date '+%H:%M:%S') ${DATA}/${MODEL}/${NORM}/pred=${PL} -> ${LOG_FILE}" | tee -a "${MASTER_LOG}"

                python -u train.py \
                    --data "${DATA}" \
                    --model "${MODEL}" \
                    --norm "${NORM}" \
                    --seq_len "${SEQ_LEN}" \
                    --pred_len "${PL}" \
                    --batch_size "${BATCH_SIZE}" \
                    --alpha "${ALPHA}" \
                    --patience "${PATIENCE}" \
                    --seed "${SEED}" \
                    --gpu "${GPU}" \
                    > "${LOG_FILE}" 2>&1

                # extract and log summary (last line of output = DataFrame)
                RESULT_LINE=$(tail -n 3 "${LOG_FILE}" | grep -E "^[[:space:]]*0[[:space:]]" | head -n 1)
                if [ -n "${RESULT_LINE}" ]; then
                    echo "    => ${RESULT_LINE}" | tee -a "${MASTER_LOG}"
                else
                    echo "    => (未找到结果行, 请检查 ${LOG_FILE})" | tee -a "${MASTER_LOG}"
                fi
            done
        done
    done
done

echo "========================================" | tee -a "${MASTER_LOG}"
echo " All ${TOTAL} simplified experiments finished." | tee -a "${MASTER_LOG}"
echo " Master log: ${MASTER_LOG}"            | tee -a "${MASTER_LOG}"
echo " Individual logs: ${LOG_DIR}/*.log"    | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
