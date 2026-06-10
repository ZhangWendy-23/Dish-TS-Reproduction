#!/bin/bash
# Dish-TS - Final Simplified Experiments (18 runs)
LOG_DIR="/root/autodl-tmp/Dish-TS/logs/simplified"
TRAIN_SCRIPT="/root/autodl-tmp/Dish-TS/train.py"
MASTER_LOG="${LOG_DIR}/_master.log"

echo "========================================" | tee "${MASTER_LOG}"
echo " Dish-TS  -  Simplified Experiments    " | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
echo " datasets   : ETTm2" | tee -a "${MASTER_LOG}"
echo " models     : Autoformer Informer" | tee -a "${MASTER_LOG}"
echo " norms      : none revin dishts" | tee -a "${MASTER_LOG}"
echo " pred_len   : 24 96 336" | tee -a "${MASTER_LOG}"
echo " batch_size : 0 (paper default)" | tee -a "${MASTER_LOG}"
echo " alpha      : 0.5" | tee -a "${MASTER_LOG}"
echo " patience   : 7" | tee -a "${MASTER_LOG}"
echo " seed       : 2023" | tee -a "${MASTER_LOG}"
echo " gpu        : 0" | tee -a "${MASTER_LOG}"
echo "----------------------------------------" | tee -a "${MASTER_LOG}"

TOTAL=18
COUNTER=0

for DATA in ETTm2; do
    for MODEL in Autoformer Informer; do
        for NORM in none revin dishts; do
            for PL in 24 96 336; do
                COUNTER=$((COUNTER + 1))
                LOG_FILE="${LOG_DIR}/${DATA}_${MODEL}_${NORM}_pred${PL}.log"
                echo "[$(date '+%H:%M:%S')] [${COUNTER}/${TOTAL}] ${DATA}/${MODEL}/${NORM}/pred=${PL}" | tee -a "${MASTER_LOG}"

                python -u "${TRAIN_SCRIPT}" \
                    --data "${DATA}" \
                    --model "${MODEL}" \
                    --norm "${NORM}" \
                    --seq_len 96 \
                    --pred_len "${PL}" \
                    --batch_size 0 \
                    --alpha 0.5 \
                    --patience 7 \
                    --seed 2023 \
                    --gpu 0 \
                    > "${LOG_FILE}" 2>&1

                RESULT=$(tail -2 "${LOG_FILE}" | grep "^0 " | head -1)
                if [ -n "${RESULT}" ]; then
                    echo "  -> ${RESULT}" | tee -a "${MASTER_LOG}"
                else
                    echo "  -> (未找到结果，日志末尾：)" | tee -a "${MASTER_LOG}"
                    tail -3 "${LOG_FILE}" | tee -a "${MASTER_LOG}"
                fi
            done
        done
    done
done

echo "========================================" | tee -a "${MASTER_LOG}"
echo " All experiments completed at $(date '+%H:%M:%S')" | tee -a "${MASTER_LOG}"
echo "========================================" | tee -a "${MASTER_LOG}"
