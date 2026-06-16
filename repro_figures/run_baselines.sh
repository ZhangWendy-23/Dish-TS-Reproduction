#!/bin/bash
cd "$(dirname "$0")"/..

DATASET="ETTm2"
MODEL="Autoformer"
SEQLEN=96

for PRED in 24 96 168 336; do
    LABELLEN=$(( PRED < 48 ? PRED : 48))
    for NORM in none revin; do
        for SEED in 2023 2024 2025; do
            python3 -u train.py \
                --data "$DATASET" \
                --model "$MODEL" \
                --norm "$NORM" \
                --dish_init avg \
                --seq_len "$SEQLEN" \
                --pred_len "$PRED" \
                --label_len "$LABELLEN" \
                --batch_size 128 \
                --train_epochs 50 \
                --patience 7 \
                --seed "$SEED" \
                --gpu 0 \
                --figure3
        done
    done
done

echo "All 所有 baseline jobs done"
