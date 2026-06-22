#!/usr/bin/env bash
# ============================================================
# 停机后重启脚本 —— 只跑尚未完成的实验，跳过已完成的
# ============================================================
# Table 2: 432 jobs → 已完成 402，缺失 30（全部在 WTH）
# Table 3: 540 jobs → 缺失 60（含 H=48 的 T3-only）
# 总计需补跑: ~90 jobs (~8 小时)
# ============================================================

set -u
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

GPU="${GPU:-0}"
MASTER_LOG="logs/restart_missing_master.log"

echo "=== 停机后重启 $(date) ===" | tee "$MASTER_LOG"

# ----------  Table 2 缺失 (30 jobs, 全部 WTH) ----------
# WTH Autoformer dishts H=336 (3 seeds)
# WTH NBEATS none H=24/96/168/336 (12 seeds)
# WTH NBEATS revin H=24/96/168/336 (12 seeds)
# WTH NBEATS dishts H=336 (3 seeds)
echo "[restart] Table 2 — 30 missing jobs" | tee -a "$MASTER_LOG"

TABLE2_JOBS=(
  "WTH Autoformer dishts 336"
  "WTH NBEATS none 24"
  "WTH NBEATS none 96"
  "WTH NBEATS none 168"
  "WTH NBEATS none 336"
  "WTH NBEATS revin 24"
  "WTH NBEATS revin 96"
  "WTH NBEATS revin 168"
  "WTH NBEATS revin 336"
  "WTH NBEATS dishts 336"
)

SEEDS=(2023 2024 2025)
i=0
for job in "${TABLE2_JOBS[@]}"; do
  read -r DATA MODEL NORM H <<< "$job"
  bs=128
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    log="logs/table2_${DATA}_${MODEL}_${NORM}_L${H}_H${H}_seed${seed}.log"
    echo "[$i/60] $DATA $MODEL $NORM H=$H seed=$seed" | tee -a "$MASTER_LOG"
    python3 -u train.py \
      --data "$DATA" --model "$MODEL" --norm "$NORM" \
      --alpha 0.0 --prior none --dish_init avg \
      --seq_len "$H" --pred_len "$H" --label_len 48 \
      --batch_size "$bs" --lr 1e-3 --train_epochs 100 --patience 10 \
      --features M --seed "$seed" --gpu "$GPU" --figure3 \
      >> "$log" 2>&1
  done
done

# ----------  Table 3 缺失 (60 jobs) ----------
echo "[restart] Table 3 — 60 missing jobs (incl. H=48)" | tee -a "$MASTER_LOG"

TABLE3_JOBS=(
  # H=48 (T3-only)
  "ECL NBEATS revin 48"
  "ETTh1 NBEATS revin 48"
  "ETTm2 NBEATS none 48"
  "ETTm2 NBEATS revin 48"
  "ETTm2 Autoformer dishts 48"
  "WTH Autoformer dishts 48"
  "WTH NBEATS none 48"
  "WTH NBEATS revin 48"
  "WTH NBEATS dishts 48"
  "WTH Informer none 48"
  # H=24/96/168/336 gaps
  "ECL NBEATS revin 24"
  "ECL NBEATS revin 96"
  "ECL NBEATS revin 168"
  "ECL NBEATS revin 336"
  "ETTh1 NBEATS revin 24"
  "ETTh1 NBEATS revin 96"
  "ETTh1 NBEATS revin 168"
  "ETTh1 NBEATS revin 336"
  "ETTm2 NBEATS none 24"
  "ETTm2 NBEATS none 96"
  "ETTm2 NBEATS none 168"
  "ETTm2 NBEATS none 336"
  "ETTm2 NBEATS revin 24"
  "ETTm2 NBEATS revin 96"
  "ETTm2 NBEATS revin 168"
  "ETTm2 NBEATS revin 336"
  "ETTm2 Autoformer dishts 24"
  "ETTm2 Autoformer dishts 96"
  "ETTm2 Autoformer dishts 168"
  "ETTm2 Autoformer dishts 336"
)

for job in "${TABLE3_JOBS[@]}"; do
  read -r DATA MODEL NORM H <<< "$job"
  if [[ "$DATA" == "ECL" ]]; then
    bs=64
  elif [[ "$MODEL" == "Informer" ]]; then
    bs=256
  else
    bs=128
  fi
  for seed in "${SEEDS[@]}"; do
    i=$((i+1))
    log="logs/table2_${DATA}_${MODEL}_${NORM}_L${H}_H${H}_seed${seed}.log"
    echo "[$i/90] $DATA $MODEL $NORM H=$H seed=$seed" | tee -a "$MASTER_LOG"
    python3 -u train.py \
      --data "$DATA" --model "$MODEL" --norm "$NORM" \
      --seq_len 0 --pred_len "$H" \
      --batch_size "$bs" --lr 1e-3 \
      --train_epochs 100 --patience 7 \
      --features M --seed "$seed" --gpu "$GPU" \
      --figure3 \
      >> "$log" 2>&1
  done
done

echo "=== DONE $(date) ===" | tee -a "$MASTER_LOG"
echo "Total jobs run: $i" | tee -a "$MASTER_LOG"
