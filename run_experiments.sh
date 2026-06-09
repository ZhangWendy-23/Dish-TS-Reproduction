#!/bin/bash
# ============================================================================
# run_experiments.sh — Dish-TS Paper Reproduction Suite
# ============================================================================
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh phase1    # ETTm2 full comparison (1 seed)
#   ./run_experiments.sh phase2    # ETTh1 full comparison (1 seed)
#   ./run_experiments.sh phase3    # ECL/WTH/ILI (1 seed)
#   ./run_experiments.sh phase4    # Multi-seed final results (ETTm2)
#   ./run_experiments.sh phase5    # Ablation: DishTS init methods
#   ./run_experiments.sh all       # Run all phases (CAUTION: very long!)
# ============================================================================

set -e
GPU=0
SEED=2023
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# ─── Phase 1: ETTm2 — 3 models × 3 norms × 4 pred_lens = 36 runs ───
phase1() {
    echo "========================================"
    echo "  Phase 1: ETTm2 Full Comparison"
    echo "========================================"
    DATA=ETTm2
    for model in Transformer Informer Autoformer; do
        for norm in none revin dishts; do
            for pred_len in 96 192 336 720; do
                LOG="$LOG_DIR/${DATA}_${model}_${norm}_pred${pred_len}_seed${SEED}.log"
                echo "[$(date +%H:%M:%S)] $model + $norm, pred_len=$pred_len"
                python train.py \
                    --data $DATA \
                    --model $model \
                    --norm $norm \
                    --pred_len $pred_len \
                    --seq_len 96 \
                    --seed $SEED \
                    --gpu $GPU 2>&1 | tee "$LOG"
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Phase 1 completed."
}

# ─── Phase 2: ETTh1 — 3 models × 3 norms × 4 pred_lens = 36 runs ───
phase2() {
    echo "========================================"
    echo "  Phase 2: ETTh1 Full Comparison"
    echo "========================================"
    DATA=ETTh1
    for model in Transformer Informer Autoformer; do
        for norm in none revin dishts; do
            for pred_len in 96 192 336 720; do
                LOG="$LOG_DIR/${DATA}_${model}_${norm}_pred${pred_len}_seed${SEED}.log"
                echo "[$(date +%H:%M:%S)] $model + $norm, pred_len=$pred_len"
                python train.py \
                    --data $DATA \
                    --model $model \
                    --norm $norm \
                    --pred_len $pred_len \
                    --seq_len 96 \
                    --seed $SEED \
                    --gpu $GPU 2>&1 | tee "$LOG"
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Phase 2 completed."
}

# ─── Phase 3: ECL / WTH / ILI — Autoformer × 2 norms × 4 pred_lens ───
phase3() {
    echo "========================================"
    echo "  Phase 3: Multi-Dataset Evaluation"
    echo "========================================"
    for data in ECL WTH ILI; do
        for norm in none dishts; do
            for pred_len in 96 192 336 720; do
                LOG="$LOG_DIR/${data}_Autoformer_${norm}_pred${pred_len}_seed${SEED}.log"
                echo "[$(date +%H:%M:%S)] $data + Autoformer + $norm, pred_len=$pred_len"
                python train.py \
                    --data $data \
                    --model Autoformer \
                    --norm $norm \
                    --pred_len $pred_len \
                    --seq_len 96 \
                    --seed $SEED \
                    --gpu $GPU 2>&1 | tee "$LOG"
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Phase 3 completed."
}

# ─── Phase 4: Multi-seed — ETTm2, Autoformer, 3 norms, 4 pred_lens, 5 seeds ───
phase4() {
    echo "========================================"
    echo "  Phase 4: Multi-Seed Evaluation (Paper-level)"
    echo "========================================"
    DATA=ETTm2
    for seed in 2023 2024 2025 2026 2027; do
        for norm in none revin dishts; do
            for pred_len in 96 192 336 720; do
                LOG="$LOG_DIR/${DATA}_Autoformer_${norm}_pred${pred_len}_seed${seed}.log"
                echo "[$(date +%H:%M:%S)] Autoformer + $norm, pred_len=$pred_len, seed=$seed"
                python train.py \
                    --data $DATA \
                    --model Autoformer \
                    --norm $norm \
                    --pred_len $pred_len \
                    --seq_len 96 \
                    --seed $seed \
                    --gpu $GPU 2>&1 | tee "$LOG"
            done
        done
    done
    echo "[$(date +%H:%M:%S)] Phase 4 completed."

    # Auto-compute mean ± std
    echo ""
    echo "=== Results Summary ==="
    for norm in none revin dishts; do
        for pred_len in 96 192 336 720; do
            echo "--- Autoformer + $norm, pred_len=$pred_len ---"
            grep -h "Autoformer.*$norm" "$LOG_DIR"/ETTm2_Autoformer_${norm}_pred${pred_len}_seed*.log | awk '{print $7}' | sort -n
        done
    done
}

# ─── Phase 5: Ablation — DishTS initialization methods ───
phase5() {
    echo "========================================"
    echo "  Phase 5: DishTS Initialization Ablation"
    echo "========================================"
    DATA=ETTm2
    for init in standard avg uniform; do
        for pred_len in 96 192 336 720; do
            LOG="$LOG_DIR/${DATA}_Autoformer_dishts-init${init}_pred${pred_len}_seed${SEED}.log"
            echo "[$(date +%H:%M:%S)] DishTS init=$init, pred_len=$pred_len"
            python train.py \
                --data $DATA \
                --model Autoformer \
                --norm dishts \
                --dish_init $init \
                --pred_len $pred_len \
                --seq_len 96 \
                --seed $SEED \
                --gpu $GPU 2>&1 | tee "$LOG"
        done
    done
    echo "[$(date +%H:%M:%S)] Phase 5 completed."
}

# ─── Dispatch ───
case "${1:-}" in
    phase1) phase1 ;;
    phase2) phase2 ;;
    phase3) phase3 ;;
    phase4) phase4 ;;
    phase5) phase5 ;;
    all)
        phase1
        phase2
        phase3
        phase4
        phase5
        ;;
    *)
        echo "Usage: $0 {phase1|phase2|phase3|phase4|phase5|all}"
        echo ""
        echo "  phase1  — ETTm2: 3 models × 3 norms × 4 pred_lens (36 runs)"
        echo "  phase2  — ETTh1: 3 models × 3 norms × 4 pred_lens (36 runs)"
        echo "  phase3  — ECL/WTH/ILI: Autoformer × 2 norms × 4 pred_lens (24 runs)"
        echo "  phase4  — ETTm2 multi-seed: Autoformer × 3 norms × 4 pred_lens × 5 seeds (60 runs)"
        echo "  phase5  — DishTS init ablation: 3 inits × 4 pred_lens (12 runs)"
        echo "  all     — Run everything (~168 runs, several hours)"
        exit 1
        ;;
esac
