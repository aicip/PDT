#!/usr/bin/env bash
# Memory, runtime and FLOP profiling of PDT vs baseline (Table 7 configuration). Metrics are logged to wandb (--use_wandb).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr 0.05 --train_batch_size 256
        --train_epochs 60 --cosine_lr_sched --lr_min 1e-3 --random_seed "$SEED" --count_flops_fast --profile_memory --profile_timing)
python scripts/train.py "${COMMON[@]}" --trainer base --log_dir "$LOG_ROOT/cifar10/profiling_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/cifar10/profiling_pdt_seed$SEED" "$@"
