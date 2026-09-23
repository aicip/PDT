#!/usr/bin/env bash
# Learning rate x batch size sweep without a schedule (Table 4 configuration). LR and BATCH select one cell.
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
LR="${LR:-0.05}"
BATCH="${BATCH:-256}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr "$LR" --train_batch_size "$BATCH"
        --train_epochs 60 --random_seed "$SEED")
python scripts/train.py "${COMMON[@]}" --trainer base --log_dir "$LOG_ROOT/cifar10/sweep_lr${LR}_bs${BATCH}_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/cifar10/sweep_lr${LR}_bs${BATCH}_pdt_seed$SEED" "$@"
