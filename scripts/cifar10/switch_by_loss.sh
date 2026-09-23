#!/usr/bin/env bash
# Prediction/SGD switching on the validation loss (Fig. 7 configuration, lr 0.05, no schedule).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr 0.05 --train_batch_size 256
        --train_epochs 30 --random_seed "$SEED" --predict_start_epoch 10 --predict_epoch_interval 1 --predicted_num 1 --n_past_weights 10)
python scripts/train.py "${COMMON[@]}" --trainer base           --log_dir "$LOG_ROOT/cifar10/switch_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer switch_by_loss --log_dir "$LOG_ROOT/cifar10/switch_by_loss_seed$SEED" "$@"
