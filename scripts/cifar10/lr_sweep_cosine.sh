#!/usr/bin/env bash
# Learning rate sweep with cosine annealing (Table 5 / Table 11 configuration). LR selects the learning rate.
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
LR="${LR:-0.05}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr "$LR" --train_batch_size 256
        --train_epochs 60 --cosine_lr_sched --lr_min 1e-3 --random_seed "$SEED" --log_gradient_metrics)
python scripts/train.py "${COMMON[@]}" --trainer base --log_dir "$LOG_ROOT/cifar10/cosine_lr${LR}_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/cifar10/cosine_lr${LR}_pdt_seed$SEED" "$@"
