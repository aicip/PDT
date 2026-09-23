#!/usr/bin/env bash
# AlexNet on CIFAR-10, PDT on top of SGD (Fig. 4 / Fig. 10b / Table 5 configuration, lr 0.05).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train.py --dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --train_batch_size 256 --train_epochs 60 --random_seed "$SEED" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --lr 0.05 --cosine_lr_sched --lr_min 1e-3 \
    --log_dir "$LOG_ROOT/cifar10/alexnet_pdt_seed$SEED" "$@"
