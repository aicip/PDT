#!/usr/bin/env bash
# AlexNet on CIFAR-10, SGD baseline (Fig. 4 / Fig. 10b / Table 5 configuration, lr 0.05).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train.py --dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --train_batch_size 256 --train_epochs 60 --random_seed "$SEED" --trainer base --lr 0.05 --cosine_lr_sched --lr_min 1e-3 \
    --log_dir "$LOG_ROOT/cifar10/alexnet_baseline_seed$SEED" "$@"
