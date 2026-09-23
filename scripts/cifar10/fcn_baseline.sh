#!/usr/bin/env bash
# FCN (3.9M parameters) on CIFAR-10, SGD baseline (Fig. 10a / Table 1 configuration).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train.py --dataset cifar10 --data_dir "$DATA_DIR" --model fcn --optimizer sgd --lr 0.01 \
    --train_batch_size 256 --train_epochs 100 --cosine_lr_sched --lr_min 1e-3 --random_seed "$SEED" \
    --trainer base --log_dir "$LOG_ROOT/cifar10/fcn_baseline_seed$SEED" "$@"
