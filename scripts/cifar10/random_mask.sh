#!/usr/bin/env bash
# Random-mask prediction baseline vs PDT (Fig. 6 configuration, lr 0.01, tau 3, no schedule).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr 0.01 --train_batch_size 256
        --train_epochs 30 --random_seed "$SEED" --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 3 --n_past_weights 5)
python scripts/train.py "${COMMON[@]}" --trainer base        --log_dir "$LOG_ROOT/cifar10/randmask_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt         --log_dir "$LOG_ROOT/cifar10/randmask_pdt_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer random_mask --random_mask_ratio 0.097 --random_mask_seed "${MASK_SEED:-0}" \
    --log_dir "$LOG_ROOT/cifar10/randmask_random_seed$SEED" "$@"
