#!/usr/bin/env bash
# Masking-criterion ablation (Table 10 / Fig. 14 configuration): MASK_MODE in both, accel_only, consistency_only; TAU in 3, 5; LR in 0.01, 0.05.
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
MASK_MODE="${MASK_MODE:-both}"
TAU="${TAU:-5}"
LR="${LR:-0.05}"
python scripts/train.py --dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer sgd --lr "$LR" \
    --train_batch_size 256 --train_epochs 60 --cosine_lr_sched --lr_min 1e-3 --random_seed "$SEED" \
    --trainer pdt --mask_mode "$MASK_MODE" --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num "$TAU" --n_past_weights 5 \
    --log_dir "$LOG_ROOT/cifar10/ablation_${MASK_MODE}_tau${TAU}_lr${LR}_seed$SEED" "$@"
