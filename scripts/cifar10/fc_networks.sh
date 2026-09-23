#!/usr/bin/env bash
# Fully connected networks of Fig. 2: SGD, non-selective prediction and PDT for 2/4/6 layers (FC_LAYERS=2,4,6).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
FC_LAYERS="${FC_LAYERS:-4}"
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model fcnet --fc_layers "$FC_LAYERS" --optimizer sgd --lr 0.01
        --train_batch_size 256 --train_epochs 30 --random_seed "$SEED"
        --predict_start_epoch 5 --predict_epoch_interval 3 --predicted_num 5 --n_past_weights 5)
python scripts/train.py "${COMMON[@]}" --trainer base         --log_dir "$LOG_ROOT/cifar10/fc${FC_LAYERS}_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer nonselective --log_dir "$LOG_ROOT/cifar10/fc${FC_LAYERS}_nonselective_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt          --log_dir "$LOG_ROOT/cifar10/fc${FC_LAYERS}_pdt_seed$SEED" "$@"
