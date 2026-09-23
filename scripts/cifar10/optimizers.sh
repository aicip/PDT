#!/usr/bin/env bash
# PDT as a plug-in for different base optimizers (Table 2 configuration). OPTIMIZER selects one of them.
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
OPTIMIZER="${OPTIMIZER:-adamw}"
case "$OPTIMIZER" in
    sgd)     LR=0.1;    LR_MIN=1e-3 ;;
    sgd_m)   LR=0.001;  LR_MIN=1e-4 ;;
    adam)    LR=0.0005; LR_MIN=1e-5 ;;
    adamw)   LR=5e-5;   LR_MIN=1e-6 ;;
    rmsprop) LR=1e-4;   LR_MIN=1e-6 ;;
    shampoo) LR=1e-3;   LR_MIN=1e-6 ;;
    lamb)    LR=1e-3;   LR_MIN=1e-6 ;;
    *) echo "unknown OPTIMIZER=$OPTIMIZER"; exit 1 ;;
esac
COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --optimizer "$OPTIMIZER" --lr "$LR" --weight_decay 1e-4
        --train_batch_size 256 --train_epochs 60 --cosine_lr_sched --lr_min "$LR_MIN" --random_seed "$SEED")
python scripts/train.py "${COMMON[@]}" --trainer base --log_dir "$LOG_ROOT/cifar10/opt_${OPTIMIZER}_baseline_seed$SEED" "$@"
python scripts/train.py "${COMMON[@]}" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/cifar10/opt_${OPTIMIZER}_pdt_seed$SEED" "$@"
