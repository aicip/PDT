#!/usr/bin/env bash
# SimSiam pre-training on CIFAR-10 with a ResNet-18 backbone, PDT (Table 3 configuration).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train_ssl.py --dataset cifar10 --data_dir "$DATA_DIR" --ssl_method simsiam --backbone resnet18 --projection_dim 512 --prediction_dim 128 --optimizer sgd_m --lr 0.03 --momentum 0.9 --weight_decay 1e-4 --train_batch_size 256 --train_epochs 200 --cosine_lr_sched --lr_min 1e-3 --eval_freq 10 --random_seed "$SEED" --trainer predicted --svd_rank 0 --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/simsiam/pdt_seed$SEED" "$@"
