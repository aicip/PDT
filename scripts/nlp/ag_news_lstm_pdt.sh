#!/usr/bin/env bash
# 4-layer LSTM on AG News, PDT on top of SGD (Table 13 configuration).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train.py --dataset ag_news --model text_lstm --optimizer sgd --lr 0.1 --train_batch_size 128 --train_epochs 30 --cosine_lr_sched --lr_min 1e-2 --max_seq_len 50 --vocab_size 5000 --random_seed "$SEED" --trainer pdt --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/ag_news/lstm_pdt_seed$SEED" "$@"
