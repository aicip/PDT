#!/usr/bin/env bash
# 4-layer LSTM on AG News (downloaded through HuggingFace datasets), SGD baseline (Table 13 configuration).
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
python scripts/train.py --dataset ag_news --model text_lstm --optimizer sgd --lr 0.1 --train_batch_size 128 --train_epochs 30 --cosine_lr_sched --lr_min 1e-2 --max_seq_len 50 --vocab_size 5000 --random_seed "$SEED" --trainer base --log_dir "$LOG_ROOT/ag_news/lstm_baseline_seed$SEED" "$@"
