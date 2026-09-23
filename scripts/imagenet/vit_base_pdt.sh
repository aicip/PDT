#!/usr/bin/env bash
# ViT-Base/16 on ImageNet-1k, PDT (distributed trainer with the acceleration scheduler).
# One process per GPU via torchrun. NGPUS is the number of GPUs on this node; the batch size
# below is per GPU (the paper used 600 per GPU on three H100, i.e. a global batch of 1800).
# Environment: DATA_DIR (ImageNet root with train/ and val/), LOG_ROOT, SEED, NGPUS.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data/imagenet}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
NGPUS="${NGPUS:-3}"
torchrun --nproc_per_node="$NGPUS" scripts/train.py --distributed --dataset imagenet --data_dir "$DATA_DIR" --model vit_base --optimizer adamw --lr 0.0015 --weight_decay 0.05 --train_batch_size 600 --test_batch_size 128 --train_epochs 200 --cosine_lr_sched --lr_warmup_epochs 15 --lr_min 1e-6 --num_workers 16 --random_seed "$SEED" --trainer pdt_distributed --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 --log_dir "$LOG_ROOT/imagenet/vit_base_pdt_seed$SEED" "$@"
