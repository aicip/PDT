#!/usr/bin/env bash
# ViT-Huge/14 on ImageNet-1k, AdamW baseline (the paper used 6 H100 on two nodes, 128 images per GPU). Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT.
# One process per GPU via torchrun. NGPUS is the number of GPUs on this node; the batch size
# below is per GPU (the paper used 600 per GPU on three H100, i.e. a global batch of 1800).
# Environment: DATA_DIR (ImageNet root with train/ and val/), LOG_ROOT, SEED, NGPUS.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data/imagenet}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
NGPUS="${NGPUS:-3}"
torchrun --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" --nproc_per_node="$NGPUS" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
    scripts/train.py --distributed --dataset imagenet --data_dir "$DATA_DIR" --model vit_huge --optimizer adamw --lr 0.0005 --weight_decay 0.1 --train_batch_size 128 --test_batch_size 64 --train_epochs 200 --two_stage_training --first_stage_epochs 180 --lr_warmup_epochs 15 --lr_min 1e-6 --num_workers 16 --random_seed "$SEED" --trainer base --log_dir "$LOG_ROOT/imagenet/vit_huge_baseline_seed$SEED" "$@"
