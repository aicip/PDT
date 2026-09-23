#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name ablation_baseline \
    --lr 0.05 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 60 \
    --trainer base \
    --log_dir logs/alexnet/sgd/ablation_baseline \
    --wandb_project_name Masking-Ablation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --lr_min 1e-3

# Baseline: Standard SGD training without prediction
