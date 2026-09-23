#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name ablation_both \
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
    --trainer predicted_conf \
    --log_dir logs/alexnet/sgd/ablation_both \
    --wandb_project_name Masking-Ablation \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --mask_mode both \
    --cosine_lr_sched \
    --lr_min 1e-3

# Full PDT with both criteria (Eq 6 + Eq 7): acceleration + consistency
