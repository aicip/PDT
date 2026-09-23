#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name ablation_accel_only \
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
    --log_dir logs/alexnet/sgd/ablation_accel_only \
    --wandb_project_name Masking-Ablation \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --mask_mode accel_only \
    --cosine_lr_sched \
    --lr_min 1e-3

# PDT with only acceleration criterion (Eq 6): lower and upper bounds
