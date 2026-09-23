#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name sgd-lr0.1-explr-epoch200 \
    --lr  0.1 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 200 \
    --trainer base \
    --log_dir logs/alexnet/sgd/base-lr0.1-explr \
    --wandb_project_name Prediction-alexnet \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5
    
CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name pred_conf-sgd-lr0.1-explr-epoch200 \
    --lr  0.1 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 200 \
    --trainer predicted_conf \
    --log_dir logs/alexnet/sgd/pred_conf-lr0.1-explr \
    --wandb_project_name Prediction-alexnet \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name pred_conf_adaptive-sgd-lr0.1-explr-epoch200 \
    --lr  0.1 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 200 \
    --trainer predicted_conf_adaptive \
    --log_dir logs/alexnet/sgd/pred_conf-lr0.1-explr \
    --wandb_project_name Prediction-alexnet \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 

# CUDA_VISIBLE_DEVICES=1 python train_test.py \
#     --data_dir ${DATA_DIR} \
#     --dataset cifar10 \
#     --job_name sgd_m-lr0.01-steplr-epoch200 \
#     --lr  0.01 \
#     --model alexnet \
#     --optimizer sgd_m \
#     --random_seed 0 \
#     --svd_mode full \
#     --save_freq 200 \
#     --svd_rank 0 \
#     --train_batch_size 256 \
#     --test_batch_size 256 \
#     --train_epochs 200 \
#     --trainer base \
#     --log_dir logs/alexnet/sgd_m/base-lr0.01-steplr \
#     --wandb_project_name Prediction-alexnet \
#     --wandb_entity ${WANDB_ENTITY} \
#     --predict_start_epoch 5 \
#     --predict_epoch_interval 1 \
#     --predicted_num 5 \
#     --n_past_weights 5

# CUDA_VISIBLE_DEVICES=1 python train_test.py \
#     --data_dir ${DATA_DIR} \
#     --dataset cifar10 \
#     --job_name pred_conf-sgd_m-lr0.01-steplr-epoch200 \
#     --lr  0.01 \
#     --model alexnet \
#     --optimizer sgd_m \
#     --random_seed 0 \
#     --svd_mode full \
#     --save_freq 200 \
#     --svd_rank 0 \
#     --train_batch_size 256 \
#     --test_batch_size 256 \
#     --train_epochs 200 \
#     --trainer predicted_conf \
#     --log_dir logs/alexnet/sgd_m/pred_conf-lr0.01-steplr \
#     --wandb_project_name Prediction-alexnet \
#     --wandb_entity ${WANDB_ENTITY} \
#     --predict_start_epoch 5 \
#     --predict_epoch_interval 1 \
#     --predicted_num 5 \
#     --n_past_weights 5 

