#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name sgd-lr0.05-seed_1 \
    --lr  0.05 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 1 \
    --svd_mode full \
    --save_freq 30 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 30 \
    --trainer base \
    --log_dir logs/alexnet/sgd/predicted \
    --wandb_project_name Prediction-alexnet \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 10 \
    --predict_epoch_interval 1 \
    --predicted_num 1 \
    --n_past_weights 10 
    #--if_LV True 

##  optimal learning rate:
##  alexnet: SGD: 0.1, Adam: 0.0005
##  fcn: SGD: 0.01, Adam: 0.0005
##  resnet50: Adam: 0.001

##  trainer: base, predicted_conf, Random_accelerated, predicted_conf_adaptive
## pred_conf-RandomizedDMD-lr0.0005-step5-int5
## pred_switch_by_loss-lr0.05-step1-start10-past10-seed_1 

## --predict_start_epoch 10 \
## --predict_epoch_interval 5 \
## --predicted_num 5 \
## --n_past_weights 10 



