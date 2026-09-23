#!/bin/bash


CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --single_class_batch \
    --job_name pred_conf-lr0.1-lrmin_1e-3-single_class_batch \
    --lr  0.1 \
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
    --log_dir logs/alexnet/sgd/predicted_conf \
    --wandb_project_name Prediction-alexnet \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --cosine_lr_sched \
    --lr_min 1e-3


##  optimal learning rate:
##  alexnet: SGD: 0.1, Adam: 0.0005
##  fcn: SGD: 0.01, Adam: 0.0005
##  resnet50: Adam: 0.001

##  trainer: base, predicted_conf, Random_accelerated, predicted_conf_adaptive
## pred_conf-RandomizedDMD-lr0.0005-step5-int5

## --predict_start_epoch 10 \
## --predict_epoch_interval 5 \
## --predicted_num 5 \
## --n_past_weights 10 



