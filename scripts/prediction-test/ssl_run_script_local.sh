#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python train_test_ssl.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name simsiam-resnet18-sgd-lr0.03-epoch200-cosine-lrmin_1e-3-test3 \
    --ssl_method simsiam \
    --backbone resnet18 \
    --projection_dim 512 \
    --prediction_dim 128 \
    --lr 0.03 \
    --optimizer sgd_m \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --random_seed 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 200 \
    --trainer base \
    --log_dir logs/simsiam/sgd \
    --wandb_project_name SSL-simsiam \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --lr_min 1e-3 \
    --eval_freq 10 \
    --save_weights \
    --save_freq 50


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



