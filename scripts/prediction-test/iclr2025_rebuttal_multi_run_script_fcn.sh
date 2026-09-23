#!/bin/bash

lrs=(0.01) 
predicted_nums=(3) 
batch_sizes=(256)
random_seeds=(100 200)

predict_epoch_interval=1
predict_start_epoch=5
n_past_weights=5


for predicted_num in "${predicted_nums[@]}"
do
  for lr in "${lrs[@]}"
  do
    for batch_size in "${batch_sizes[@]}"
    do
      for seed in "${random_seeds[@]}"
      do

        job_name="sgd-iclr_rebuttal-lr_min_1e-3-step${predicted_num}-lr${lr}-batch${batch_size}-seed${seed}"
        
        CUDA_VISIBLE_DEVICES=0 python train_test.py \
        --data_dir ${DATA_DIR} \
        --dataset cifar10 \
        --job_name $job_name \
        --lr  $lr \
        --model fcn \
        --optimizer sgd \
        --random_seed $seed \
        --svd_mode full \
        --save_freq 200 \
        --svd_rank 0 \
        --train_batch_size $batch_size \
        --test_batch_size 256 \
        --train_epochs 100 \
        --trainer base \
        --log_dir logs/alexnet/sgd/pred_conf_iclr_rebuttal \
        --wandb_project_name Prediction-fcn \
        --wandb_entity ${WANDB_ENTITY} \
        --predict_start_epoch $predict_start_epoch \
        --predict_epoch_interval $predict_epoch_interval \
        --predicted_num $predicted_num \
        --n_past_weights $n_past_weights \
        --cosine_lr_sched \
        --lr_min 1e-3 
        sleep 10
      done
    done
  done
done


# predicted_nums=(1 2 3 4 5)  
# predict_epoch_intervals=(1 2 3 4 5)
# start_epoch_past_weight_pairs=( "10 10" "5 5" "10 5" "15 10" )