#!/bin/bash

lrs=(0.05) 
predicted_nums=(5) 
batch_sizes=(64 128 256)
random_seeds=(0 100 200 300 400)


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

        job_name="pred_conf-iclr_rebuttal-step${predicted_num}-lr${lr}-batch${batch_size}-seed${seed}"
        
        CUDA_VISIBLE_DEVICES=1 python train_test.py \
        --data_dir ${DATA_DIR} \
        --dataset cifar10 \
        --job_name $job_name \
        --lr  $lr \
        --model alexnet \
        --optimizer sgd \
        --random_seed $seed \
        --svd_mode full \
        --save_freq 100 \
        --svd_rank 0 \
        --train_batch_size $batch_size \
        --test_batch_size 256 \
        --train_epochs 60 \
        --trainer predicted_conf \
        --log_dir logs/alexnet/sgd/pred_conf_iclr \
        --wandb_project_name Prediction-alexnet \
        --wandb_entity ${WANDB_ENTITY} \
        --predict_start_epoch $predict_start_epoch \
        --predict_epoch_interval $predict_epoch_interval \
        --predicted_num $predicted_num \
        --n_past_weights $n_past_weights
        sleep 10
      done
    done
  done
done


# predicted_nums=(1 2 3 4 5)  
# predict_epoch_intervals=(1 2 3 4 5)
# start_epoch_past_weight_pairs=( "10 10" "5 5" "10 5" "15 10" )