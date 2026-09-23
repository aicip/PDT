#!/bin/bash

lrs=(0.01)  
predicted_nums=(3 5)  
predict_epoch_intervals=(1 3)
start_epoch_past_weight_pairs=( "5 5" )

for lr in "${lrs[@]}"
do
  for predicted_num in "${predicted_nums[@]}"
  do
    for predict_epoch_interval in "${predict_epoch_intervals[@]}"
    do
      for pair in "${start_epoch_past_weight_pairs[@]}"
      do
        IFS=' ' read -r -a pair_arr <<< "$pair"
        predict_start_epoch=${pair_arr[0]}
        n_past_weights=${pair_arr[1]}

        job_name="pred_conf-cos_lrmin_1e-3-lr${lr}-step${predicted_num}-int${predict_epoch_interval}-start${predict_start_epoch}-past${n_past_weights}"
        
        CUDA_VISIBLE_DEVICES=1 python train_test.py \
        --data_dir ${DATA_DIR} \
        --dataset cifar10 \
        --job_name $job_name \
        --lr  $lr \
        --model fcn \
        --optimizer sgd \
        --random_seed 0 \
        --svd_mode full \
        --save_freq 200 \
        --svd_rank 0 \
        --train_batch_size 256 \
        --test_batch_size 256 \
        --train_epochs 100 \
        --trainer predicted_conf \
        --log_dir logs/alexnet/sgd/pred_conf_test \
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