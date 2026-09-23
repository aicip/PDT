#!/bin/bash


# Ablation study: masking strategy under different lr and predicted_num settings
lrs=(0.05 0.01)
predicted_nums=(3 5)
mask_modes=(accel_only consistency_only both)
batch_size=256
random_seeds=(300 400)

predict_epoch_interval=1
predict_start_epoch=5
n_past_weights=5

for lr in "${lrs[@]}"
do
  for predicted_num in "${predicted_nums[@]}"
  do
    for mask_mode in "${mask_modes[@]}"
    do
      for random_seed in "${random_seeds[@]}"
      do
        job_name="ablation-mask_${mask_mode}-lr${lr}-step${predicted_num}-seed${random_seed}"

        echo "=========================================="
        echo "Running: lr=$lr, predicted_num=$predicted_num, mask_mode=$mask_mode"
        echo "Job name: $job_name"
        echo "=========================================="

        CUDA_VISIBLE_DEVICES=1 python train_test.py \
          --data_dir ${DATA_DIR} \
          --dataset cifar10 \
          --job_name $job_name \
          --lr $lr \
          --model alexnet \
          --optimizer sgd \
          --random_seed $random_seed \
          --svd_mode full \
          --save_freq 200 \
          --svd_rank 0 \
          --train_batch_size $batch_size \
          --test_batch_size 256 \
          --train_epochs 60 \
          --trainer predicted_conf \
          --log_dir logs/alexnet/sgd/ablation_masking \
          --wandb_project_name PDT-Masking-Ablation \
          --wandb_entity ${WANDB_ENTITY} \
          --predict_start_epoch $predict_start_epoch \
          --predict_epoch_interval $predict_epoch_interval \
          --predicted_num $predicted_num \
          --n_past_weights $n_past_weights \
          --mask_mode $mask_mode \
          --cosine_lr_sched \
          --lr_min 1e-3

        sleep 10
      done
    done
  done
done

echo "All ablation experiments completed!"
