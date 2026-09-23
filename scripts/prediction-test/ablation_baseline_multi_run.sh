#!/bin/bash


# Baseline experiments for ablation study comparison
lrs=(0.05 0.01)
batch_size=256
random_seeds=(300 400)

for lr in "${lrs[@]}"
do
  for random_seed in "${random_seeds[@]}"
  do
    job_name="ablation-sgd-baseline-lr${lr}-seed${random_seed}"

    echo "=========================================="
    echo "Running baseline: lr=$lr"
    echo "Job name: $job_name"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=0 python train_test.py \
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
      --trainer base \
      --log_dir logs/alexnet/sgd/ablation_baseline \
      --wandb_project_name PDT-Masking-Ablation \
      --wandb_entity ${WANDB_ENTITY} \
      --cosine_lr_sched \
      --lr_min 1e-3

    sleep 10
  done
done

echo "All baseline experiments completed!"
