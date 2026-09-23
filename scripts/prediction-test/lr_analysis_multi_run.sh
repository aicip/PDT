#!/bin/bash


# Learning Rate vs. Mask Ratio Analysis
# Purpose: Analyze how different learning rates affect mask acceptance ratio and gradient stability
# Metrics: mask ratio, FLOPs, gradient norm, gradient direction stability, loss/accuracy

# lrs=(0.001 0.01 0.05 0.1)
lrs=(0.1)
batch_size=256
random_seeds=(100)  # Multiple runs for statistical significance

# PDT hyperparameters (fixed for all runs)
predict_epoch_interval=1
predict_start_epoch=5
predicted_num=5
n_past_weights=5

echo "=========================================="
echo "Learning Rate vs. Mask Ratio Analysis"
echo "Testing LRs: ${lrs[@]}"
echo "Random seeds: ${random_seeds[@]}"
echo "=========================================="

# # First run baselines for each LR (for comparison)
# echo ""
# echo "==================== Running Baselines ===================="
# for lr in "${lrs[@]}"
# do
#   for random_seed in "${random_seeds[@]}"
#   do
#     job_name="lr_analysis-baseline-lr${lr}-seed${random_seed}"

#     echo "=========================================="
#     echo "Baseline: lr=$lr, seed=$random_seed"
#     echo "Job name: $job_name"
#     echo "=========================================="

#     CUDA_VISIBLE_DEVICES=0 python train_test.py \
#       --data_dir ${DATA_DIR} \
#       --dataset cifar10 \
#       --job_name $job_name \
#       --lr $lr \
#       --model alexnet \
#       --optimizer sgd \
#       --random_seed $random_seed \
#       --svd_mode full \
#       --save_freq 200 \
#       --svd_rank 0 \
#       --train_batch_size $batch_size \
#       --test_batch_size 256 \
#       --train_epochs 60 \
#       --trainer base \
#       --log_dir logs/alexnet/sgd/lr_analysis \
#       --wandb_project_name LR-MaskRatio-Analysis \
#       --wandb_entity ${WANDB_ENTITY} \
#       --count_flops_fast \
#       --log_gradient_metrics \
#       --cosine_lr_sched \
#       --lr_min 1e-3

#     sleep 10
#   done
# done

# Then run PDT for each LR with gradient metrics enabled
echo ""
echo "==================== Running PDT Experiments ===================="
for lr in "${lrs[@]}"
do
  for random_seed in "${random_seeds[@]}"
  do
    job_name="lr_analysis-pdt-lr${lr}-seed${random_seed}"

    echo "=========================================="
    echo "PDT: lr=$lr, seed=$random_seed"
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
      --trainer predicted_conf \
      --log_dir logs/alexnet/sgd/lr_analysis \
      --wandb_project_name LR-MaskRatio-Analysis \
      --wandb_entity ${WANDB_ENTITY} \
      --predict_start_epoch $predict_start_epoch \
      --predict_epoch_interval $predict_epoch_interval \
      --predicted_num $predicted_num \
      --n_past_weights $n_past_weights \
      --count_flops_fast \
      --log_gradient_metrics \
      --cosine_lr_sched \
      --lr_min 1e-3

    sleep 10
  done
done

echo ""
echo "=========================================="
echo "All LR analysis experiments completed!"
echo ""
echo "Total runs: $((${#lrs[@]} * ${#random_seeds[@]} * 2)) (baselines + PDT)"
echo ""
echo "Expected metrics in wandb:"
echo "  - Baselines: flops_fast/*, train_loss, test_loss, accuracy"
echo "  - PDT: mask_ratio_global, gradient/norm, gradient/direction_stability,"
echo "         gradient/direction_angle_deg, flops_fast/*, train_loss, test_loss, accuracy"
echo ""
echo "Next steps:"
echo "  1. Go to https://wandb.ai/${WANDB_ENTITY}/LR-MaskRatio-Analysis"
echo "  2. Compare mask_ratio vs gradient/direction_stability across different LRs"
echo "  3. Analyze FLOPs savings correlation with mask ratio"
echo "  4. Create figures showing LR -> gradient stability -> mask ratio -> efficiency"
echo "=========================================="
