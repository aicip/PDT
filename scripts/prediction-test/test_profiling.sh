#!/bin/bash


# Test script for all profiling features
# This script tests: --count_flops_fast, --profile_memory, --profile_timing

CUDA_VISIBLE_DEVICES=0 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name profiling_pdt_v2-seed100-int2 \
    --lr 0.05 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 100 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 60 \
    --trainer predicted_conf \
    --log_dir logs/alexnet/sgd/profiling_comparison \
    --wandb_project_name Profiling-Comparison \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 2 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --mask_mode both \
    --count_flops_fast \
    --profile_memory \
    --profile_timing \
    --cosine_lr_sched \
    --lr_min 1e-3

# This will log to wandb:
# - flops_fast/epoch_flops, flops_fast/total_flops
# - flops_fast/dmd_svd_flops, dmd_matrix_flops, dmd_prediction_flops, dmd_total_flops
# - memory/snapshot_storage_mb, svd_workspace_mb, allocated_mb, reserved_mb, peak_mb, dmd_overhead_mb
# - timing/svd_ms, prediction_ms, masking_ms, total_dmd_ms
