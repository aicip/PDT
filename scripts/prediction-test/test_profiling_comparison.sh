#!/bin/bash


# Profiling comparison experiment: Baseline vs PDT
# This will allow direct comparison of overhead

echo "=========================================="
echo "Running BASELINE with profiling..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name profiling_baseline_v2-seed200 \
    --lr 0.05 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 200 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --train_epochs 60 \
    --trainer base \
    --log_dir logs/alexnet/sgd/profiling_comparison \
    --wandb_project_name Profiling-Comparison \
    --wandb_entity ${WANDB_ENTITY} \
    --count_flops_fast \
    --profile_timing \
    --profile_memory \
    --cosine_lr_sched \
    --lr_min 1e-3

sleep 10

echo "=========================================="
echo "Running PDT with profiling..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset cifar10 \
    --job_name profiling_pdt_v2-seed200 \
    --lr 0.05 \
    --model alexnet \
    --optimizer sgd \
    --random_seed 200 \
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
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --mask_mode both \
    --count_flops_fast \
    --profile_memory \
    --profile_timing \
    --cosine_lr_sched \
    --lr_min 1e-3

echo "=========================================="
echo "Profiling comparison completed!"
echo "=========================================="
echo ""
echo "Check wandb for the following metrics:"
echo ""
echo "BASELINE metrics:"
echo "  - timing_baseline/epoch_time_s"
echo "  - timing_baseline/avg_batch_time_ms"
echo "  - memory/* (if available)"
echo ""
echo "PDT metrics:"
echo "  - timing_pdt/total_epoch_time_s (full epoch including DMD)"
echo "  - timing_pdt/sgd_only_time_s (just the SGD part)"
echo "  - timing_pdt/dmd_overhead_s (DMD overhead)"
echo "  - timing_pdt/dmd_overhead_ratio (overhead as fraction of total)"
echo "  - timing_pdt/has_dmd (1 if epoch had DMD, 0 otherwise)"
echo "  - timing/svd_ms, timing/prediction_ms, timing/masking_ms (DMD breakdown)"
echo "  - memory/* (all memory metrics)"
echo "  - flops_fast/* (FLOPs metrics)"
echo ""
echo "KEY COMPARISONS TO MAKE:"
echo "  1. Baseline epoch time vs PDT epoch time (without DMD)"
echo "  2. PDT DMD overhead vs potential savings"
echo "  3. Memory overhead of weight snapshots"
