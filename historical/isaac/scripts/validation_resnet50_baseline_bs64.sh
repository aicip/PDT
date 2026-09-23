#!/bin/bash
#SBATCH -J val_bs64_base
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/%j_validation_bs64_baseline.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-06:00:00

# MEMORY OVERHEAD VALIDATION EXPERIMENT
# Goal: Test if overhead appears with smaller batch size
# Expected: batch_size=64 -> baseline ~2.8 GB -> overhead ~21% (snapshot 0.6GB / 2.8GB)

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

echo "=================================================================================================="
echo "VALIDATION: ResNet-50 Baseline with batch_size=64"
echo "Expected baseline peak: ~2.8 GB"
echo "=================================================================================================="

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name validation_resnet50_baseline_bs64 \
    --lr 0.1 \
    --model resnet50 \
    --optimizer sgd_m \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 10 \
    --svd_rank 0 \
    --train_batch_size 64 \
    --test_batch_size 128 \
    --train_epochs 2 \
    --trainer base \
    --log_dir logs/validation/resnet50_baseline_bs64/ \
    --wandb_project_name isaac-validation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --weight_decay 1e-4 \
    --lr_min 1e-4 \
    --count_flops_fast \
    --profile_memory \
    --profile_timing

echo ""
echo "Baseline batch_size=64 completed"
echo "Check memory_baseline/peak_mb in wandb"
