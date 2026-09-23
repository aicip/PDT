#!/bin/bash
#SBATCH -J val_bs256_base
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/%j_validation_bs256_baseline.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-06:00:00

# MEMORY OVERHEAD VALIDATION EXPERIMENT
# Expected: batch_size=256 -> baseline ~11 GB -> overhead ~5% (snapshot 0.6GB / 11GB)

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23460

echo "=================================================================================================="
echo "VALIDATION: ResNet-50 Baseline with batch_size=256"
echo "Expected baseline peak: ~11 GB"
echo "=================================================================================================="

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name validation_resnet50_baseline_bs256 \
    --lr 0.1 \
    --model resnet50 \
    --optimizer sgd_m \
    --random_seed 0 \
    --train_batch_size 256 \
    --test_batch_size 128 \
    --train_epochs 2 \
    --trainer base \
    --log_dir logs/validation/resnet50_baseline_bs256/ \
    --wandb_project_name isaac-validation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --weight_decay 1e-4 \
    --lr_min 1e-4 \
    --profile_memory \
    --profile_timing

echo "Baseline batch_size=256 completed"
