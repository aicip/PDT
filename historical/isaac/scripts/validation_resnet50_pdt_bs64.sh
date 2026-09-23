#!/bin/bash
#SBATCH -J val_bs64_pdt
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/%j_validation_bs64_pdt.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-08:00:00

# MEMORY OVERHEAD VALIDATION EXPERIMENT
# Goal: Test if overhead appears with smaller batch size
# Expected: batch_size=64 -> baseline ~2.8 GB -> PDT ~3.4 GB -> overhead ~21%

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23457

echo "=================================================================================================="
echo "VALIDATION: ResNet-50 PDT with batch_size=64"
echo "Expected PDT peak: ~3.4 GB (baseline 2.8 GB + snapshot 0.6 GB)"
echo "Expected overhead: ~21%"
echo "=================================================================================================="

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name validation_resnet50_pdt_bs64 \
    --lr 0.1 \
    --model resnet50 \
    --optimizer sgd_m \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 10 \
    --svd_rank 0 \
    --train_batch_size 64 \
    --test_batch_size 128 \
    --train_epochs 10 \
    --trainer predicted_conf_adaptive \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --log_dir logs/validation/resnet50_pdt_bs64/ \
    --wandb_project_name isaac-validation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --weight_decay 1e-4 \
    --lr_min 1e-4 \
    --count_flops_fast \
    --profile_memory \
    --profile_timing

echo ""
echo "PDT batch_size=64 completed"
echo "Check memory/peak_mb in wandb and compare with baseline"
