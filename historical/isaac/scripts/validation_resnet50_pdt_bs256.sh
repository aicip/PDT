#!/bin/bash
#SBATCH -J val_bs256_pdt
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/%j_validation_bs256_pdt.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-08:00:00

# MEMORY OVERHEAD VALIDATION EXPERIMENT
# Expected: batch_size=256 -> baseline ~11 GB -> PDT ~11.6 GB -> overhead ~5%

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23461

echo "=================================================================================================="
echo "VALIDATION: ResNet-50 PDT with batch_size=256"
echo "Expected PDT peak: ~11.6 GB (baseline 11 GB + snapshot 0.6 GB)"
echo "Expected overhead: ~5%"
echo "=================================================================================================="

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name validation_resnet50_pdt_bs256 \
    --lr 0.1 \
    --model resnet50 \
    --optimizer sgd_m \
    --random_seed 0 \
    --train_batch_size 256 \
    --test_batch_size 128 \
    --train_epochs 10 \
    --trainer predicted_conf_adaptive \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --log_dir logs/validation/resnet50_pdt_bs256/ \
    --wandb_project_name isaac-validation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --weight_decay 1e-4 \
    --lr_min 1e-4 \
    --profile_memory \
    --profile_timing

echo "PDT batch_size=256 completed"
