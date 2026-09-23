#!/bin/bash
#SBATCH -J vit_val_baseline
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/vit_baseline_bs256_%j.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-16:00:00
#SBATCH --mem=128G

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456

echo "======================================================================"
echo "ViT-Base + ImageNet Validation (Baseline, bs=256)"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "NOTE: Single GPU, batch_size=256 (reduced from 600 in original)"
echo "NOTE: LR scaled from 0.0015 to 0.0006 (256/600 ratio)"
echo "======================================================================"

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name validation_vit_base_baseline_bs256 \
    --model vit \
    --optimizer adamw \
    --lr 0.0006 \
    --weight_decay 0.05 \
    --random_seed 0 \
    --svd_mode full \
    --svd_rank 0 \
    --train_batch_size 256 \
    --test_batch_size 128 \
    --train_epochs 10 \
    --trainer base \
    --log_dir logs/validation/vit_base_baseline_bs256/ \
    --wandb_project_name isaac-validation \
    --wandb_entity ${WANDB_ENTITY} \
    --cosine_lr_sched \
    --lr_warmup_epochs 5 \
    --lr_min 1e-6 \
    --profile_memory \
    --profile_timing

echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "======================================================================"
