#!/bin/bash
#SBATCH -J imnet
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o terminal_logs/%j.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23401

# NOTE: This script uses --train_epochs 300 to match the LR schedule of full training
# but is intended to be manually stopped after 20 epochs using: scancel <job_id>
# This ensures the LR schedule matches previous 300-epoch experiments

srun singularity exec --nv ${CONTAINER_SIF} \
    python ${PDT_ROOT}/scripts/prediction-test/train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name profiling_resnet50_pdt_300ep \
    --lr 0.1 \
    --model resnet50 \
    --optimizer sgd_m \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 10 \
    --svd_rank 0 \
    --train_batch_size 600 \
    --test_batch_size 128 \
    --train_epochs 300 \
    --trainer predicted_conf_adaptive \
    --log_dir logs/profiling/resnet50_pdt_300ep/ \
    --wandb_project_name isaac-prediction \
    --wandb_entity ${WANDB_ENTITY} \
    --distributed \
    --cosine_lr_sched \
    --weight_decay 1e-4 \
    --lr_warmup_epochs 15 \
    --lr_min 1e-4 \
    --two_stage_training \
    --first_stage_epochs 200 \
    --count_flops_fast \
    --profile_memory \
    --profile_timing \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --w_device cuda:3
