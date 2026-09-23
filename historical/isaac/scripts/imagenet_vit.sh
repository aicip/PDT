#!/bin/bash 
#SBATCH -J imnet
#SBATCH --nodes=1
#SBATCH -A <account>
#SBATCH --partition=<partition>
#SBATCH -o %j.log
#SBATCH --qos=<qos>
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --reservation=hqi_paper

export PYTHONPATH=$PYTHONPATH:${PDT_ROOT}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23401

srun singularity exec --nv ${CONTAINER_SIF} \
    python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset imagenet \
    --job_name vit_pred \
    --lr  0.0015 \
    --model vit \
    --optimizer adamw \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 1000 \
    --svd_rank 0 \
    --train_batch_size 600 \
    --test_batch_size 100 \
    --train_epochs 200 \
    --trainer predicted_conf_adaptive \
    --log_dir logs/alexnet/pred_conf_test \
    --wandb_project_name prediction1 \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --distributed \
    --cosine_lr_sched \
    --weight_decay 0.3 \
    --lr_warmup_epochs 30 \
    --w_device cuda:3
