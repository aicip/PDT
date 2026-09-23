#!/bin/bash
# SLURM template for the ImageNet experiments: one task per GPU on one node, launched with
# srun (scripts/train.py reads SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID). Edit the
# account, partition, environment activation and the argument line, then
#     sbatch scripts/imagenet/slurm/train.sbatch
# The argument line below is the ResNet-50 PDT configuration; copy the arguments from the
# other scripts in scripts/imagenet/ for the remaining experiments. Batch size is per GPU.
#SBATCH --job-name=pdt
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.log
set -euo pipefail
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
DATA_DIR="${DATA_DIR:-data/imagenet}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
cd "$SLURM_SUBMIT_DIR"
# activate your environment here, e.g. `source activate pdt` or `module load ...`

srun python scripts/train.py --distributed --dataset imagenet --data_dir "$DATA_DIR" --model resnet50 \
    --optimizer sgd_m --lr 0.1 --weight_decay 1e-4 --train_batch_size 600 --test_batch_size 128 \
    --train_epochs 300 --cosine_lr_sched --lr_warmup_epochs 15 --lr_min 1e-4 --two_stage_training --first_stage_epochs 200 \
    --num_workers 16 --random_seed "$SEED" --trainer pdt_distributed \
    --predict_start_epoch 5 --predict_epoch_interval 1 --predicted_num 5 --n_past_weights 5 \
    --log_dir "$LOG_ROOT/imagenet/resnet50_pdt_seed$SEED"
