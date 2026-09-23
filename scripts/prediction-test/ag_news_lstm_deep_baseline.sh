#!/bin/bash


# AG News + Deep LSTM Baseline
# Expected: Converges in 15-20 epochs (vs 5 for standard LSTM)

cd ${PDT_ROOT}/scripts/prediction-test

echo "=========================================="
echo "AG News Deep LSTM Baseline"
echo "Model: 4-layer LSTM, hidden=512 (~8.3M params)"
echo "Expected: Converges in 15-20 epochs"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset ag_news \
    --job_name nlp_baseline_lstm_deep_sgd_lr0.1-seed0_cosine \
    --lr 0.1 \
    --model lstm_deep \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 0 \
    --svd_rank 0 \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --train_epochs 30 \
    --trainer base \
    --log_dir logs/nlp/baseline_deep \
    --wandb_project_name NLP-Preliminary \
    --wandb_entity ${WANDB_ENTITY} \
    --count_flops_fast \
    --cosine_lr_sched \
    --lr_min 1e-2

echo ""
echo "=========================================="
echo "Baseline experiment completed!"
echo "Check wandb: https://wandb.ai/${WANDB_ENTITY}/NLP-Preliminary"
echo "=========================================="
