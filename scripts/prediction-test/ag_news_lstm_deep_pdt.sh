#!/bin/bash


# AG News + Deep LSTM + PDT
# Expected: Similar accuracy to baseline, but faster wall-clock time

cd ${PDT_ROOT}/scripts/prediction-test

echo "=========================================="
echo "AG News Deep LSTM + PDT"
echo "Model: 4-layer LSTM, hidden=512 (~8.3M params)"
echo "Expected: Accelerates training vs baseline"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python train_test.py \
    --data_dir ${DATA_DIR} \
    --dataset ag_news \
    --job_name nlp_pdt_lstm_deep_sgd_lr0.1-seed0_cosine \
    --lr 0.1 \
    --model lstm_deep \
    --optimizer sgd \
    --random_seed 0 \
    --svd_mode full \
    --save_freq 200 \
    --svd_rank 0 \
    --train_batch_size 128 \
    --test_batch_size 128 \
    --train_epochs 30 \
    --trainer predicted_conf \
    --log_dir logs/nlp/pdt_deep \
    --wandb_project_name NLP-Preliminary \
    --wandb_entity ${WANDB_ENTITY} \
    --predict_start_epoch 5 \
    --predict_epoch_interval 1 \
    --predicted_num 5 \
    --n_past_weights 5 \
    --count_flops_fast \
    --cosine_lr_sched \
    --lr_min 1e-2

echo ""
echo "=========================================="
echo "PDT experiment completed!"
echo "Check wandb: https://wandb.ai/${WANDB_ENTITY}/NLP-Preliminary"
echo "=========================================="
