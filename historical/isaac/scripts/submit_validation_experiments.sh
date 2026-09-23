#!/bin/bash
# Submit all validation experiments to ISAAC
# This validates that 0% overhead is due to large baseline (batch_size=600)
# by showing overhead appears with smaller batch sizes

echo "=================================================================================================="
echo "MEMORY OVERHEAD VALIDATION - BATCH SIZE SWEEP"
echo "=================================================================================================="
echo ""
echo "Goal: Validate that 0% overhead for ResNet-50 is due to large baseline memory,"
echo "      not measurement error or over-reservation."
echo ""
echo "Strategy: Run ResNet-50 on ImageNet with different batch sizes"
echo "  - batch_size=64  -> Expected overhead: ~21% (snapshot 0.6GB / baseline 2.8GB)"
echo "  - batch_size=128 -> Expected overhead: ~11% (snapshot 0.6GB / baseline 5.5GB)"
echo "  - batch_size=256 -> Expected overhead: ~5%  (snapshot 0.6GB / baseline 11GB)"
echo "  - batch_size=600 (already done) -> 0% (snapshot 0.6GB / baseline 26GB)"
echo ""
echo "If overhead increases as batch_size decreases, this confirms our hypothesis!"
echo "=================================================================================================="
echo ""

# Create terminal_logs directory if it doesn't exist
mkdir -p terminal_logs

# Submit all experiments
echo "Submitting experiments..."
echo ""

echo "[1/6] Submitting baseline batch_size=64..."
job1=$(sbatch validation_resnet50_baseline_bs64.sh | awk '{print $4}')
echo "  -> Job ID: $job1"

echo "[2/6] Submitting PDT batch_size=64..."
job2=$(sbatch validation_resnet50_pdt_bs64.sh | awk '{print $4}')
echo "  -> Job ID: $job2"

echo "[3/6] Submitting baseline batch_size=128..."
job3=$(sbatch validation_resnet50_baseline_bs128.sh | awk '{print $4}')
echo "  -> Job ID: $job3"

echo "[4/6] Submitting PDT batch_size=128..."
job4=$(sbatch validation_resnet50_pdt_bs128.sh | awk '{print $4}')
echo "  -> Job ID: $job4"

echo "[5/6] Submitting baseline batch_size=256..."
job5=$(sbatch validation_resnet50_baseline_bs256.sh | awk '{print $4}')
echo "  -> Job ID: $job5"

echo "[6/6] Submitting PDT batch_size=256..."
job6=$(sbatch validation_resnet50_pdt_bs256.sh | awk '{print $4}')
echo "  -> Job ID: $job6"

echo ""
echo "=================================================================================================="
echo "ALL JOBS SUBMITTED"
echo "=================================================================================================="
echo ""
echo "Job IDs:"
echo "  Baseline bs=64:  $job1"
echo "  PDT bs=64:       $job2"
echo "  Baseline bs=128: $job3"
echo "  PDT bs=128:      $job4"
echo "  Baseline bs=256: $job5"
echo "  PDT bs=256:      $job6"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  terminal_logs/<job_id>_validation_*.log"
echo ""
echo "Results will be logged to wandb project: isaac-validation"
echo "  Baseline runs: memory_baseline/peak_mb"
echo "  PDT runs: memory/peak_mb"
echo ""
echo "Expected timeline:"
echo "  Baseline (2 epochs): ~2-3 hours each"
echo "  PDT (10 epochs): ~5-6 hours each"
echo "  Total: ~12-16 hours for all experiments"
echo ""
echo "=================================================================================================="
