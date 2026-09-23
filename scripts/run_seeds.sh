#!/usr/bin/env bash
# Run one of the experiment scripts for several seeds: bash scripts/run_seeds.sh scripts/cifar10/alexnet_pdt.sh [seeds...]
# Environment: DATA_DIR (data root), LOG_ROOT (output root), SEED (random seed).
# Extra arguments are passed through to scripts/train.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${DATA_DIR:-data}"
LOG_ROOT="${LOG_ROOT:-logs}"
SEED="${SEED:-0}"
SCRIPT="$1"; shift
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 100 200 300 400)
for s in "${SEEDS[@]}"; do
    SEED="$s" bash "$SCRIPT"
done
