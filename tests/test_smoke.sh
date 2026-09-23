#!/usr/bin/env bash
# Smoke test: every trainer runs a few short epochs on CIFAR-10 and the prediction
# step is actually triggered (start epoch 2, history 2, tau 2). Takes a few minutes on
# one GPU. Usage: DATA_DIR=/path/to/data bash tests/test_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_DIR="${DATA_DIR:-data}"
OUT="${OUT:-logs/smoke}"
rm -rf "$OUT"

COMMON=(--dataset cifar10 --data_dir "$DATA_DIR" --model alexnet --img_size 64 --optimizer sgd --lr 0.05
        --train_batch_size 256 --train_epochs 4 --cosine_lr_sched --lr_min 1e-3
        --predict_start_epoch 2 --n_past_weights 2 --predicted_num 2 --predict_epoch_interval 1 --log_interval 1000)

check() {  # <log_dir> <expect_prediction>
    local csv="$1/metrics.csv"
    [ -f "$csv" ] || { echo "FAIL: $csv missing"; exit 1; }
    python - "$csv" "$2" <<'EOF'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 4, f"expected 4 epochs, got {len(rows)}"
for r in rows:
    assert math.isfinite(float(r["train_loss"])) and math.isfinite(float(r["test_loss"])), "non-finite loss"
if sys.argv[2] == "1":
    assert any(r["prediction_epoch"] == "1" for r in rows), "no prediction epoch recorded"
print(f"  OK: {len(rows)} epochs, final accuracy {float(rows[-1]['accuracy']):.3f}")
EOF
}

for trainer in base pdt random_accelerated random_mask nonselective switch_by_loss; do
    echo "== trainer: $trainer"
    python scripts/train.py "${COMMON[@]}" --trainer "$trainer" --log_dir "$OUT/$trainer" > "$OUT.$trainer.log" 2>&1 \
        || { echo "FAIL: see $OUT.$trainer.log"; tail -20 "$OUT.$trainer.log"; exit 1; }
    expect=1; [ "$trainer" = base ] && expect=0; [ "$trainer" = switch_by_loss ] && expect=0
    check "$OUT/$trainer" "$expect"
done

echo "== masking ablation flags"
python scripts/train.py "${COMMON[@]}" --trainer pdt --mask_mode accel_only --save_masks --log_dir "$OUT/accel_only" > "$OUT.accel_only.log" 2>&1
check "$OUT/accel_only" 1
ls "$OUT/accel_only/masks" | head -1 > /dev/null || { echo "FAIL: no mask file written"; exit 1; }

echo "== profiling flags"
python scripts/train.py "${COMMON[@]}" --trainer pdt --count_flops_fast --profile_memory --profile_timing --log_dir "$OUT/profiling" > "$OUT.profiling.log" 2>&1
check "$OUT/profiling" 1

echo "== fully connected network (Figure 2 configuration)"
python scripts/train.py "${COMMON[@]}" --model fcnet --fc_layers 2 --img_size 128 --lr 0.01 --trainer nonselective --log_dir "$OUT/fcnet" > "$OUT.fcnet.log" 2>&1
check "$OUT/fcnet" 1

echo "smoke test passed"
