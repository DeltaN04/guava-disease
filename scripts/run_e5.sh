#!/usr/bin/env bash
# E5: Model A optimiser/schedule comparison on the leak-free split.
# 8-epoch budget (task cap is 12; the cut is recorded in DECISIONS.md D13).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
DATA=Dataset_leakfree
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-10}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export PYTHONUNBUFFERED=1

common=(--model custom --data-dir "$DATA" --epochs-custom 8 \
  --es-monitor val_accuracy --es-patience 8 --task E5)

$PY train.py "${common[@]}" --tag e5-adam --optimizer adam \
  --custom-lr 1e-3 --lr-schedule plateau \
  --note "Adam, ReduceLROnPlateau, 8 epochs" \
  > outputs/logs/e5-adam.out 2>&1
$PY train.py "${common[@]}" --tag e5-sgd --optimizer sgd \
  --custom-lr 1e-2 --lr-schedule plateau \
  --note "SGD Nesterov 0.9, lr 1e-2, ReduceLROnPlateau, 8 epochs" \
  > outputs/logs/e5-sgd.out 2>&1
$PY train.py "${common[@]}" --tag e5-adam-cosine --optimizer adam \
  --custom-lr 1e-3 --lr-schedule cosine \
  --note "Adam, cosine decay, 8 epochs" \
  > outputs/logs/e5-adam-cosine.out 2>&1

echo "E5 DONE"
