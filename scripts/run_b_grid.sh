#!/usr/bin/env bash
# E1-E4 optimisation grid on the leak-free split (validation only).
# Test split is NOT touched here; final configs are evaluated later.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
DATA=Dataset_leakfree
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-5}
export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-5}
export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
export PYTHONUNBUFFERED=1

common=(--model transfer --data-dir "$DATA" --head-epochs 8 --ft-epochs 8)

# E1: optimiser (Adam reference = default LRs)
$PY train.py "${common[@]}" --tag e1-adam --optimizer adam --task E1 \
  --note "reference: Adam head 1e-3 ft 1e-5, unfreeze 30" \
  > outputs/logs/e1-adam.out 2>&1
$PY train.py "${common[@]}" --tag e1-adamw --optimizer adamw --task E1 \
  --note "AdamW Keras default weight decay 0.004" \
  > outputs/logs/e1-adamw.out 2>&1
$PY train.py "${common[@]}" --tag e1-sgd --optimizer sgd --head-lr 1e-2 \
  --ft-lr 1e-3 --task E1 --note "SGD Nesterov 0.9, head 1e-2 ft 1e-3" \
  > outputs/logs/e1-sgd.out 2>&1
$PY train.py "${common[@]}" --tag e1-rmsprop --optimizer rmsprop --task E1 \
  --note "RMSprop head 1e-3 ft 1e-5" \
  > outputs/logs/e1-rmsprop.out 2>&1

# E2: one-factor-at-a-time learning rates (Adam, reference = e1-adam)
$PY train.py "${common[@]}" --tag e2-head1e2 --optimizer adam --head-lr 1e-2 \
  --task E2 --note "head lr 1e-2; ft lr default 1e-5" \
  > outputs/logs/e2-head1e2.out 2>&1
$PY train.py "${common[@]}" --tag e2-head3e4 --optimizer adam --head-lr 3e-4 \
  --task E2 --note "head lr 3e-4; ft lr default 1e-5" \
  > outputs/logs/e2-head3e4.out 2>&1
$PY train.py "${common[@]}" --tag e2-ft1e4 --optimizer adam --ft-lr 1e-4 \
  --task E2 --note "ft lr 1e-4; head lr default 1e-3" \
  > outputs/logs/e2-ft1e4.out 2>&1
$PY train.py "${common[@]}" --tag e2-ft1e6 --optimizer adam --ft-lr 1e-6 \
  --task E2 --note "ft lr 1e-6; head lr default 1e-3" \
  > outputs/logs/e2-ft1e6.out 2>&1

# E3: fine-tune depth (Adam, reference = e1-adam for last 30)
$PY train.py "${common[@]}" --tag e3-unfreeze0 --optimizer adam \
  --unfreeze-last 0 --ft-epochs 0 --task E3 \
  --note "head only, no fine-tune" > outputs/logs/e3-unfreeze0.out 2>&1
$PY train.py "${common[@]}" --tag e3-unfreeze60 --optimizer adam \
  --unfreeze-last 60 --task E3 --note "unfreeze last 60, BN frozen" \
  > outputs/logs/e3-unfreeze60.out 2>&1
$PY train.py "${common[@]}" --tag e3-unfreezeall --optimizer adam \
  --unfreeze-last 9999 --task E3 --note "unfreeze all non-BN layers" \
  > outputs/logs/e3-unfreezeall.out 2>&1

# E4: backbone (Adam, reference = e1-adam for MobileNetV2)
$PY train.py "${common[@]}" --tag e4-effnet --optimizer adam \
  --backbone EfficientNetB0 --task E4 \
  --note "EfficientNetB0, [0,255] preprocessing (internal 1/255)" \
  > outputs/logs/e4-effnet.out 2>&1
$PY train.py "${common[@]}" --tag e4-resnet --optimizer adam \
  --backbone ResNet50 --task E4 \
  --note "ResNet50, approximate RGB mean subtraction (no BGR flip)" \
  > outputs/logs/e4-resnet.out 2>&1

echo "B GRID DONE"
