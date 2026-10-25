#!/usr/bin/env bash
# Parallel E2-E4 grid (leak-free validation only). Two transfer jobs run at a
# time with 3 intra-op threads each; the heavy ResNet50 run gets 6 threads on
# its own. E2-E4 use 6+6 epochs (vs 8+8 in E1) as an explicit CPU-budget cut
# recorded in DECISIONS.md D12/D17.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./venv/bin/python
DATA=Dataset_leakfree
export PYTHONUNBUFFERED=1

LAST_PID=""
run_bg() {
  local tag="$1"; shift
  local threads="$1"; shift
  env OMP_NUM_THREADS="$threads" TF_NUM_INTRAOP_THREADS="$threads" \
      TF_NUM_INTEROP_THREADS=1 "$PY" train.py "$@" \
      > "outputs/logs/${tag}.out" 2>&1 &
  LAST_PID=$!
}

common=(--model transfer --data-dir "$DATA" --head-epochs 6 --ft-epochs 6)

# Matched 6+6 reference for E2-E4. E1's Adam reference is 8+8, so it is not
# used as the baseline for the 6+6 one-factor comparisons.
run_bg e2-ref 3 "${common[@]}" --tag e2-ref --optimizer adam \
  --task E2 --note "reference: head 1e-3, ft 1e-5, 6+6 epochs"
wait "$LAST_PID" || true

# Wave 1
run_bg e2-head1e2 3 "${common[@]}" --tag e2-head1e2 --optimizer adam \
  --head-lr 1e-2 --task E2 --note "head lr 1e-2; ft 1e-5; 6+6 epochs"
p1=$LAST_PID
run_bg e2-head3e4 3 "${common[@]}" --tag e2-head3e4 --optimizer adam \
  --head-lr 3e-4 --task E2 --note "head lr 3e-4; ft 1e-5; 6+6 epochs"
p2=$LAST_PID
wait "$p1" || true; wait "$p2" || true

# Wave 2
run_bg e2-ft1e4 3 "${common[@]}" --tag e2-ft1e4 --optimizer adam \
  --ft-lr 1e-4 --task E2 --note "ft lr 1e-4; head 1e-3; 6+6 epochs"
p1=$LAST_PID
run_bg e2-ft1e6 3 "${common[@]}" --tag e2-ft1e6 --optimizer adam \
  --ft-lr 1e-6 --task E2 --note "ft lr 1e-6; head 1e-3; 6+6 epochs"
p2=$LAST_PID
wait "$p1" || true; wait "$p2" || true

# Wave 3
run_bg e3-unfreeze0 3 "${common[@]}" --tag e3-unfreeze0 --optimizer adam \
  --unfreeze-last 0 --ft-epochs 0 --task E3 --note "head only; 6 epochs"
p1=$LAST_PID
run_bg e3-unfreeze60 3 "${common[@]}" --tag e3-unfreeze60 --optimizer adam \
  --unfreeze-last 60 --task E3 --note "unfreeze last 60; 6+6 epochs"
p2=$LAST_PID
wait "$p1" || true; wait "$p2" || true

# Wave 4
run_bg e3-unfreezeall 3 "${common[@]}" --tag e3-unfreezeall --optimizer adam \
  --unfreeze-last 9999 --task E3 --note "unfreeze all non-BN; 6+6 epochs"
p1=$LAST_PID
run_bg e4-effnet 3 "${common[@]}" --tag e4-effnet --optimizer adam \
  --backbone EfficientNetB0 --task E4 --note "EfficientNetB0; [0,255]; 6+6 epochs"
p2=$LAST_PID
wait "$p1" || true; wait "$p2" || true

# Wave 5: heavy ResNet50 alone
env OMP_NUM_THREADS=6 TF_NUM_INTRAOP_THREADS=6 TF_NUM_INTEROP_THREADS=1 \
  "$PY" train.py "${common[@]}" --tag e4-resnet --optimizer adam \
  --backbone ResNet50 --task E4 \
  --note "ResNet50; approximate RGB mean subtraction; 6+6 epochs" \
  > outputs/logs/e4-resnet.out 2>&1

echo "PARALLEL B GRID DONE"
