#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-results/experiments/maze2d_existing_qrl}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/eval"
STEPS="${STEPS:-20000}"
EPISODES="${EPISODES:-1500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
EVAL_PAIRS="${EVAL_PAIRS:-200}"

"${PYTHON_BIN}" minimal_qrl/train.py \
  --env-type maze2d \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --output-dir "${TRAIN_DIR}" \
  --grid-size 15 15 \
  --num-episodes "${EPISODES}" \
  --max-steps-per-episode 120 \
  --batch-size "${BATCH_SIZE}" \
  --total-steps "${STEPS}" \
  --log-interval 100 \
  --save-interval 5000 \
  --eval-interval 5000 \
  --eval-n-pairs 500 \
  --visualization-interval 5000 \
  --planning-eval-interval 0

"${PYTHON_BIN}" experiments/qrl_checkpoint_value_eval.py \
  --env-type maze2d \
  --checkpoint "${TRAIN_DIR}/checkpoint_final.pth" \
  --output-dir "${EVAL_DIR}" \
  --device cpu \
  --seed "${SEED}" \
  --grid-size 15 15 \
  --eval-pairs "${EVAL_PAIRS}" \
  --max-steps-per-episode 120 \
  --lookahead-horizon 4

echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$TRAIN_DIR/tensorboard"