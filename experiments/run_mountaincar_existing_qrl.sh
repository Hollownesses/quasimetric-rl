#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_DIR="${OUT_DIR:-results/experiments/mountaincar_existing_qrl}"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/eval"
STEPS="${STEPS:-50000}"
EPISODES="${EPISODES:-1019}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"

"${PYTHON_BIN}" minimal_qrl/train.py \
  --env-type mountaincar \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --output-dir "${TRAIN_DIR}" \
  --num-episodes "${EPISODES}" \
  --max-steps-per-episode 250 \
  --batch-size "${BATCH_SIZE}" \
  --total-steps "${STEPS}" \
  --log-interval 100 \
  --save-interval 5000 \
  --eval-interval 5000 \
  --eval-n-pairs 500 \
  --visualization-interval 0 \
  --planning-eval-interval 0 \
  --mountaincar-dataset-mode random_policy_paper \
  --mountaincar-abstract-goal-transition-repeats 15 \
  --mountaincar-gt-pos-bins 160 \
  --mountaincar-gt-vel-bins 160 \
  --mountaincar-gt-goal-mode threshold

"${PYTHON_BIN}" experiments/qrl_checkpoint_value_eval.py \
  --env-type mountaincar \
  --checkpoint "${TRAIN_DIR}/checkpoint_final.pth" \
  --output-dir "${EVAL_DIR}" \
  --device cpu \
  --seed "${SEED}" \
  --gt-pos-bins 160 \
  --gt-vel-bins 160 \
  --gt-goal-mode threshold \
  --goals "0.50,0.0,1.0"

echo "TensorBoard:"
echo "  $PYTHON_BIN -m tensorboard.main --logdir=$TRAIN_DIR/tensorboard"
