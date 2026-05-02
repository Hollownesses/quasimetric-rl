# Thesis value-structure experiments

This note documents the MountainCar and maze2D thesis experiments. Training
uses the existing QRL implementation in `minimal_qrl/train.py`
(`QRLConf`, `QRLAgent`, `QuasimetricCriticLosses`); the experiment scripts here
only wrap training and evaluate saved checkpoints.

## MountainCar

Run the full training + checkpoint evaluation:

```bash
experiments/run_mountaincar_existing_qrl.sh
```

Main outputs:

- `results/experiments/mountaincar_existing_qrl/train/checkpoint_final.pth`: trained existing-QRL checkpoint.
- `results/experiments/mountaincar_existing_qrl/train/train_metrics.csv`: scalar losses exported from the existing QRL training loop.
- `results/experiments/mountaincar_existing_qrl/eval/mountaincar_heatmap_goal_*.png`: learned QRL distance field and graph shortest-step field for each goal.
- `results/experiments/mountaincar_existing_qrl/eval/distance_metrics.csv`: Position distance, Euclidean distance, and QRL distance compared with discretized graph shortest steps using MSE, MAE, Pearson, and Spearman.

The default setup follows the paper-style MountainCar reproduction path used in
this repo: states are snapped to a 160 x 160 position-velocity grid, random
offline rollouts are collected on the snapped dynamics, and an abstract
mountain-top goal is represented with an indicator dimension. The evaluation
uses graph shortest-step distance on the same discretized dynamics.

## maze2D

Run the full training + checkpoint evaluation:

```bash
experiments/run_maze2d_existing_qrl.sh
```

Main outputs:

- `results/experiments/maze2d_existing_qrl/train/checkpoint_final.pth`: trained existing-QRL checkpoint.
- `results/experiments/maze2d_existing_qrl/train/train_metrics.csv`: scalar losses exported from the existing QRL training loop.
- `results/experiments/maze2d_existing_qrl/eval/maze2d_heatmap_goal_*.png`: QRL, BFS shortest path, and Euclidean distance fields.
- `results/experiments/maze2d_existing_qrl/eval/distance_metrics.csv`: Euclidean, Manhattan, and QRL distance compared with BFS shortest-path distance using MSE, MAE, Pearson, and Spearman.
- `results/experiments/maze2d_existing_qrl/eval/navigation_metrics.csv`: Euclidean Greedy, QRL Greedy, and QRL Lookahead success rate, average successful steps, and average successful path length.

Both scripts use affine calibration before scale-dependent MSE/MAE are computed.
This keeps the comparison focused on distance structure rather than the
arbitrary raw scale of different distance estimators. Disable this behavior with
`--no-fit-affine`.

## Quick smoke tests

These commands only verify that the existing-QRL pipeline runs; they are not
thesis-quality training settings.

```bash
.venv/bin/python minimal_qrl/train.py \
  --env-type mountaincar --device cpu --output-dir /tmp/qrl_existing_mc_smoke \
  --num-episodes 4 --max-steps-per-episode 20 --batch-size 8 \
  --total-steps 2 --log-interval 1 --eval-interval 100 \
  --visualization-interval 0 --planning-eval-interval 0 \
  --mountaincar-gt-pos-bins 21 --mountaincar-gt-vel-bins 21

.venv/bin/python experiments/qrl_checkpoint_value_eval.py \
  --env-type mountaincar --checkpoint /tmp/qrl_existing_mc_smoke/checkpoint_final.pth \
  --output-dir /tmp/qrl_existing_mc_eval_smoke --device cpu \
  --gt-pos-bins 21 --gt-vel-bins 21 --goals 0.50,0.00

.venv/bin/python minimal_qrl/train.py \
  --env-type maze2d --device cpu --output-dir /tmp/qrl_existing_maze_smoke \
  --grid-size 9 9 --num-episodes 4 --max-steps-per-episode 20 \
  --batch-size 8 --total-steps 2 --log-interval 1 --eval-interval 100 \
  --visualization-interval 0 --planning-eval-interval 0

.venv/bin/python experiments/qrl_checkpoint_value_eval.py \
  --env-type maze2d --checkpoint /tmp/qrl_existing_maze_smoke/checkpoint_final.pth \
  --output-dir /tmp/qrl_existing_maze_eval_smoke --device cpu \
  --grid-size 9 9 --eval-pairs 5 --max-steps-per-episode 20
```
