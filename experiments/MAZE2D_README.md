# maze2D existing-QRL experiment

This directory now keeps only the maze2D experiment path used for the thesis:
training goes through the existing QRL implementation in `minimal_qrl/train.py`,
and checkpoint evaluation is handled by `experiments/qrl_checkpoint_value_eval.py`.

Run the full experiment with:

```bash
experiments/run_maze2d_existing_qrl.sh
```

Main files:

- `run_maze2d_existing_qrl.sh`: trains existing QRL on `Maze2DNavigation` and evaluates the saved checkpoint.
- `qrl_checkpoint_value_eval.py`: produces distance heatmaps, distance metrics, and simple navigation metrics.
- `minimal_qrl/envs/maze2d_navigation.py`: maze2D environment used by the training pipeline.

Main outputs:

- `results/experiments/maze2d_existing_qrl/train/checkpoint_final.pth`
- `results/experiments/maze2d_existing_qrl/train/train_metrics.csv`
- `results/experiments/maze2d_existing_qrl/eval/maze2d_heatmap_goal_*.png`
- `results/experiments/maze2d_existing_qrl/eval/distance_metrics.csv`
- `results/experiments/maze2d_existing_qrl/eval/navigation_metrics.csv`

The removed early scripts (`maze2d_qrl.py`, `maze2d_navigation_experiment.py`,
and `test_maze_env.py`) implemented standalone sanity checks or reimplemented
QRL logic locally. They are no longer part of the thesis experiment pipeline.
