# Industrial inspection preliminary experiments

This directory contains four small experiments for the fixed-site, communication-aware
industrial UAV inspection setting. All shell entrypoints resolve the repository root
automatically and write to `results/experiments/` by default.

The scripts prefer `.venv/bin/python`, then the sibling checkout
`../quasimetric-rl/.venv/bin/python`, and finally `python3`. Override this with
`PYTHON_BIN=/path/to/python` when needed.

## 1. Task-feasible-set visualization and quantification

Run the default near/middle/far device analysis:

```bash
bash minimal_qrl/run_comm_inspection_feasible_set.sh
```

Main outputs:

- `feasible_set_results.json`: configuration and all quantitative records.
- `feasible_set_summary.csv`: Monte Carlo feasible fractions.
- `feasible_set_maps/*.png`: observation, communication, and joint feasible maps.
- `threshold_sensitivity.png`: joint feasible fraction against communication threshold.

Useful overrides:

```bash
DEVICE_IDS=reactor_r101,separator_v201,emergency_vent_ev01 \
NUM_SAMPLES=100000 \
GRID_RESOLUTION=180 \
COMM_THRESHOLDS=-0.5,0.0,0.5,1.0,1.5 \
COMM_THRESHOLD=0.5 \
OBSTACLE_CONFIG=medium \
OUTPUT_DIR=results/experiments/my_feasible_set \
bash minimal_qrl/run_comm_inspection_feasible_set.sh
```

`NUM_SAMPLES` controls the three-dimensional Monte Carlo estimate over `(x, y, heading)`.
`GRID_RESOLUTION` affects only the two-dimensional figures. When `DEVICE_IDS` is omitted,
the script selects devices near, midway from, and far from the ground station.

## 2. Fixed-site multi-start/multi-target reuse

Run the paired benchmark with one QRL checkpoint:

```bash
bash minimal_qrl/run_comm_inspection_multitask_reuse.sh
```

The default checkpoint is
`results/industrial_inspection_env_che/checkpoint_final.pth`. Replace it explicitly when
evaluating another model:

```bash
QRL_CHECKPOINTS=/absolute/path/to/checkpoint_final.pth \
STARTS_PER_DEVICE=25 \
METHODS=qrl_mppi,mppi_no_terminal,hybrid_astar \
DEVICE=cpu \
OUTPUT_DIR=results/experiments/my_multitask_reuse \
bash minimal_qrl/run_comm_inspection_multitask_reuse.sh
```

Multiple independently trained QRL checkpoints can be supplied as a space-separated list:

```bash
QRL_CHECKPOINTS="/path/seed0.pth /path/seed1.pth /path/seed2.pth" \
bash minimal_qrl/run_comm_inspection_multitask_reuse.sh
```

Main outputs:

- `baseline_results.json` and `baseline_results.csv`: episode-level paired results.
- `baseline_results.partial.*`: interruption-safe incremental records.
- `report/multitask_summary.csv`: compact method table with bootstrap intervals.
- `report/overall_performance_and_latency.png`: success, latency, and collision overview.
- `report/per_device_success_heatmap.png`: reuse performance for every device target.

`STARTS_PER_DEVICE=25` gives 600 tasks for a 24-device catalog. The same episode seeds are
used for every method. Hybrid A* may be slow; use `ASTAR_TIMEOUT_SEC` to bound each task.

## 3. Online latency and repeated QRL queries

```bash
bash minimal_qrl/run_comm_inspection_latency.sh
```

Example parameter replacement:

```bash
CHECKPOINT=/absolute/path/to/checkpoint_final.pth \
BATCH_SIZES=1,8,24,128,600 \
QUERY_REPEATS=1000 \
WARMUP=20 \
CONTROLLER_METHODS=qrl_mppi,mppi_no_terminal,hybrid_astar \
CONTROLLER_TRIALS=25 \
ASTAR_TIMEOUT_SEC=30 \
DEVICE=cpu \
OUTPUT_DIR=results/experiments/my_latency \
bash minimal_qrl/run_comm_inspection_latency.sh
```

Main outputs:

- `qrl_query_latency.csv`: every synchronized QRL batch timing measurement.
- `first_decision_latency.csv`: end-to-end `begin_episode + first act` timing.
- `latency_results.json`: median, P95, P99, mean, and bootstrap interval summaries.
- `latency_summary.png`: batch scaling and first-decision comparison.

Use the same `DEVICE`, hardware, MPPI parameters, and thread environment in every reported
run. Warm-up measurements are excluded from QRL query statistics. Hybrid A* defaults to a
five-second timeout in this script so that a preliminary run finishes in bounded time;
increase it for a final comparison.

## 4. Goal representation and communication-planning ablation

```bash
bash minimal_qrl/run_comm_inspection_modeling_ablation.sh
```

This is a paired 2x2 factorial evaluation:

| Goal guidance | Planning model |
| --- | --- |
| terminal goal set | communication-aware |
| one fixed terminal point | communication-aware |
| terminal goal set | communication-unaware |
| one fixed terminal point | communication-unaware |

All four conditions execute in the original complete environment. Communication is removed
only inside the unaware planner; final success is still judged by the original joint
observation, communication, and safety constraints. The goal set is approximated with
`MPPI_TERMINAL_SAMPLES` feasible states, while the point condition uses exactly one sampled
feasible terminal state.

Example parameter replacement:

```bash
STARTS_PER_DEVICE=10 \
MPPI_HORIZON=10 \
MPPI_NUM_SAMPLES=128 \
MPPI_TERMINAL_SAMPLES=128 \
COMM_THRESHOLD=0.5 \
OUTPUT_DIR=results/experiments/my_modeling_ablation \
bash minimal_qrl/run_comm_inspection_modeling_ablation.sh
```

Main outputs:

- `ablation_results.json` and `ablation_results.csv`: full results and paired summaries.
- `ablation_results.partial.csv`: flushed after every completed episode.
- `ablation_summary.png`: success, communication-feasible time, and cost comparison.

The point/set comparison changes the number of terminal representatives but keeps the MPPI
dynamics, sampling budget, and cost model fixed. The communication comparison keeps the
controller fixed and changes only the planner's communication model.

## Fast smoke commands

These commands validate the pipeline; their results are not suitable for the thesis:

```bash
NUM_SAMPLES=200 GRID_RESOLUTION=15 DEVICE_IDS=reactor_r101 \
OUTPUT_DIR=/tmp/inspection_exp1_smoke \
bash minimal_qrl/run_comm_inspection_feasible_set.sh

STARTS_PER_DEVICE=1 METHODS=qrl_mppi,mppi_no_terminal \
MPPI_HORIZON=2 MPPI_NUM_SAMPLES=4 MAX_EPISODE_STEPS=3 \
OUTPUT_DIR=/tmp/inspection_exp2_smoke \
bash minimal_qrl/run_comm_inspection_multitask_reuse.sh

BATCH_SIZES=1,2 QUERY_REPEATS=2 WARMUP=1 \
CONTROLLER_METHODS=qrl_mppi,mppi_no_terminal CONTROLLER_TRIALS=1 \
MPPI_HORIZON=2 MPPI_NUM_SAMPLES=4 \
OUTPUT_DIR=/tmp/inspection_exp3_smoke \
bash minimal_qrl/run_comm_inspection_latency.sh

STARTS_PER_DEVICE=1 MAX_EPISODE_STEPS=2 MPPI_HORIZON=2 \
MPPI_NUM_SAMPLES=4 MPPI_TERMINAL_SAMPLES=2 \
OUTPUT_DIR=/tmp/inspection_exp4_smoke \
bash minimal_qrl/run_comm_inspection_modeling_ablation.sh
```

## Common environment replacements

Every shell script accepts these environment-variable overrides where relevant:

- `DEVICE_CATALOG`: device catalog JSON.
- `BOUNDS`: quoted space-separated bounds, for example `"0 0 10 10"`.
- `OBSTACLE_CONFIG`: `none`, `simple`, `medium`, or `hard`.
- `COMM_THRESHOLD`: communication-quality cutoff.
- `REQUIRE_GROUND_STATION_LOS=1`: require direct ground-station LOS.
- `SEED`: base seed used to construct paired tasks.
- `OUTPUT_DIR`: target result directory.
- `PYTHON_BIN`: Python interpreter containing the project dependencies.

Use a new `OUTPUT_DIR` for every formal run. The unified multi-task benchmark and modeling
ablation write incremental files, but changing parameters while reusing a result directory
can otherwise make provenance ambiguous.
