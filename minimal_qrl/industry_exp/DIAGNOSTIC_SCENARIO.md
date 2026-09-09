# Long-horizon communication-inspection diagnostic

This diagnostic is intentionally separate from the original `medium` layout.
It is a controlled stress test, not a replacement for the main-distribution
result.

The single map contains three fixed strata:

- `u_trap`: starts are inside a west-opening U, face away from the device, and
  have a blocked direct path. A successful route must first increase Euclidean
  target distance and leave the U before going around its closed eastern wall.
- `comm_shadow_corridor`: starts are connected and face a central block. The
  upper and lower detours have similar early geometry; the upper route then
  enters a long radio shadow, while probes along the lower route remain
  communication-feasible.
- `easy_open`: direct, unobstructed tasks remain in both the north and south
  open regions. They prevent a method from looking better only because the
  benchmark consists entirely of hard cases.

The task bank is source-controlled through deterministic coordinates. It has
4 validation and 12 test tasks per stratum (12 validation and 36 test tasks in
total). Every record carries `task_id`, `stratum`, `difficulty`, `device_id`,
seed, and normalized start state. Both QRL execution evaluation and the unified
QRL/GCRL baseline evaluator report per-stratum metrics.

Prepare the canonical JSON files:

```bash
PHASE=prepare bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Train QRL on the diagnostic map:

```bash
PHASE=train_qrl DEVICE=mps bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Evaluate the fixed test split:

```bash
PHASE=eval_qrl QRL_CHECKPOINT=path/to/checkpoint_final.pth \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Run paired QRL/GCRL/MPPI evaluation on exactly the same starts:

```bash
PHASE=benchmark \
QRL_CHECKPOINTS="path/to/qrl_checkpoint.pth" \
CONTEXT_CHECKPOINTS="path/to/gcrl_checkpoint.pth" \
METHODS="mppi_no_terminal,qrl_mppi,context_her_ddpg_mppi" \
bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Run the highest-priority Oracle-MPPI falsification check on only the 12 fixed
test U-traps. This uses the same horizon 10, 128 samples, MPPI noise,
temperature, and running cost as the existing diagnostic benchmark; only the
terminal value changes to the exhaustive Hybrid A* lattice cost-to-go:

```bash
PHASE=oracle_mppi bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The first run constructs and caches the reverse Hybrid A* value table. Use
`RESUME=1` after interruption. Results are written to
`results/diagnostic_u_shadow_corridors/oracle_mppi_test_u_trap/` by default.

Run the label-free point-goal QRL control against one fixed physical terminal
lattice state:

```bash
PHASE=full_graph_point_goal_qrl DEVICE=mps \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The default fixes terminal candidate 0 (of the 24 candidates at the standard
0.25 x 24 lattice). Override it with `POINT_GOAL_CANDIDATE_INDEX`. This arm
keeps the IQE architecture, optimizer, 120,000 update budget, seed, primitive
set, and uniform edge minibatch sampling of the non-stratified full-graph
goal-set arm. It removes the synthetic `G` observation and all terminal-to-`G`
zero edges. Every task-goal pair is instead `d_theta(s, g*)`, and both reverse
reachability and post-training Dijkstra values are recomputed with `g*` as the
sole physical terminal node. Dijkstra values are never used by the trainer.

The run writes `full_graph_dataset_stats.json` with the selected physical state
and `point_goal_diagnostics/point_goal_qrl_metrics.json` with global, U-trap,
fixed-probe, and greedy-edge topology metrics. Do not reuse goal-set Oracle
records for this arm because they target a different terminal condition.

Run the supervised-IQE representability experiment and then evaluate its
terminal value with the same MPPI controller on the 12 test U-traps:

```bash
PHASE=supervised_iqe DEVICE=mps bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

This keeps the default QRL encoder and IQE quasimetric head unchanged and
optimizes only direct Huber/MSE regression to reverse-Dijkstra cost-to-go
labels. It does not instantiate or optimize the QRL constraint losses.

Run the optimizer-only 2x2 Tabular Potential-QRL diagnostic:

```bash
PHASE=tabular_potential_qrl DEVICE=mps \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The four cells cross `full` versus current-style `minibatch` constraint
sampling with `zero` versus practical `(0.25, 0.25, 0)` family epsilons. Each
goal-reachable state has one projected non-negative scalar parameter, terminal
states are fixed to zero, and the induced quasimetric is
`relu(u_s - u_s_prime)`. Oracle and Dijkstra values are unavailable to the
trainer and are created only for checkpoint evaluation. Defaults run five
seeds and write the aggregate JSON, step history CSV, and per-run value arrays
under `results/diagnostic_u_shadow_corridors/tabular_potential_qrl_2x2/`.

For a short smoke run, select one cell and seed explicitly:

```bash
PHASE=tabular_potential_qrl \
TABULAR_POTENTIAL_CELLS=full_zero TABULAR_POTENTIAL_SEEDS=0 \
TABULAR_FULL_STEPS=10 TABULAR_EVAL_INTERVAL=5 \
bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Run the Joint-feasible IQE constructive search:

```bash
PHASE=joint_feasible_iqe DEVICE=mps \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

This is a finite-graph capacity certificate rather than label-free QRL. It
keeps the default encoder, projector, and `iqe(dim=2048,components=64)` head,
while explicitly allowing all reverse-Dijkstra state labels and all 352,243
graph constraints. Ordinary edges are covered by shuffled sweeps plus a
periodically refreshed worst-edge active set; every direct-to-goal and
terminal-to-goal edge is included in every update. Global push, dual variables,
latent dynamics, bootstrapping, and trajectory losses are disabled.

When the targeted supervised checkpoint exists it is used automatically as a
constructive warm start. Set `JOINT_FEASIBLE_INIT_CHECKPOINT=none` to start from
random initialization. The phase writes both the final checkpoint and a best
checkpoint chosen by the pre-registered worst normalized certificate gap, then
runs the standard 16-probe U-trap diagnostic and 12 fixed U-trap MPPI tasks.

After the original run, use the pre-registered stronger optimizer as a separate
control (it writes to `joint_feasible_iqe_strong_optimizer`, so it does not
overwrite the original result):

```bash
PHASE=joint_feasible_iqe_strong DEVICE=mps \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

This variant keeps the union of every historical worst-4,096 edge set, samples
its cumulative replay on every update, and separately optimizes the current
worst 0.1% ordinary edges with both mean-square and p=8 max-like losses. All
edge terms use a quadratic 5,000-step ramp, while all 2,869 U-trap graph states
are a fixed additional supervised goal slice on every update. The graph,
warm-start checkpoint, architecture, learning rate, 30,000-step budget,
certificate thresholds, checkpoint rule, and downstream evaluations remain
unchanged. The exact graph determines the 2,869 anchor count; `-1` means all
states in the configured U-trap region rather than a hard-coded count.

The four interventions can be overridden with
`JOINT_FEASIBLE_STRONG_REPLAY_MODE`, `JOINT_FEASIBLE_STRONG_TAIL_FRACTION`,
`JOINT_FEASIBLE_STRONG_WARMUP_STEPS`/`JOINT_FEASIBLE_STRONG_WARMUP_POWER`, and
`JOINT_FEASIBLE_STRONG_U_GOAL_ANCHORS`. Do not tune them against the held-out
12-task MPPI result; inspect the full-graph certificate trajectory first.

Run the pre-registered 2x IQE capacity arm after the 1x strong result:

```bash
PHASE=joint_feasible_iqe_capacity_2x DEVICE=mps \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

The capacity arm changes `iqe(dim=2048,components=64)` to
`iqe(dim=4096,components=128)`, preserving 32 coordinates per component and
leaving the encoder, projector hidden layer, graph, strong losses, sampling,
30,000-step budget, seed, thresholds, and checkpoint rule unchanged. Because a
1x checkpoint cannot initialize the enlarged projector, the phase first trains
a capacity-matched supervised warm start for the same 10,000 updates. It reuses
the exact saved 1x targeted-supervised `.npz` dataset and records its SHA-256,
so no Oracle resampling is introduced.

The warm start and joint results are written separately to
`targeted_supervised_iqe_oracle_capacity_2x/` and
`joint_feasible_iqe_strong_optimizer_capacity_2x/`. If the warm-start
checkpoint already exists it is reused; set
`JOINT_FEASIBLE_CAPACITY_RETRAIN_WARM=1` to deliberately retrain it. Do not use
that flag after inspecting the joint result unless reporting the rerun as a
separate trial.

Interpret the arm with the existing certificate thresholds, without selecting
a new tolerance after training. A 2x run that preserves both topology checks
and reaches ordinary/Bellman max excess at most 0.25 is evidence for a finite
1x capacity bottleneck. Improvement without a pass is a capacity effect but not
a certificate. Failure to materially move the max excess leaves capacity and
sup-norm optimization unresolved; it is not proof of non-representability.

For a bounded pipeline smoke run without those downstream evaluations:

```bash
PHASE=joint_feasible_iqe DEVICE=cpu \
FULL_GRAPH_POSITION_RESOLUTION=0.5 FULL_GRAPH_HEADING_BINS=12 \
NUM_CRITICS=1 JOINT_FEASIBLE_INIT_CHECKPOINT=none \
JOINT_FEASIBLE_PRETRAIN_STEPS=1 JOINT_FEASIBLE_STEPS=1 \
JOINT_FEASIBLE_GOAL_BATCH_SIZE=32 JOINT_FEASIBLE_ORDINARY_BATCH_SIZE=32 \
JOINT_FEASIBLE_ACTIVE_SET_SIZE=16 JOINT_FEASIBLE_ACTIVE_BATCH_SIZE=16 \
JOINT_FEASIBLE_ACTIVE_REFRESH=1 JOINT_FEASIBLE_EVAL_BATCH_SIZE=512 \
JOINT_FEASIBLE_RUN_LOCAL_EVAL=0 JOINT_FEASIBLE_RUN_MPPI=0 \
bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Use `TASK_SPLIT=validation` while tuning. Reserve the default `test` split for
the final paired report.

Visualize all three strata (three individual PNG files plus one overview):

```bash
PHASE=visualize TASK_SPLIT=validation SAMPLE_INDEX=0 \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

Use `SAMPLE_INDEX=N` to highlight another fixed start in each stratum. The
underlying Python entry point also accepts `--scenario-config` and `--task-bank`
to render previously generated canonical JSON files directly.
