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
