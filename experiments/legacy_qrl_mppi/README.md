Historical GP300 checkpoint evaluation only. This adds one arm to the frozen
`results/r_series_mppi_20261026/main` protocol without changing that experiment.

Run from repository root, always with a new output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/qrl-r-mppi ../quasimetric-rl/.venv/bin/python \
  -m experiments.legacy_qrl_mppi.run \
  --output results/legacy_qrl_mppi_20261026/reproduction
```

The script strictly loads both critics from the old `aligned_v2` 100k checkpoint,
retains every saved tensor, uses critic zero as in the production value adapter,
sets distance_scale=1 for its negative-reward cost units, and leaves the later
terminal quotient disabled. The only newly inserted state buffers are the
backward-compatible False quotient flags. No training or output calibration.

Before the 64 formal episodes: verify the frozen baseline sources/configuration,
environment/task/cost signatures, direct and chunked value inference, original
weight equality, and two independent development episodes repeated exactly.
Freeze all inputs and the new source. Use the same 32 initial poses, two planner
seeds, H10/N128 controller and 300-step limit. Check first candidate banks against
R0, preserve every failure, replay all actions through real dynamics, and check
model/input/source hashes again after completion.

Existing none/R0/R1/oracle rows are reused only under their frozen protocol.
The report gives region/heading summaries and state/seed paired outcomes and
common-success cost differences. It is a historical system comparison, not an
algorithm-only ablation: training data, losses, task range, seed, and budget
differ. Oracle is a discretized environment reference, not continuous optimal
control. Two planner seeds share 32 initial poses.

Outputs include model validation, development smoke checks, manifest/FROZEN,
formal per-episode JSON/NPZ and JSONL, comparison/summary.json, Chinese report,
source snapshot, final_integrity.json, and COMPLETE. Original artifacts are read
only. The new code does not write into the sibling historical repository.
