# Source-balance constraint exposure diagnostic

Read-only analysis of existing models. No gradient updates or new rollouts.
Definitions and numerical limits are frozen in `PROTOCOL.md`.

From the repository root (output directory must not exist):

```sh
export MPLCONFIGDIR=/tmp/qrl-constraint-exposure OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest -q experiments/constraint_exposure_diagnostic/test_diagnostic.py
$PY -m experiments.constraint_exposure_diagnostic.analyze \
  --output results/NEW_CONSTRAINT_EXPOSURE
$PY -m experiments.constraint_exposure_diagnostic.report \
  --root results/NEW_CONSTRAINT_EXPOSURE
```

Defaults read `task_source_balance_20261025`, `task_source_balance_multiseed_20261025`
and `task_partial_balance_20261025`, the connected expert replay, and its corrected
reference. Run summaries include input/source hashes, group definitions and
verification checks. Detailed edge/branch tables are local JSON/NPZ output.
The committed report and figures live in
`results/constraint_exposure_diagnostic_20261026/verified/`; reruns
should use a fresh directory and should reproduce numeric findings within the
documented floating-point tolerance.
