#!/usr/bin/env bash
set -euo pipefail
: "${1:?historical raw worktree}" "${2:?new output directory}"
scripts=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$scripts/../.." && pwd)
raw=$(cd "$1" && pwd)
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qrl-objective-witness-mpl"
mkdir -p "$(dirname "$2")"
mkdir "$2"
cp "$scripts/PROTOCOL.md" "$scripts/STATISTICAL_NOTES.md" "$2/"
upy=${U_PY:-$repo/../quasimetric-rl/.venv/bin/python}
opy=${OG_PY:-$raw/experiments/ogbench_official_qrl/.venv/bin/python}
"$upy" "$scripts/utrap.py" "$raw" "$2/u"
"$opy" "$scripts/ogbench.py" "$raw" "$2/og"
"$opy" "$scripts/query_history.py" "$raw" "$2/og"
"$upy" "$scripts/reaggregate.py" "$2/og"
"$upy" "$scripts/solver_merit.py" "$raw" "$2" --mode u
"$opy" "$scripts/solver_merit.py" "$raw" "$2" --mode og
"$upy" "$scripts/summarize.py" "$raw" "$2"
"$opy" "$scripts/verify_og.py" "$raw" "$2/og"
"$upy" "$scripts/verify.py" "$raw" "$2"
