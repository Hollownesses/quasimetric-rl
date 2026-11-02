#!/usr/bin/env bash
set -euo pipefail
: "${1:?historical raw worktree}" "${2:?new output directory}"
scripts=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$scripts/../.." && pwd)
raw=$(cd "$1" && pwd)
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qrl-feasible-mpl"
mkdir -p "$(dirname "$2")"
mkdir "$2"
cp "$scripts/PROTOCOL.md" "$2/PROTOCOL.md"
upy=${U_PY:-$repo/../quasimetric-rl/.venv/bin/python}
opy=${OG_PY:-$raw/experiments/ogbench_official_qrl/.venv/bin/python}
"$upy" "$scripts/utrap.py" "$raw" "$2/u"
"$opy" "$scripts/ogbench.py" "$raw" "$2/og"
"$upy" "$scripts/summarize.py" "$2"
"$upy" "$scripts/verify.py" "$raw" "$2"
