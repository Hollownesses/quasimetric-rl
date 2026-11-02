#!/usr/bin/env bash
set -euo pipefail
: "${1:?historical raw worktree}" "${2:?new output directory}"
scripts=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$scripts/../.." && pwd)
raw=$(cd "$1" && pwd)
upy=${U_PY:-$repo/../quasimetric-rl/.venv/bin/python}
opy=${OG_PY:-$raw/experiments/ogbench_official_qrl/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qrl-native-ablation-mpl"
mkdir -p "$(dirname "$2")"
mkdir "$2"
cp "$scripts/PROTOCOL.md" "$2/PROTOCOL.md"
"$upy" "$scripts/train_u.py" "$raw" "$2/u"
"$opy" "$scripts/train_og.py" "$raw" "$2/og"
"$upy" "$scripts/eval_u.py" "$raw" "$2/u"
"$opy" "$scripts/eval_og.py" "$raw" "$2/og"
"$upy" "$scripts/summarize.py" "$2"
"$upy" "$scripts/verify.py" "$raw" "$2"
