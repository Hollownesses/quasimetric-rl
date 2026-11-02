#!/usr/bin/env bash
# Usage: bash experiments/unified_mechanism/run.sh RAW_WORKTREE NEW_OUTPUT
set -euo pipefail
: "${1:?pass the historical worktree containing the raw checkpoints}"
: "${2:?pass a new output directory}"
raw=$(cd "$1" && pwd)
scripts=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$scripts/../.." && pwd)
upy=${U_PY:-"$repo/../quasimetric-rl/.venv/bin/python"}
opy=${OG_PY:-"$raw/experiments/ogbench_official_qrl/.venv/bin/python"}
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/qrl-unified-mpl"}
mkdir -p "$(dirname "$2")"
mkdir "$2"  # Refuse to overwrite a previous run.
"$upy" "$scripts/utrap.py" "$raw" "$2/u"
"$opy" "$scripts/pointmaze_analysis.py" "$raw" "$2/og"
"$opy" "$scripts/sensitivity.py" "$raw" "$2"
"$upy" "$scripts/summarize.py" "$raw" "$2"
"$upy" "$scripts/verify.py" "$raw" "$2"
