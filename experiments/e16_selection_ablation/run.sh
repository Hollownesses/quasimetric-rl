#!/usr/bin/env bash
set -euo pipefail
: "${1:?pass a new output directory}"
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
raw=${ABLATION_RAW:-$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["raw_root"])' "$repo/results/unified_mechanism_20261102/audit.json")}
export ABLATION_RAW="$raw" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/e16-selection-mpl"}
py=${OG_PY:-"$raw/experiments/ogbench_official_qrl/.venv/bin/python"}
"$py" -u "$here/train.py" "$1"
cp "$here/EVALUATION_NOTES.md" "$1/"
"$py" -u "$here/official.py" "$1" >"$1/official.log" 2>&1
"$py" -u "$here/diagnose.py" "$1" >"$1/diagnose.log" 2>&1
"$py" -u "$here/finish.py" "$1"
