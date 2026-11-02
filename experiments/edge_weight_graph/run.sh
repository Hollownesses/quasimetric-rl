#!/usr/bin/env bash
set -euo pipefail
: "${1:?pass a new output directory}"
here=$(cd "$(dirname "$0")" && pwd)
py=${PY:-"$here/../../../quasimetric-rl/.venv/bin/python"}
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=${MPLCONFIGDIR:-"${TMPDIR:-/tmp}/edge-weight-graph-mpl"}
PYTHONPATH="$here" "$py" -c 'from graph import freeze; from pathlib import Path; import sys; freeze(Path(sys.argv[1]))' "$1"
"$py" "$here/table.py" "$1"
"$py" "$here/iqe.py" "$1"
"$py" "$here/iqe.py" "$1" --witness
"$py" "$here/finish.py" "$1"
