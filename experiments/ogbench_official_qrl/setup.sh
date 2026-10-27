#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${OGBENCH_PYTHON:-python3.10}"
COMMIT="$(cat "$ROOT/OFFICIAL_COMMIT")"
if [ ! -d "$ROOT/vendor/ogbench/.git" ]; then
  git clone https://github.com/seohongpark/ogbench.git "$ROOT/vendor/ogbench"
fi
git -C "$ROOT/vendor/ogbench" checkout --detach "$COMMIT"
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PYTHON" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
if [ -f "$ROOT/requirements-lock.txt" ]; then
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-lock.txt"
else
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements-runtime.txt"
fi
"$ROOT/.venv/bin/python" -m pip install --no-deps -e "$ROOT/vendor/ogbench"
"$ROOT/.venv/bin/python" -m pip check
"$ROOT/.venv/bin/python" -c 'import jax; print(jax.__version__, jax.devices())'
