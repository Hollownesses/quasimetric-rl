#!/usr/bin/env bash
# 运行通信感知巡检 Dubins UAV 环境的轻量测试与可视化 smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[1/3] 运行 task-aware QRL loss 测试..."
"$PYTHON_BIN" minimal_qrl/test_task_aware_qrl_loss.py

echo "[2/3] 运行环境测试..."
"$PYTHON_BIN" minimal_qrl/test_comm_inspection_dubins_uav.py

echo "[3/3] 运行可视化 smoke test..."
"$PYTHON_BIN" -m minimal_qrl.visualize_comm_inspection_dubins_uav \
  --out results/minimal_qrl_inspection_dubins/comm_inspection_dubins_uav_vis/smoke_test.png

echo "全部完成。"
echo "Loss 测试文件: minimal_qrl/test_task_aware_qrl_loss.py"
echo "测试文件: minimal_qrl/test_comm_inspection_dubins_uav.py"
echo "可视化输出: results/minimal_qrl_inspection_dubins/comm_inspection_dubins_uav_vis/smoke_test.png"
