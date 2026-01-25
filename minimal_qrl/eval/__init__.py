"""
minimal_qrl.eval

将所有评估相关脚本/模块集中在此目录下，避免污染 minimal_qrl 顶层结构。

对外提供训练脚本常用的评估 API（见 minimal_qrl/train.py）。
"""

from .evaluation import evaluate_planning, evaluate_quasimetric, visualize_distance_field_heatmap
from .planning_evaluation import (
    evaluate_planning_reachability,
    greedy_navigation_rollout,
    navigation_rollout,
    LookaheadConfig,
)

__all__ = [
    "evaluate_quasimetric",
    "visualize_distance_field_heatmap",
    "evaluate_planning",
    "evaluate_planning_reachability",
    "greedy_navigation_rollout",
    "navigation_rollout",
    "LookaheadConfig",
]

