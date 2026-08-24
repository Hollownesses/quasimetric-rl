#!/usr/bin/env python3
"""
最小可运行 QRL 核心训练脚本（环境无关版本）
使用核心 QRL 模块，不依赖 d4rl/mujoco/gym 等复杂环境
支持多种环境：SimpleGrid2D, ContinuousObstacle2D 等
"""
import os
import sys
import argparse
import csv
import hashlib
import json
import logging
import platform
import tempfile
from time import perf_counter
from pathlib import Path
from typing import *
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "quasimetric_rl_cache")
for _cache_dir in (
    _CACHE_ROOT,
    os.path.join(_CACHE_ROOT, "matplotlib"),
    os.path.join(_CACHE_ROOT, "xdg"),
    os.path.join(_CACHE_ROOT, "xdg", "fontconfig"),
):
    os.makedirs(_cache_dir, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CACHE_ROOT, "xdg"))

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.models import QuasimetricCritic
from quasimetric_rl.modules.quasimetric_critic.models.encoder import Encoder
from quasimetric_rl.modules.quasimetric_critic.models.latent_dynamics import LatentDynamics
from quasimetric_rl.modules.quasimetric_critic.models.quasimetric_model import QuasimetricModel
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from quasimetric_rl.modules.quasimetric_critic.losses.global_push import GlobalPushLoss
from quasimetric_rl.modules.quasimetric_critic.losses.latent_dynamics import LatentDynamicsLoss
from quasimetric_rl.modules.quasimetric_critic.losses.abstract_goal_edge import AbstractGoalEdgeLoss
from quasimetric_rl.modules.quasimetric_critic.losses.temporal_path import (
    GoalReturnConstraintLoss,
    NstepGoalConsistencyLoss,
    TemporalPathConstraintLoss,
)
from quasimetric_rl.data import BatchData, Dataset, EpisodeData, register_offline_env
from minimal_qrl.envs import (
    SimpleGrid2D,
    ContinuousObstacle2D,
    Maze2DNavigation,
    MountainCar2D,
    DubinsUAV2D,
    CircleObstacle,
    CommInspectionDubinsUAV2D,
)
from minimal_qrl.dataset import (
    DenseUTrapTransitionConfig,
    FullGraphGoalSetQRLConfig,
    QRLExploreConfig,
    build_qrl_exploration_start_bank,
    create_dataset,
    create_full_graph_goal_set_qrl_dataset,
)
from minimal_qrl.eval import evaluate_quasimetric, visualize_distance_field_heatmap, evaluate_planning, LookaheadConfig
from minimal_qrl.eval.comm_inspection_oracle_bank import (
    CommInspectionOracleBankConfig,
    ensure_comm_inspection_oracle_bank,
    evaluate_qrl_on_oracle_bank,
    visualize_qrl_oracle_bank_heatmap,
)
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.subgoal_actor import (
    SubgoalActor,
    SubgoalActorTrainConfig,
    save_subgoal_actor_checkpoint,
    train_subgoal_actor,
)
from minimal_qrl.industry_exp.scalability_scenarios import (
    load_scenario_config,
    scenario_to_env_kwargs,
)


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_env_factory(env_type: str, **env_kwargs):
    """
    创建环境工厂函数
    
    Args:
        env_type: 环境类型 ('simple_grid', 'obstacle', 'maze2d', 'mountaincar',
                  'dubins_uav', 或 'comm_inspection_dubins_uav')
        **env_kwargs: 环境参数
    
    Returns:
        环境工厂函数
    """
    if env_type == 'simple_grid':
        def factory():
            return SimpleGrid2D(**env_kwargs)
        return factory
    elif env_type == 'obstacle':
        def factory():
            return ContinuousObstacle2D(**env_kwargs)
        return factory
    elif env_type == 'maze2d':
        def factory():
            return Maze2DNavigation(**env_kwargs)
        return factory
    elif env_type == 'mountaincar':
        def factory():
            return MountainCar2D(**env_kwargs)
        return factory
    elif env_type == 'dubins_uav':
        def factory():
            return DubinsUAV2D(**env_kwargs)
        return factory
    elif env_type == 'comm_inspection_dubins_uav':
        def factory():
            return CommInspectionDubinsUAV2D(**env_kwargs)
        return factory
    else:
        raise ValueError(f"未知的环境类型: {env_type}")


def _dubins_obstacles_from_args(args) -> list:
    """根据 --obstacle-config 或 --obstacles 为 Dubins 环境生成圆形障碍列表。"""
    bounds = tuple(args.bounds) if (hasattr(args, 'bounds') and args.bounds) else (0.0, 0.0, 5.0, 5.0)
    x_min, y_min, x_max, y_max = bounds
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w, h = x_max - x_min, y_max - y_min
    if getattr(args, 'obstacles', None) and len(args.obstacles) > 0:
        vals = list(args.obstacles)
        if len(vals) % 3 != 0:
            raise ValueError("--obstacles 必须是 3 的倍数 (x y r x y r ...)")
        return [CircleObstacle(x=float(vals[i]), y=float(vals[i + 1]), radius=float(vals[i + 2])) for i in range(0, len(vals), 3)]
    config = getattr(args, 'obstacle_config', 'none') or 'none'
    if config == 'none':
        return []
    if config == 'simple':
        return [CircleObstacle(x=cx, y=cy, radius=0.12 * min(w, h))]
    if config == 'medium':
        r = 0.10 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.35 * w, y=cy, radius=r),
            CircleObstacle(x=x_min + 0.65 * w, y=cy, radius=r),
            CircleObstacle(x=cx, y=y_min + 0.3 * h, radius=r * 0.8),
        ]
    if config == 'hard':
        r = 0.08 * min(w, h)
        return [
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.25 * h, radius=r),
            CircleObstacle(x=x_min + 0.25 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=x_min + 0.75 * w, y=y_min + 0.75 * h, radius=r),
            CircleObstacle(x=cx, y=cy, radius=r * 1.2),
        ]
    return []


def get_env_kwargs(args) -> dict:
    """
    根据环境类型和参数获取环境参数字典
    
    Args:
        args: 命令行参数
    
    Returns:
        环境参数字典
    """
    if getattr(args, '_scenario_data', None) is not None:
        if args.env_type != 'comm_inspection_dubins_uav':
            raise ValueError("--scenario-config currently supports comm_inspection_dubins_uav only")
        return scenario_to_env_kwargs(args._scenario_data)
    if args.env_type == 'simple_grid':
        return {
            'grid_size': tuple(args.grid_size),
            'max_episode_steps': args.max_steps_per_episode,
        }
    elif args.env_type == 'obstacle':
        return {
            'max_episode_steps': args.max_steps_per_episode,
            'grid_resolution': args.grid_resolution,
        }
    elif args.env_type == 'maze2d':
        return {
            'grid_size': tuple(args.grid_size),
            'max_episode_steps': args.max_steps_per_episode,
        }
    elif args.env_type == 'mountaincar':
        return {
            'goal_position': args.mountaincar_goal_position,
            'goal_velocity': args.mountaincar_goal_velocity,
            'goal_tolerance_pos': args.mountaincar_goal_tolerance_pos,
            'goal_tolerance_vel': args.mountaincar_goal_tolerance_vel,
            'max_episode_steps': args.max_steps_per_episode,
            'gt_pos_bins': args.mountaincar_gt_pos_bins,
            'gt_vel_bins': args.mountaincar_gt_vel_bins,
            'gt_goal_mode': args.mountaincar_gt_goal_mode,
            'dataset_mode': args.mountaincar_dataset_mode,
            'abstract_goal_transition_repeats': args.mountaincar_abstract_goal_transition_repeats,
        }
    elif args.env_type == 'dubins_uav':
        obstacles = _dubins_obstacles_from_args(args)
        kwargs = {
            'max_episode_steps': args.max_steps_per_episode,
            'bounds': tuple(args.bounds) if (hasattr(args, 'bounds') and args.bounds) else (0.0, 0.0, 5.0, 5.0),
            'omega_max': args.omega_max if (hasattr(args, 'omega_max') and args.omega_max is not None) else 3.0,
            'v': args.v if (hasattr(args, 'v') and args.v is not None) else 1.0,
            'dt': args.dt if (hasattr(args, 'dt') and args.dt is not None) else 0.1,
            'epsilon_pos': args.epsilon_pos if (hasattr(args, 'epsilon_pos') and args.epsilon_pos is not None) else 0.15,
            'epsilon_theta': args.epsilon_theta if (hasattr(args, 'epsilon_theta') and args.epsilon_theta is not None) else 0.2,
            'obstacles': obstacles,
            'use_cos_sin_obs': getattr(args, 'use_cos_sin_obs', True),
        }
        if hasattr(args, 'collision_penalty') and args.collision_penalty is not None:
            kwargs['collision_penalty'] = args.collision_penalty
        else:
            kwargs['collision_penalty'] = getattr(args, 'collision_penalty', -10.0)
        return kwargs
    elif args.env_type == 'comm_inspection_dubins_uav':
        obstacles = _dubins_obstacles_from_args(args)
        if not getattr(args, 'device_catalog', None):
            raise ValueError("comm_inspection_dubins_uav requires --device-catalog")
        kwargs = {
            'device_catalog': args.device_catalog,
            'max_steps': args.max_steps_per_episode,
            'bounds': tuple(args.bounds) if (hasattr(args, 'bounds') and args.bounds) else (0.0, 0.0, 10.0, 10.0),
            'omega_max': args.omega_max if (hasattr(args, 'omega_max') and args.omega_max is not None) else 3.0,
            'v': args.v if (hasattr(args, 'v') and args.v is not None) else 1.0,
            'dt': args.dt if (hasattr(args, 'dt') and args.dt is not None) else 0.1,
            'obstacles': obstacles,
            'comm_alpha': getattr(args, 'comm_alpha', 2.0),
            'comm_bias': getattr(args, 'comm_bias', 5.0),
            'comm_occlusion_penalty': getattr(args, 'comm_occlusion_penalty', 6.0),
            'comm_threshold': getattr(args, 'comm_threshold', 0.0),
            'require_ground_station_los': getattr(args, 'require_ground_station_los', False),
            'collision_cost': abs(getattr(args, 'collision_cost', 10.0)),
            'out_of_bounds_cost': abs(getattr(args, 'out_of_bounds_cost', 10.0)),
            'communication_break_cost': abs(getattr(args, 'communication_break_cost', 1.0)),
            'observation_violation_cost_weight': getattr(args, 'observation_violation_cost_weight', 1.0),
            'communication_violation_cost_weight': getattr(args, 'communication_violation_cost_weight', 0.5),
            'observation_failure_cost': abs(getattr(args, 'observation_failure_cost', 0.25)),
            'taskscore_beta_obs': getattr(args, 'taskscore_beta_obs', 1.0),
            'taskscore_beta_comm': getattr(args, 'taskscore_beta_comm', 1.0),
            'taskscore_beta_feas': getattr(args, 'taskscore_beta_feas', 0.5),
            'taskscore_margin_clip': getattr(args, 'taskscore_margin_clip', 2.0),
        }
        return kwargs
    else:
        raise ValueError(f"未知的环境类型: {args.env_type}")


def _log_planning_results(planning_results: dict, execution_modes: list, writer, logger, optim_steps: int, prefix: str = 'planning'):
    """
    统一的 planning 评估结果记录和打印函数
    
    Args:
        planning_results: evaluate_planning 返回的结果字典
        execution_modes: 执行模式列表
        writer: TensorBoard SummaryWriter
        logger: 日志记录器
        optim_steps: 当前优化步数
        prefix: TensorBoard tag 前缀（用于区分不同的 distance_type）
    """
    # 记录到 TensorBoard
    for mode in execution_modes:
        key = 'greedy_navigation' if mode == 'greedy' else f'{mode}_navigation'
        if key in planning_results:
            mres = planning_results[key]
            writer.add_scalar(f'{prefix}/{mode}/success_rate', mres['success_rate'], optim_steps)
            writer.add_scalar(f'{prefix}/{mode}/avg_steps', mres['avg_steps'], optim_steps)
            writer.add_scalar(f'{prefix}/{mode}/avg_path_length', mres['avg_path_length'], optim_steps)
    
    # Path efficiency
    pe_by_mode = planning_results.get('path_efficiency_by_mode', {})
    if isinstance(pe_by_mode, dict):
        for mode, pe in pe_by_mode.items():
            if not isinstance(pe, dict):
                continue
            writer.add_scalar(f'{prefix}/{mode}/avg_efficiency_ratio', pe.get('avg_efficiency_ratio', 0.0), optim_steps)
            writer.add_scalar(f'{prefix}/{mode}/median_efficiency_ratio', pe.get('median_efficiency_ratio', 0.0), optim_steps)
    
    # 打印结果
    for mode in execution_modes:
        key = 'greedy_navigation' if mode == 'greedy' else f'{mode}_navigation'
        if key in planning_results:
            mres = planning_results[key]
            mode_prefix = f"[{prefix.split('/')[-1]}] " if '/' in prefix else ""
            logger.info(
                f"{mode_prefix}Planning评估({mode}): Success Rate={mres['success_rate']:.3f}, "
                f"Avg Steps={mres['avg_steps']:.1f}, "
                f"Avg Path Length={mres['avg_path_length']:.3f}"
            )
    
    if pe_by_mode:
        for mode, pe in pe_by_mode.items():
            if not isinstance(pe, dict):
                continue
            mode_prefix = f"[{prefix.split('/')[-1]}] " if '/' in prefix else ""
            logger.info(
                f"{mode_prefix}Path Efficiency({mode}): Avg Ratio={float(pe.get('avg_efficiency_ratio', 0.0)):.3f}, "
                f"Median={float(pe.get('median_efficiency_ratio', 0.0)):.3f}"
            )


def _scalarize_for_csv(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value.mean().item())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


def _flatten_scalar_info(info: dict, prefix: str = '') -> dict:
    row = {}
    for key, value in info.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            row.update(_flatten_scalar_info(value, prefix=name))
        else:
            scalar = _scalarize_for_csv(value)
            if scalar is not None:
                row[name] = scalar
    return row


def _write_metric_rows_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comm_inspection_global_push_conf(args) -> GlobalPushLoss.Conf:
    """Build the task-aware GlobalPush config from communication CLI args."""
    return GlobalPushLoss.Conf(
        softplus_offset=float(args.global_push_softplus_offset),
        softplus_beta=float(args.global_push_softplus_beta),
        abstract_goal_ratio=float(args.global_push_abstract_goal_ratio),
        state_goal_ratio=float(args.global_push_state_goal_ratio),
    )


def _load_excluded_exploration_starts(
    task_bank_path: Optional[str],
    bounds: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    if not task_bank_path:
        return ()
    with open(task_bank_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    x_min, y_min, x_max, y_max = (float(value) for value in bounds)
    width = x_max - x_min
    height = y_max - y_min
    starts: list[tuple[float, float, float]] = []
    for record in payload.get('records', []):
        if 'start' in record:
            raw = record['start']
            starts.append((float(raw[0]), float(raw[1]), float(raw[2])))
        elif 'start_normalized' in record:
            raw = record['start_normalized']
            starts.append(
                (
                    x_min + float(raw[0]) * width,
                    y_min + float(raw[1]) * height,
                    float(raw[2]),
                )
            )
    return tuple(starts)


def _load_failed_u_trap_starts(path: str) -> tuple[tuple[float, float, float], ...]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    records = payload.get('episode_results', payload) if isinstance(payload, dict) else payload
    failed = [
        tuple(float(value) for value in record['start'])
        for record in records
        if str(record.get('stratum', '')) == 'u_trap'
        and not bool(record.get('success', False))
        and record.get('start') is not None
    ]
    if not failed:
        raise ValueError(f'no failed U-trap starts found in {path}')
    return tuple(failed)


def _exploration_regions_from_scenario(
    scenario: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Sequence[float]]]:
    if not scenario:
        return None
    metadata = scenario.get('metadata', {})
    regions = metadata.get('exploration_diagnostic_regions')
    return regions if isinstance(regions, Mapping) else None


def _exploration_routes_from_scenario(
    scenario: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Sequence[str]]]:
    if not scenario:
        return None
    routes = scenario.get('metadata', {}).get('exploration_diagnostic_routes')
    return routes if isinstance(routes, Mapping) else None


def _exploration_start_strata_from_scenario(
    scenario: Optional[Mapping[str, Any]],
) -> tuple[tuple[str, float, tuple[float, float, float, float]], ...]:
    if not scenario:
        return ()
    raw_strata = scenario.get('metadata', {}).get('exploration_start_strata', ())
    result = []
    for record in raw_strata:
        if not isinstance(record, Mapping):
            raise ValueError('exploration_start_strata records must be mappings')
        bounds = tuple(float(value) for value in record['bounds'])
        if len(bounds) != 4:
            raise ValueError('exploration_start_strata bounds must contain four values')
        result.append((str(record['name']), float(record['weight']), bounds))
    return tuple(result)


def _write_exploration_start_bank(
    output_dir: str,
    *,
    states: np.ndarray,
    args,
    excluded_count: int,
) -> str:
    records = [
        {
            'start_index': int(index),
            'state': [float(value) for value in state],
        }
        for index, state in enumerate(np.asarray(states, dtype=np.float32))
    ]
    payload: Dict[str, Any] = {
        'schema_version': 1,
        'generation_mode': 'fixed_goal_blind_stratified_xy_heading',
        'collection_mode': 'qrl_explore',
        'seed': int(args.seed),
        'position_resolution': float(args.explore_start_position_resolution),
        'heading_bins': int(args.explore_start_heading_bins),
        'exclusion_task_bank': (
            os.path.abspath(args.explore_exclusion_task_bank)
            if args.explore_exclusion_task_bank
            else None
        ),
        'exclusion_radius': float(args.explore_exclusion_radius),
        'excluded_eval_start_count': int(excluded_count),
        'start_boundary_margin': float(args.explore_start_boundary_margin),
        'start_strata': [
            {
                'name': name,
                'weight': float(weight),
                'bounds': [float(value) for value in bounds],
            }
            for name, weight, bounds in _exploration_start_strata_from_scenario(
                getattr(args, '_scenario_data', None)
            )
        ],
        'records': records,
    }
    digest_source = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    payload['content_digest'] = hashlib.sha256(digest_source).hexdigest()
    path = os.path.join(output_dir, 'exploration_start_bank.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def train(args):
    """训练主函数"""
    end_to_end_start = perf_counter()
    checkpoint_io_time = 0.0
    evaluation_time = 0.0
    oracle_generation_time = 0.0
    collection_stats: Dict[str, Any] = {}
    args._scenario_data = None
    training_mode = str(getattr(args, 'comm_dataset_mode', 'standard'))
    full_graph_modes = {
        'full_graph_goal_set',
        'full_graph_goal_set_stratified_constraints',
    }
    stratified_full_graph_constraints = (
        training_mode == 'full_graph_goal_set_stratified_constraints'
    )
    if getattr(args, 'scenario_config', None):
        args._scenario_data = load_scenario_config(args.scenario_config)
        args.env_type = 'comm_inspection_dubins_uav'
        args.max_steps_per_episode = int(args._scenario_data['max_episode_steps'])
    if getattr(args, 'comm_dataset_mode', 'standard') == 'qrl_explore':
        args.task_aware_teacher_ratio = 0.0
        args.target_env_transitions = None
    if getattr(args, 'comm_dataset_mode', 'standard') == 'dense_transition_original':
        # This mode is intentionally a clean ablation of the original QRL core.
        args.qrl_temporal_constraint_weight = 0.0
        args.qrl_goal_return_constraint_weight = 0.0
        args.qrl_nstep_goal_constraint_weight = 0.0
        args.qrl_success_transition_weight = 1.0
    if training_mode in full_graph_modes:
        # Exact baseline-QRL diagnostic: one graph, one explicit G, no enhanced
        # temporal targets or replay-distribution effects.
        args.task_aware_teacher_ratio = 0.0
        args.target_env_transitions = None
        args.qrl_temporal_constraint_weight = 0.0
        args.qrl_goal_return_constraint_weight = 0.0
        args.qrl_nstep_goal_constraint_weight = 0.0
        args.qrl_success_transition_weight = 1.0
        args.global_push_abstract_goal_ratio = 1.0
        args.global_push_state_goal_ratio = 0.0
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timing_progress_path = os.path.join(output_dir, 'timing_progress.json')
    prior_timing: Dict[str, Any] = {}
    if getattr(args, 'init_checkpoint', None) and os.path.exists(timing_progress_path):
        with open(timing_progress_path, 'r', encoding='utf-8') as f:
            prior_timing = json.load(f)
        checkpoint_io_time = float(prior_timing.get('checkpoint_io_time_sec', 0.0))
        evaluation_time = float(prior_timing.get('evaluation_time_sec', 0.0))
        oracle_generation_time = float(
            prior_timing.get('oracle_generation_time_sec', 0.0)
        )
    
    # 设置日志
    logger = setup_logging(output_dir)
    logger.info(f"开始训练，输出目录: {output_dir}")
    logger.info(f"环境类型: {args.env_type}")
    logger.info(f"参数: {args}")
    
    # 设置设备（支持 Apple Silicon MPS 加速）
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    # MPS 在部分评估/热力图运算上会触发 buffer 错误，评估与可视化改用 CPU
    eval_device = torch.device('cpu') if device.type == 'mps' else device
    eval_device_str = str(eval_device)
    if eval_device_str == 'cpu' and device.type == 'mps':
        logger.info("评估与可视化将使用 CPU 以避免 MPS 兼容性问题")

    # 获取环境参数
    env_kwargs = get_env_kwargs(args)
    logger.info(f"环境参数: {env_kwargs}")

    if args.env_type == 'comm_inspection_dubins_uav':
        precheck_env = CommInspectionDubinsUAV2D(**env_kwargs)
        try:
            precheck_env.reset(seed=args.seed)
            probe_goal = precheck_env.sample_task_terminal_state(seed=args.seed)
            logger.info(f"通信巡检 goal-set 环境预检查通过，示例 G_task(xi) 终态: {probe_goal}")
        except RuntimeError as e:
            raise ValueError(
                "当前通信巡检环境配置下不存在可采样的任务可行目标。"
                "请检查 device_catalog / obstacle_config / comm_threshold 等参数。"
            ) from e
    
    # 创建环境工厂函数
    create_env_fn = create_env_factory(args.env_type, **env_kwargs)
    qrl_explore_config: Optional[QRLExploreConfig] = None
    dense_u_trap_config: Optional[DenseUTrapTransitionConfig] = None
    full_graph_config: Optional[FullGraphGoalSetQRLConfig] = None
    exploration_start_bank_path: Optional[str] = None
    if getattr(args, 'comm_dataset_mode', 'standard') == 'qrl_explore':
        if args.env_type != 'comm_inspection_dubins_uav':
            raise ValueError('QRL-explore 数据模式仅支持 comm_inspection_dubins_uav')
        explore_env = create_env_fn()
        excluded_starts = _load_excluded_exploration_starts(
            getattr(args, 'explore_exclusion_task_bank', None),
            (explore_env.x_min, explore_env.y_min, explore_env.x_max, explore_env.y_max),
        )
        base_explore_config = QRLExploreConfig(
            attempted_env_steps=int(args.explore_attempted_env_steps),
            start_position_resolution=float(args.explore_start_position_resolution),
            start_heading_bins=int(args.explore_start_heading_bins),
            action_hold_min_steps=int(args.explore_action_hold_min_steps),
            action_hold_max_steps=int(args.explore_action_hold_max_steps),
            straight_action_probability=float(args.explore_straight_action_probability),
            exclusion_radius=float(args.explore_exclusion_radius),
            excluded_start_states=excluded_starts,
            diagnostic_regions=_exploration_regions_from_scenario(args._scenario_data),
            diagnostic_routes=_exploration_routes_from_scenario(args._scenario_data),
            start_strata=_exploration_start_strata_from_scenario(args._scenario_data),
            start_boundary_margin=float(args.explore_start_boundary_margin),
            local_safety_lookahead_steps=int(args.explore_local_safety_lookahead_steps),
        )
        if not base_explore_config.start_strata:
            raise ValueError(
                'qrl_explore requires metadata.exploration_start_strata in the scenario'
            )
        start_bank = build_qrl_exploration_start_bank(
            explore_env,
            base_explore_config,
            seed=int(args.seed),
        )
        qrl_explore_config = QRLExploreConfig(
            attempted_env_steps=base_explore_config.attempted_env_steps,
            start_position_resolution=base_explore_config.start_position_resolution,
            start_heading_bins=base_explore_config.start_heading_bins,
            action_hold_min_steps=base_explore_config.action_hold_min_steps,
            action_hold_max_steps=base_explore_config.action_hold_max_steps,
            straight_action_probability=base_explore_config.straight_action_probability,
            exclusion_radius=base_explore_config.exclusion_radius,
            excluded_start_states=base_explore_config.excluded_start_states,
            start_states=tuple(tuple(float(value) for value in state) for state in start_bank),
            diagnostic_regions=base_explore_config.diagnostic_regions,
            diagnostic_routes=base_explore_config.diagnostic_routes,
            start_strata=base_explore_config.start_strata,
            start_boundary_margin=base_explore_config.start_boundary_margin,
            local_safety_lookahead_steps=base_explore_config.local_safety_lookahead_steps,
        )
        exploration_start_bank_path = _write_exploration_start_bank(
            output_dir,
            states=start_bank,
            args=args,
            excluded_count=len(excluded_starts),
        )
        logger.info(
            'QRL-explore 已启用: attempted_env_steps=%d, teacher_ratio=0, '
            'start_bank=%d, persistent_action_hold=[%d,%d], bank_path=%s',
            int(qrl_explore_config.attempted_env_steps),
            int(len(start_bank)),
            int(qrl_explore_config.action_hold_min_steps),
            int(qrl_explore_config.action_hold_max_steps),
            exploration_start_bank_path,
        )
    if getattr(args, 'comm_dataset_mode', 'standard') == 'dense_transition_original':
        if args.env_type != 'comm_inspection_dubins_uav':
            raise ValueError('dense-transition QRL only supports comm_inspection_dubins_uav')
        if not args.dense_transition_failure_results:
            raise ValueError(
                'dense-transition QRL requires --dense-transition-failure-results'
            )
        dense_u_trap_config = DenseUTrapTransitionConfig(
            device_id=str(args.dense_transition_device_id),
            position_resolution=float(args.dense_transition_position_resolution),
            heading_bins=int(args.dense_transition_heading_bins),
            primitive_steps=int(args.dense_transition_primitive_steps),
            primitive_scales=tuple(float(value) for value in args.dense_transition_primitive_scales),
            local_fraction=float(args.dense_transition_local_fraction),
            diagnostic_regions=_exploration_regions_from_scenario(args._scenario_data),
            failed_start_states=_load_failed_u_trap_starts(
                args.dense_transition_failure_results
            ),
            failure_position_radius=float(args.dense_transition_failure_position_radius),
            failure_heading_radius=float(args.dense_transition_failure_heading_radius),
        )
        logger.info(
            'Dense-transition Original QRL enabled: global/local batch=%.3f/%.3f, '
            'lattice=%.3f x %d headings, primitives=%s x %d steps, Oracle training labels=none',
            1.0 - dense_u_trap_config.local_fraction,
            dense_u_trap_config.local_fraction,
            dense_u_trap_config.position_resolution,
            dense_u_trap_config.heading_bins,
            dense_u_trap_config.primitive_scales,
            dense_u_trap_config.primitive_steps,
        )
    if training_mode in full_graph_modes:
        if args.env_type != 'comm_inspection_dubins_uav':
            raise ValueError('full-graph goal-set QRL only supports comm_inspection_dubins_uav')
        full_graph_config = FullGraphGoalSetQRLConfig(
            device_id=str(args.full_graph_device_id),
            position_resolution=float(args.full_graph_position_resolution),
            heading_bins=int(args.full_graph_heading_bins),
            primitive_steps=int(args.full_graph_primitive_steps),
            primitive_scales=tuple(float(value) for value in args.full_graph_primitive_scales),
            uniform_push_seed=int(args.full_graph_uniform_push_seed),
            stratified_constraints=stratified_full_graph_constraints,
        )
        logger.info(
            'Full-Graph Baseline Goal-set QRL enabled: lattice=%.3f x %d headings, '
            'macro primitives=%s x %d, explicit unified G, uniform state->G push, '
            'continuous env.step=none, Oracle value labels=none',
            full_graph_config.position_resolution,
            full_graph_config.heading_bins,
            full_graph_config.primitive_scales,
            full_graph_config.primitive_steps,
        )
        if stratified_full_graph_constraints:
            logger.info(
                'Stratified constraint enforcement: ordinary=%d sampled/batch, '
                'direct-to-G=all, terminal-to-G=all, independent dual variables',
                int(args.batch_size),
            )
    
    # 注册环境（如果还没注册）
    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
    env_key = (args.env_type, args.env_type)  # 使用 env_type 作为 name
    if full_graph_config is None and env_key not in CREATE_ENV_REGISTRY:
        def load_episodes():
            env = create_env_fn()
            return create_dataset(
                env=env,
                num_episodes=args.num_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
                sample_valid_states=True,
                seed=args.seed,
                task_aware_teacher_ratio=(
                    float(args.task_aware_teacher_ratio)
                    if args.env_type == 'comm_inspection_dubins_uav'
                    else 0.0
                ),
                target_env_transitions=getattr(args, 'target_env_transitions', None),
                collection_stats=collection_stats,
                qrl_explore_config=qrl_explore_config,
                dense_u_trap_config=dense_u_trap_config,
            )
        
        register_offline_env(
            args.env_type, args.env_type,  # 使用 env_type 作为 name
            create_env_fn=create_env_fn,
            load_episodes_fn=load_episodes,
        )
        logger.info(f"已注册环境: {env_key}")
    
    # 创建数据集
    logger.info("创建数据集...")
    data_start = perf_counter()
    if full_graph_config is not None:
        dataset = create_full_graph_goal_set_qrl_dataset(
            create_env_fn(),
            full_graph_config,
            collection_stats=collection_stats,
        )
    else:
        dataset_conf = Dataset.Conf(
            kind=args.env_type,
            name=args.env_type,  # 使用 env_type 作为 name
            future_observation_discount=0.99,
        )
        dataset = dataset_conf.make(dummy=False)
    data_time_sec = float(prior_timing.get('data_time_sec', 0.0)) + (perf_counter() - data_start)
    logger.info(f"数据集大小: {len(dataset)} 个转移")
    success_transition_mask = dataset.raw_data.transition_infos.get(
        'task_success_episode'
    )
    if success_transition_mask is not None:
        success_transition_count = int(
            success_transition_mask.to(dtype=torch.bool).sum().item()
        )
        success_weight = float(args.qrl_success_transition_weight)
        weighted_success = success_weight * success_transition_count
        weighted_total = weighted_success + len(dataset) - success_transition_count
        expected_success_sample_ratio = float(
            weighted_success / max(weighted_total, 1.0)
        )
        collection_stats.update({
            'natural_success_transition_count': success_transition_count,
            'success_transition_sample_weight': success_weight,
            'expected_success_transition_sample_ratio': expected_success_sample_ratio,
        })
        logger.info(
            '自然成功轨迹转移: %d/%d; 采样权重=%.3f; 预期 batch 占比=%.4f',
            success_transition_count,
            len(dataset),
            success_weight,
            expected_success_sample_ratio,
        )
    if collection_stats:
        logger.info(f"数据收集统计: {collection_stats}")
    if dense_u_trap_config is not None:
        dense_stats_path = os.path.join(
            output_dir, 'dense_transition_collection_stats.json'
        )
        with open(dense_stats_path, 'w', encoding='utf-8') as handle:
            json.dump(collection_stats, handle, ensure_ascii=False, indent=2)
        logger.info('Saved dense-transition provenance: %s', dense_stats_path)
    if full_graph_config is not None:
        full_graph_stats_path = os.path.join(
            output_dir, 'full_graph_dataset_stats.json'
        )
        with open(full_graph_stats_path, 'w', encoding='utf-8') as handle:
            json.dump(collection_stats, handle, ensure_ascii=False, indent=2)
        logger.info('Saved full-graph provenance: %s', full_graph_stats_path)
    if qrl_explore_config is not None:
        exploration_stats = dict(collection_stats)
        exploration_stats.update(
            {
                'start_bank_path': os.path.abspath(exploration_start_bank_path),
                'exclusion_task_bank': (
                    os.path.abspath(args.explore_exclusion_task_bank)
                    if args.explore_exclusion_task_bank
                    else None
                ),
                'seed': int(args.seed),
            }
        )
        exploration_stats_path = os.path.join(
            output_dir, 'exploration_collection_stats.json'
        )
        with open(exploration_stats_path, 'w', encoding='utf-8') as handle:
            json.dump(exploration_stats, handle, ensure_ascii=False, indent=2)
        logger.info('保存 QRL-explore 收集统计: %s', exploration_stats_path)
    
    # 创建 QRL Agent 和 Losses
    logger.info("创建 QRL Agent 和 Losses...")
    step_cost = 1.0
    if args.env_type == 'mountaincar':
        agent_conf = QRLConf(
            actor=None,
            num_critics=args.num_critics,
            quasimetric_critic=QuasimetricCriticConf(
                model=QuasimetricCritic.Conf(
                    encoder=Encoder.Conf(
                        arch=tuple(args.mountaincar_encoder_arch),
                        latent_size=args.mountaincar_latent_size,
                    ),
                    quasimetric_model=QuasimetricModel.Conf(
                        projector_arch=tuple(args.mountaincar_projector_arch),
                        quasimetric_head_spec=(
                            f"iqe(dim={args.mountaincar_iqe_dim},"
                            f"components={args.mountaincar_iqe_components})"
                        ),
                    ),
                    latent_dynamics=LatentDynamics.Conf(
                        arch=tuple(args.mountaincar_transition_arch),
                        residual=True,
                    ),
                ),
                losses=QuasimetricCriticLosses.Conf(
                    global_push=GlobalPushLoss.Conf(
                        softplus_beta=args.mountaincar_global_beta,
                        softplus_offset=args.mountaincar_global_offset,
                    ),
                    local_constraint=LocalConstraintLoss.Conf(
                        epsilon=args.mountaincar_epsilon,
                        step_cost=step_cost,
                        init_lagrange_multiplier=args.mountaincar_lambda_init,
                    ),
                    latent_dynamics=LatentDynamicsLoss.Conf(
                        weight=args.mountaincar_transition_loss_weight,
                    ),
                    critic_optim=AdamWSpec.Conf(lr=args.mountaincar_model_lr),
                    lagrange_mult_optim=AdamWSpec.Conf(lr=args.mountaincar_lambda_lr),
                )
            ),
        )
    elif args.env_type == 'dubins_uav':
        # 约束为 d(s,s') 不超过 step_cost；网络输出多为 O(1)，故用 1.0 易满足，评估时 pred*dt 得时间
        step_cost = 1.0
        agent_conf = QRLConf(
            actor=None,
            num_critics=args.num_critics,
            quasimetric_critic=QuasimetricCriticConf(
                losses=QuasimetricCriticLosses.Conf(
                    local_constraint=LocalConstraintLoss.Conf(step_cost=step_cost),
                    critic_optim=AdamWSpec.Conf(lr=5e-5),
                    lagrange_mult_optim=AdamWSpec.Conf(lr=5e-3),
                )
            ),
        )
    elif args.env_type == 'comm_inspection_dubins_uav':
        # negative_reward: reward=-cost_total，用逐 transition 非负任务 cost 锚定 QRL local constraint。
        step_cost = 1.0
        qrl_cost_source = str(args.qrl_cost_source)
        if qrl_cost_source != "negative_reward":
            raise ValueError("goal-set 通信巡检 QRL 必须使用 --qrl-cost-source negative_reward")
        logger.info(f"通信巡检 QRL local cost source: {qrl_cost_source}")
        agent_conf = QRLConf(
            actor=None,
            num_critics=args.num_critics,
            quasimetric_critic=QuasimetricCriticConf(
                losses=QuasimetricCriticLosses.Conf(
                    global_push=_comm_inspection_global_push_conf(args),
                    local_constraint=LocalConstraintLoss.Conf(
                        step_cost=step_cost,
                        cost_source=qrl_cost_source,
                        constraint_mode=(
                            'full_graph_stratified'
                            if stratified_full_graph_constraints
                            else 'unified'
                        ),
                        direct_goal_epsilon=float(
                            args.full_graph_direct_goal_epsilon
                        ),
                        terminal_goal_epsilon=float(
                            args.full_graph_terminal_goal_epsilon
                        ),
                    ),
                    abstract_goal_edge=AbstractGoalEdgeLoss.Conf(weight=float(args.abstract_goal_edge_loss_weight)),
                    temporal_path=TemporalPathConstraintLoss.Conf(
                        weight=float(args.qrl_temporal_constraint_weight),
                        min_future_steps=int(args.qrl_temporal_min_future_steps),
                    ),
                    goal_return=GoalReturnConstraintLoss.Conf(
                        weight=float(args.qrl_goal_return_constraint_weight),
                    ),
                    nstep_goal=NstepGoalConsistencyLoss.Conf(
                        weight=float(args.qrl_nstep_goal_constraint_weight),
                        min_future_steps=int(args.qrl_temporal_min_future_steps),
                        target_tau=float(args.qrl_nstep_target_tau),
                    ),
                    critic_optim=AdamWSpec.Conf(lr=5e-5),
                    lagrange_mult_optim=AdamWSpec.Conf(lr=5e-3),
                )
            ),
        )
    else:
        agent_conf = QRLConf(
            actor=None,
            num_critics=args.num_critics,
        )
    agent, losses = agent_conf.make(
        env_spec=dataset.env_spec,
        total_optim_steps=max(1, int(args.total_steps)),
    )
    agent.to(device)
    losses.to(device)
    logger.info(f"Agent: {agent}")
    logger.info(f"Losses: {losses}")

    loaded_optim_steps = 0
    if getattr(args, 'init_checkpoint', None):
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state_dict = ckpt
        losses_state = None
        if isinstance(ckpt, dict) and 'agent' in ckpt:
            state_dict = ckpt['agent']
            losses_state = ckpt.get('losses')
            loaded_optim_steps = int(ckpt.get('optim_steps', 0))
        agent.load_state_dict(state_dict)
        if losses_state is not None:
            try:
                losses.load_state_dict(losses_state)
            except Exception as exc:
                logger.warning(f"加载 losses 状态失败，将仅加载 critic 参数: {exc}")
        logger.info(f"已加载初始 checkpoint: {args.init_checkpoint} (optim_steps={loaded_optim_steps})")
    
    # 创建数据加载器
    # MPS 不支持 pin_memory，只在 CUDA 时启用
    use_pin_memory = (device.type == 'cuda')
    dataloader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # 简化版本，不使用多进程
        pin_memory=use_pin_memory,
        successful_transition_weight=(
            float(args.qrl_success_transition_weight)
            if args.env_type == 'comm_inspection_dubins_uav'
            else 1.0
        ),
        dense_transition_fraction=(
            float(dense_u_trap_config.local_fraction)
            if dense_u_trap_config is not None
            else None
        ),
        full_graph_stratified_constraints=stratified_full_graph_constraints,
    )
    
    # 创建 TensorBoard writer（使用带时间戳的子目录，便于区分不同训练）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard', timestamp)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard 日志目录: {tensorboard_dir}")
    
    # 创建环境实例用于评估
    eval_env = create_env_fn()
    # 通信巡检使用 negative_reward 学习逐 transition 的真实环境代价，
    # critic 输出已与 cost_total 同单位，不能沿用固定 step-cost Dubins 的 dt 缩放。
    evaluation_distance_scale = (
        1.0 if args.env_type == 'comm_inspection_dubins_uav' else None
    )
    constraint_checkpoint_rows: List[Dict[str, Any]] = []
    constraint_checkpoint_path = os.path.join(
        output_dir,
        'constraint_checkpoint_metrics.csv',
    )
    constraint_checkpoint_data = None
    if stratified_full_graph_constraints:
        from minimal_qrl.baselines import HybridAStarConfig, HybridAStarValueOracle
        from minimal_qrl.industry_exp.supervised_iqe_oracle import (
            _ranking_accuracy,
            _regression_metrics,
            _successor_ranking_dataset,
            _u_trap_local_dataset,
        )

        checkpoint_oracle_started = perf_counter()
        checkpoint_oracle_config = HybridAStarConfig(
            position_resolution=float(full_graph_config.position_resolution),
            heading_bins=int(full_graph_config.heading_bins),
            primitive_steps=int(full_graph_config.primitive_steps),
            primitive_scales=tuple(full_graph_config.primitive_scales),
        )
        checkpoint_oracle = HybridAStarValueOracle(
            checkpoint_oracle_config,
            cache_dir=Path(output_dir) / 'checkpoint_oracle_cache',
        )
        checkpoint_local = _u_trap_local_dataset(
            eval_env,
            checkpoint_oracle,
            args._scenario_data,
            seed=20260823,
        )
        checkpoint_successor = _successor_ranking_dataset(
            eval_env,
            checkpoint_oracle,
            checkpoint_local,
            checkpoint_oracle_config,
            seed=20260823,
        )
        checkpoint_immediate_costs = np.asarray(
            [
                record['immediate_cost']
                for record in checkpoint_successor['details']
            ],
            dtype=np.float32,
        )
        constraint_checkpoint_data = (
            checkpoint_local,
            checkpoint_successor,
            checkpoint_immediate_costs,
            _regression_metrics,
            _ranking_accuracy,
        )
        oracle_generation_time += perf_counter() - checkpoint_oracle_started
        if getattr(args, 'init_checkpoint', None) and os.path.exists(
            constraint_checkpoint_path
        ):
            with open(
                constraint_checkpoint_path,
                'r',
                encoding='utf-8',
                newline='',
            ) as handle:
                constraint_checkpoint_rows = list(csv.DictReader(handle))
        logger.info(
            'Prepared checkpoint-only U-trap diagnostics: local=%d, successors=%d; '
            'Oracle labels never enter the training loss',
            len(checkpoint_local['values']),
            len(checkpoint_successor['oracle_scores']),
        )

    def run_constraint_checkpoint_diagnostics(
        step: int,
        current_batch_data,
    ) -> None:
        nonlocal evaluation_time, constraint_checkpoint_rows
        if constraint_checkpoint_data is None or current_batch_data is None:
            return
        started = perf_counter()
        local, successor, immediate_costs, regression_metrics, ranking_accuracy = (
            constraint_checkpoint_data
        )
        agent.eval()
        if device != eval_device:
            agent.to(eval_device)
        try:
            adapter = QRLGoalValueAdapter(
                agent,
                eval_env,
                eval_device,
                distance_scale=1.0,
            )
            local_predictions = adapter.batch_value(
                local['observations'],
                local['goals'],
            )
            local_metrics = regression_metrics(
                local_predictions,
                local['values'],
            )
            successor_values = adapter.batch_value(
                successor['observations'],
                successor['goals'],
            )
            successor_accuracy, successor_pairs = ranking_accuracy(
                immediate_costs + successor_values,
                successor['oracle_scores'],
                successor['groups'],
            )
            row: Dict[str, Any] = {
                'step': int(step),
                'u_trap_local_mae': float(local_metrics['mae']),
                'u_trap_local_pearson': local_metrics['pearson'],
                'u_trap_local_spearman': local_metrics['spearman'],
                'successor_ranking_accuracy': successor_accuracy,
                'successor_ranking_pairs': int(successor_pairs),
            }
            audit_batch = current_batch_data.to(eval_device)
            critic = agent.critics[0]
            with torch.no_grad():
                zx, zy = critic.encoder(
                    torch.stack(
                        [audit_batch.observations, audit_batch.next_observations]
                    )
                ).unbind(0)
                distances = critic.quasimetric_model(zx, zy).reshape(-1)
            costs = (-audit_batch.rewards).to(
                device=eval_device,
                dtype=distances.dtype,
            ).reshape_as(distances).clamp_min(0)
            terminal_mask = audit_batch.transition_infos[
                'abstract_goal_edge'
            ].to(device=eval_device, dtype=torch.bool)
            direct_mask = audit_batch.transition_infos[
                'full_graph_direct_goal_edge'
            ].to(device=eval_device, dtype=torch.bool) & ~terminal_mask
            family_masks = {
                'ordinary': ~direct_mask & ~terminal_mask,
                'direct_goal': direct_mask,
                'terminal_goal': terminal_mask,
            }
            family_fractions = []
            for family_name, mask in family_masks.items():
                family_distances = distances[mask]
                family_excess = (
                    family_distances - costs[mask]
                ).clamp_min(0)
                fraction = (family_excess > 0).to(
                    dtype=distances.dtype
                ).mean()
                family_fractions.append(fraction)
                row[f'{family_name}_dist'] = float(
                    family_distances.mean().item()
                )
                row[f'{family_name}_max_dist'] = float(
                    family_distances.max().item()
                )
                row[f'{family_name}_violation_fraction'] = float(
                    fraction.item()
                )
                row[f'{family_name}_mean_excess'] = float(
                    family_excess.mean().item()
                )
                row[f'{family_name}_max_excess'] = float(
                    family_excess.max().item()
                )
                row[f'{family_name}_sq_deviation'] = float(
                    family_excess.square().mean().item()
                )
            all_excess = (distances - costs).clamp_min(0)
            row['overall_violation_fraction'] = float(
                (all_excess > 0).to(dtype=distances.dtype).mean().item()
            )
            row['overall_mean_excess'] = float(all_excess.mean().item())
            row['overall_max_excess'] = float(all_excess.max().item())
            population_counts = audit_batch.transition_infos[
                'full_graph_constraint_population_counts'
            ].to(device=eval_device, dtype=distances.dtype)
            row['overall_graph_weighted_violation_fraction'] = float(
                (
                    torch.stack(family_fractions) * population_counts
                ).sum().div(population_counts.sum()).item()
            )
            local_constraint = losses.critic_losses[0].local_constraint
            for family_name, parameter_name in (
                ('ordinary', 'raw_lagrange_multiplier'),
                ('direct_goal', 'raw_direct_goal_lagrange_multiplier'),
                ('terminal_goal', 'raw_terminal_goal_lagrange_multiplier'),
            ):
                parameter = getattr(local_constraint, parameter_name)
                row[f'{family_name}_lagrange_mult'] = float(
                    torch.nn.functional.softplus(parameter).detach().cpu().item()
                )
            constraint_checkpoint_rows = [
                existing
                for existing in constraint_checkpoint_rows
                if int(existing['step']) != int(step)
            ]
            constraint_checkpoint_rows.append(row)
            _write_metric_rows_csv(
                constraint_checkpoint_path,
                constraint_checkpoint_rows,
            )
            for name, value in row.items():
                if name != 'step' and value is not None:
                    writer.add_scalar(
                        f'constraint_checkpoint/{name}',
                        float(value),
                        int(step),
                    )
            logger.info(
                'Constraint checkpoint %d: terminal mean/max=%.6f/%.6f, '
                'direct violation=%.4f max_excess=%.4f, ordinary violation=%.4f, '
                'lambdas=%.3f/%.3f/%.3f, local Pearson=%s, successor ranking=%s',
                int(step),
                float(row.get('terminal_goal_dist', float('nan'))),
                float(row.get('terminal_goal_max_excess', float('nan'))),
                float(row.get('direct_goal_violation_fraction', float('nan'))),
                float(row.get('direct_goal_max_excess', float('nan'))),
                float(row.get('ordinary_violation_fraction', float('nan'))),
                float(row.get('ordinary_lagrange_mult', float('nan'))),
                float(row.get('direct_goal_lagrange_mult', float('nan'))),
                float(row.get('terminal_goal_lagrange_mult', float('nan'))),
                row.get('u_trap_local_pearson'),
                row.get('successor_ranking_accuracy'),
            )
        finally:
            if device != eval_device:
                agent.to(device)
            agent.train()
            evaluation_time += perf_counter() - started
    oracle_bank_enabled = bool(getattr(args, 'oracle_bank_eval', False))
    if oracle_bank_enabled and args.env_type != 'comm_inspection_dubins_uav':
        raise ValueError("--oracle-bank-eval 仅支持 comm_inspection_dubins_uav")

    validation_oracle_bank = None
    validation_oracle_path: Optional[Path] = None
    final_test_oracle_path: Optional[Path] = None
    oracle_config: Optional[CommInspectionOracleBankConfig] = None
    oracle_validation_rows: List[Dict[str, Any]] = []
    oracle_validation_metrics_path = os.path.join(
        output_dir,
        'oracle_validation_metrics.csv',
    )
    if (
        getattr(args, 'init_checkpoint', None)
        and os.path.exists(oracle_validation_metrics_path)
    ):
        with open(
            oracle_validation_metrics_path,
            'r',
            encoding='utf-8',
            newline='',
        ) as handle:
            oracle_validation_rows = list(csv.DictReader(handle))

    if oracle_bank_enabled:
        oracle_bank_dir = Path(
            args.oracle_bank_dir
            if getattr(args, 'oracle_bank_dir', None)
            else os.path.join(output_dir, 'oracle_banks')
        )
        validation_oracle_path = Path(
            args.oracle_validation_bank
            if getattr(args, 'oracle_validation_bank', None)
            else oracle_bank_dir
            / f'hybrid_astar_validation_{int(args.oracle_bank_size)}.json'
        )
        final_test_oracle_path = Path(
            args.oracle_final_test_bank
            if getattr(args, 'oracle_final_test_bank', None)
            else oracle_bank_dir
            / f'hybrid_astar_final_test_{int(args.oracle_bank_size)}.json'
        )
        oracle_config = CommInspectionOracleBankConfig(
            sample_count=int(args.oracle_bank_size),
            generation_seed=int(args.oracle_bank_seed),
            candidate_multiplier=int(args.oracle_candidate_multiplier),
            position_resolution=float(args.oracle_astar_position_resolution),
            heading_bins=int(args.oracle_astar_heading_bins),
            primitive_steps=int(args.oracle_astar_primitive_steps),
            heuristic_weight=float(args.oracle_astar_heuristic_weight),
            max_expansions=int(args.oracle_astar_max_expansions),
            timeout_sec=float(args.oracle_astar_timeout_sec),
            terminal_samples=int(args.oracle_astar_terminal_samples),
        )
        logger.info(
            "准备固定 validation oracle bank: path=%s, samples=%d, timeout=%.1fs",
            validation_oracle_path,
            int(oracle_config.sample_count),
            float(oracle_config.timeout_sec),
        )
        oracle_start = perf_counter()
        validation_oracle_bank = ensure_comm_inspection_oracle_bank(
            eval_env,
            validation_oracle_path,
            split='validation',
            config=oracle_config,
        )
        oracle_generation_time += perf_counter() - oracle_start
        logger.info(
            "Validation oracle bank 就绪: solved=%d/%d, coverage=%.3f",
            int(validation_oracle_bank['summary']['solved_samples']),
            int(validation_oracle_bank['summary']['requested_samples']),
            float(validation_oracle_bank['summary']['oracle_coverage']),
        )
        if float(validation_oracle_bank['summary']['oracle_coverage']) < 0.9:
            logger.warning(
                "Validation oracle coverage 低于 0.90；MAE/MSE/相关性只基于成功求解样本，"
                "解释结果时必须同时报告 coverage。"
            )
    
    # 初始化训练状态
    logger.info("开始训练...")
    optimization_start = perf_counter()
    optimization_core_time = float(prior_timing.get('optimization_core_time_sec', 0.0))
    optim_steps = loaded_optim_steps
    train_metric_rows = []
    prior_metrics_path = os.path.join(output_dir, 'train_metrics.csv')
    if getattr(args, 'init_checkpoint', None) and os.path.exists(prior_metrics_path):
        with open(prior_metrics_path, 'r', encoding='utf-8', newline='') as f:
            train_metric_rows = list(csv.DictReader(f))
    critic_training_enabled = not bool(getattr(args, 'skip_critic_training', False))
    critic_steps_remaining = max(0, int(args.total_steps) - int(optim_steps))
    loss_result = None
    batch_data = None
    pbar = tqdm(total=critic_steps_remaining, desc="训练进度")

    if not getattr(args, 'init_checkpoint', None):
        checkpoint_path = os.path.join(output_dir, 'checkpoint_00000.pth')
        checkpoint_start = perf_counter()
        torch.save({
            'optim_steps': optim_steps,
            'agent': agent.state_dict(),
            'losses': losses.state_dict(),
            'training_mode': training_mode,
        }, checkpoint_path)
        checkpoint_io_time += perf_counter() - checkpoint_start
        logger.info(f"保存初始检查点: {checkpoint_path}")
    elif critic_training_enabled and critic_steps_remaining > 0:
        logger.info(f"将从已加载 checkpoint 继续训练 QRL critic，剩余步数: {critic_steps_remaining}")
    elif not critic_training_enabled:
        logger.info("已启用 --skip-critic-training，跳过前半段 QRL critic 训练")
    
    # 训练循环
    epoch = 0
    while critic_training_enabled and optim_steps < args.total_steps:
        for batch_data in dataloader:
            if optim_steps >= args.total_steps:
                break

            optimization_step_start = perf_counter()
            batch_data = batch_data.to(device)
            
            # 前向传播和反向传播
            loss_result = losses(agent, batch_data, optimize=True)
            optimization_core_time += perf_counter() - optimization_step_start
            
            optim_steps += 1
            pbar.update(1)
            
            # 记录日志
            if optim_steps % args.log_interval == 0:
                def log_dict(d, prefix='', step=optim_steps):
                    for k, v in d.items():
                        if isinstance(v, dict):
                            log_dict(v, prefix=f"{prefix}{k}/", step=step)
                        elif isinstance(v, torch.Tensor):
                            writer.add_scalar(f"{prefix}{k}", v.mean().item(), step)
                        else:
                            writer.add_scalar(f"{prefix}{k}", v, step)
                
                log_dict(loss_result.info, prefix='train/')
                writer.add_scalar('train/total_loss', loss_result.loss.item(), optim_steps)
                train_row = {
                    'step': int(optim_steps),
                    'total_loss': float(loss_result.loss.item()),
                }
                train_row.update(_flatten_scalar_info(loss_result.info))
                train_metric_rows.append(train_row)
                # 监控一步距离与 TD 类偏差（便于验证收敛与震荡）
                try:
                    lc = loss_result.info.get('critic_00', {}).get('local_constraint', {})
                    if lc and 'dist' in lc:
                        d = lc['dist']
                        one_step_dist = d.mean().item() if hasattr(d, 'mean') else (d.item() if hasattr(d, 'item') else float(d))
                        writer.add_scalar('train/one_step_dist', one_step_dist, optim_steps)
                        target_cost = lc.get('target_cost_mean', None)
                        if target_cost is not None:
                            target_cost_val = (
                                target_cost.mean().item()
                                if hasattr(target_cost, 'mean')
                                else (target_cost.item() if hasattr(target_cost, 'item') else float(target_cost))
                            )
                        else:
                            target_cost_val = getattr(losses.critic_losses[0].local_constraint, 'step_cost', 1.0)
                        writer.add_scalar('train/td_like_error', one_step_dist - target_cost_val, optim_steps)
                except Exception:
                    pass
                
                # 打印到控制台（包含详细信息）
                loss_str = f"Step {optim_steps}: total_loss={loss_result.loss.item():.4f}"

                def scalar_for_log(value):
                    if isinstance(value, torch.Tensor):
                        return value.mean().item()
                    if isinstance(value, (int, float)):
                        return float(value)
                    return None
                
                # 提取各个损失组件信息（如果有）
                if hasattr(loss_result, 'info') and isinstance(loss_result.info, dict):
                    # 尝试提取 critic 损失信息
                    for key in loss_result.info.keys():
                        if 'critic' in key and isinstance(loss_result.info[key], dict):
                            crit_info = loss_result.info[key]
                            if 'local_constraint' in crit_info:
                                lc_info = crit_info['local_constraint']
                                if isinstance(lc_info, dict):
                                    violation = lc_info.get('violation', 'N/A')
                                    sq_dev = lc_info.get('sq_deviation', 'N/A')
                                    dist = lc_info.get('dist', 'N/A')
                                    target_cost = lc_info.get('target_cost_mean', 'N/A')
                                    violation = scalar_for_log(violation)
                                    sq_dev = scalar_for_log(sq_dev)
                                    dist = scalar_for_log(dist)
                                    target_cost = scalar_for_log(target_cost)
                                    loss_str += f" | violation={violation:.4f}" if violation is not None else ""
                                    loss_str += f" | sq_dev={sq_dev:.4f}" if sq_dev is not None else ""
                                    loss_str += f" | dist={dist:.4f}" if dist is not None else ""
                                    loss_str += f" | target_cost_mean={target_cost:.4f}" if target_cost is not None else ""
                                    for family_name in (
                                        'ordinary',
                                        'direct_goal',
                                        'terminal_goal',
                                    ):
                                        family = lc_info.get(family_name)
                                        if not isinstance(family, dict):
                                            continue
                                        fraction = scalar_for_log(
                                            family.get('violation_fraction')
                                        )
                                        max_excess = scalar_for_log(
                                            family.get('max_excess')
                                        )
                                        lagrange = scalar_for_log(
                                            family.get('lagrange_mult')
                                        )
                                        if fraction is not None:
                                            loss_str += (
                                                f" | {family_name}_viol={fraction:.4f}"
                                            )
                                        if max_excess is not None:
                                            loss_str += (
                                                f" | {family_name}_max={max_excess:.4f}"
                                            )
                                        if lagrange is not None:
                                            loss_str += (
                                                f" | lambda_{family_name}={lagrange:.4f}"
                                            )
                
                pbar.set_postfix_str(loss_str)
                logger.info(loss_str)
            
            # 评估和可视化
            if args.eval_interval > 0 and optim_steps % args.eval_interval == 0:
                evaluation_start = perf_counter()
                logger.info(f"Step {optim_steps}: 开始评估...")
                agent.eval()
                # MPS 时临时迁到 CPU 做评估与热力图，避免 Metal buffer 错误
                if device != eval_device:
                    agent.to(eval_device)
                try:
                    # 通信巡检使用固定 Hybrid A* validation bank；其他环境沿用原评估。
                    if validation_oracle_bank is not None:
                        eval_metrics = evaluate_qrl_on_oracle_bank(
                            agent,
                            validation_oracle_bank,
                            device=eval_device,
                            distance_scale=1.0,
                        )
                        eval_tag_prefix = 'eval_oracle'
                        oracle_row = {
                            'optim_steps': int(optim_steps),
                            **eval_metrics,
                        }
                        oracle_validation_rows = [
                            row
                            for row in oracle_validation_rows
                            if int(row['optim_steps']) != int(optim_steps)
                        ]
                        oracle_validation_rows.append(oracle_row)
                        oracle_validation_rows.sort(
                            key=lambda row: int(row['optim_steps'])
                        )
                        _write_metric_rows_csv(
                            oracle_validation_metrics_path,
                            oracle_validation_rows,
                        )
                    else:
                        eval_metrics = evaluate_quasimetric(
                            agent=agent,
                            env=eval_env,
                            n_pairs=args.eval_n_pairs,
                            device=eval_device_str,
                            seed=args.seed + optim_steps,
                            distance_scale=evaluation_distance_scale,
                        )
                        eval_tag_prefix = 'eval'
                    
                    # 记录到 TensorBoard
                    for key, value in eval_metrics.items():
                        writer.add_scalar(
                            f'{eval_tag_prefix}/{key}',
                            value,
                            optim_steps,
                        )
                    
                    # 打印评估结果
                    eval_str = f"评估结果: MSE={eval_metrics['mse']:.4f}, "
                    if 'rmse' in eval_metrics:
                        eval_str += f"RMSE={eval_metrics['rmse']:.4f}, "
                    eval_str += f"MAE={eval_metrics['mae']:.4f}, "
                    eval_str += f"Spearman={eval_metrics['spearman_corr']:.4f}, "
                    eval_str += f"Pearson={eval_metrics['pearson_corr']:.4f}"
                    if 'oracle_coverage' in eval_metrics:
                        eval_str += (
                            f", Oracle Coverage={eval_metrics['oracle_coverage']:.3f}"
                        )
                    logger.info(eval_str)
                    
                    # Planning / Reachability 评估（仅对 obstacle 环境）
                    if args.env_type == 'obstacle' and args.planning_eval_interval > 0:
                        if optim_steps % args.planning_eval_interval == 0:
                            try:
                                logger.info("开始 Planning / Reachability 评估...")
                                execution_modes = [m.strip() for m in args.planning_execution_modes.split(",") if m.strip()]
                                distance_types = [d.strip() for d in args.lookahead_distance_types.split(",") if d.strip()]
                                if not distance_types:
                                    distance_types = ["qrl"]

                                # 统一处理：对每个 distance_type 进行评估
                                for dist_type in distance_types:
                                    if len(distance_types) > 1:
                                        logger.info(f"Planning / Reachability 评估（distance_type={dist_type})...")
                                    
                                    # 构建 lookahead 配置
                                    lookahead_cfg = None
                                    if "lookahead" in execution_modes:
                                        lookahead_cfg = LookaheadConfig(
                                            horizon=args.lookahead_horizon,
                                            num_sequences=args.lookahead_num_sequences,
                                            step_cost_weight=args.lookahead_step_cost_weight,
                                            collision_penalty=args.lookahead_collision_penalty,
                                            distance_type=dist_type,
                                        )

                                    planning_results = evaluate_planning(
                                        agent=agent,
                                        env=eval_env,
                                        n_trials=args.planning_eval_n_trials,
                                        device=eval_device_str,
                                        seed=args.seed + optim_steps,
                                        num_action_candidates=args.planning_num_action_candidates,
                                        visualize_failures=(args.planning_visualize_failures and 
                                                           optim_steps % args.planning_visualize_interval == 0),
                                        output_dir=output_dir,
                                        step=optim_steps,
                                        execution_modes=execution_modes,
                                        lookahead_config=lookahead_cfg,
                                    )
                                    
                                    # 确定 TensorBoard tag 前缀（单一 QRL 时保持向后兼容）
                                    prefix = 'planning' if (distance_types == ["qrl"] and dist_type == "qrl") else f'planning/{dist_type}'
                                    
                                    # 统一记录和打印结果
                                    _log_planning_results(planning_results, execution_modes, writer, logger, optim_steps, prefix)
                                
                            except Exception as e:
                                logger.warning(f"Planning 评估失败: {e}")
                    
                    # 评估可视化（oracle 模式画固定 bank；其他模式画距离场）
                    if args.visualization_interval > 0 and optim_steps % args.visualization_interval == 0:
                        try:
                            if validation_oracle_bank is not None:
                                heatmap_path = visualize_qrl_oracle_bank_heatmap(
                                    agent,
                                    validation_oracle_bank,
                                    step=optim_steps,
                                    output_dir=output_dir,
                                    device=eval_device,
                                    distance_scale=1.0,
                                )
                                heatmap_tag = "eval_oracle/bank_heatmap"
                                logger.info(
                                    "已保存 Hybrid A* oracle bank 热力图: "
                                    f"{heatmap_path}"
                                )
                            else:
                                heatmap_path = visualize_distance_field_heatmap(
                                    agent=agent,
                                    env=eval_env,
                                    goal=None,
                                    step=optim_steps,
                                    output_dir=output_dir,
                                    device=eval_device_str,
                                    distance_scale=evaluation_distance_scale,
                                )
                                heatmap_tag = "eval/distance_heatmap"
                                logger.info(f"已保存距离场热力图: {heatmap_path}")
                            # 将图像添加到 TensorBoard
                            try:
                                from PIL import Image
                                img = Image.open(heatmap_path)
                                img_array = np.array(img)
                                writer.add_image(heatmap_tag, img_array, optim_steps, dataformats='HWC')
                            except ImportError:
                                # 如果没有 PIL，使用 matplotlib 读取
                                import matplotlib.image as mpimg
                                img_array = mpimg.imread(heatmap_path)
                                writer.add_image(heatmap_tag, img_array, optim_steps, dataformats='HWC')
                        except Exception as e:
                            logger.warning(f"可视化失败: {e}")
                    
                except Exception as e:
                    logger.warning(f"评估失败: {e}")
                finally:
                    if device != eval_device:
                        agent.to(device)
                    agent.train()
                    evaluation_time += perf_counter() - evaluation_start
            
            # 保存检查点
            if optim_steps % args.save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{optim_steps:05d}.pth')
                checkpoint_start = perf_counter()
                torch.save({
                    'optim_steps': optim_steps,
                    'agent': agent.state_dict(),
                    'losses': losses.state_dict(),
                    'training_mode': str(getattr(args, 'comm_dataset_mode', 'standard')),
                }, checkpoint_path)
                checkpoint_io_time += perf_counter() - checkpoint_start
                run_constraint_checkpoint_diagnostics(
                    optim_steps,
                    batch_data,
                )
                with open(timing_progress_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        {
                            'optim_steps': int(optim_steps),
                            'data_time_sec': float(data_time_sec),
                            'optimization_core_time_sec': float(optimization_core_time),
                            'evaluation_time_sec': float(evaluation_time),
                            'oracle_generation_time_sec': float(
                                oracle_generation_time
                            ),
                            'checkpoint_io_time_sec': float(checkpoint_io_time),
                            'end_to_end_time_sec': float(
                                prior_timing.get('end_to_end_time_sec', 0.0)
                                + perf_counter()
                                - end_to_end_start
                            ),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                logger.info(f"保存检查点: {checkpoint_path}")
        
        epoch += 1
    
    pbar.close()
    optimization_elapsed = perf_counter() - optimization_start
    
    final_path = os.path.join(output_dir, 'checkpoint_final.pth')
    checkpoint_start = perf_counter()
    torch.save({
        'optim_steps': optim_steps,
        'agent': agent.state_dict(),
        'losses': losses.state_dict(),
        'training_mode': str(getattr(args, 'comm_dataset_mode', 'standard')),
    }, final_path)
    checkpoint_io_time += perf_counter() - checkpoint_start
    logger.info(f"保存最终模型: {final_path}")
    if (
        constraint_checkpoint_data is not None
        and optim_steps % args.save_interval != 0
    ):
        run_constraint_checkpoint_diagnostics(optim_steps, batch_data)
    train_metrics_path = os.path.join(output_dir, 'train_metrics.csv')
    _write_metric_rows_csv(train_metrics_path, train_metric_rows)
    logger.info(f"保存训练指标 CSV: {train_metrics_path}")

    if (
        oracle_bank_enabled
        and oracle_config is not None
        and final_test_oracle_path is not None
    ):
        logger.info(
            "准备独立 final-test oracle bank: path=%s, samples=%d, timeout=%.1fs",
            final_test_oracle_path,
            int(oracle_config.sample_count),
            float(oracle_config.timeout_sec),
        )
        oracle_start = perf_counter()
        final_test_oracle_bank = ensure_comm_inspection_oracle_bank(
            eval_env,
            final_test_oracle_path,
            split='final_test',
            config=oracle_config,
        )
        oracle_generation_time += perf_counter() - oracle_start
        logger.info(
            "Final-test oracle bank 就绪: solved=%d/%d, coverage=%.3f",
            int(final_test_oracle_bank['summary']['solved_samples']),
            int(final_test_oracle_bank['summary']['requested_samples']),
            float(final_test_oracle_bank['summary']['oracle_coverage']),
        )
        if float(final_test_oracle_bank['summary']['oracle_coverage']) < 0.9:
            logger.warning(
                "Final-test oracle coverage 低于 0.90；最终指标不能脱离 coverage "
                "单独作为全任务误差结论。"
            )

        agent.eval()
        if device != eval_device:
            agent.to(eval_device)
        try:
            final_test_metrics = evaluate_qrl_on_oracle_bank(
                agent,
                final_test_oracle_bank,
                device=eval_device,
                distance_scale=1.0,
                bootstrap_samples=int(args.oracle_final_bootstrap_samples),
                bootstrap_seed=int(args.oracle_bank_seed) + 200_000_033,
            )
            for key, value in final_test_metrics.items():
                writer.add_scalar(f'final_test_oracle/{key}', value, optim_steps)
            final_test_payload = {
                'checkpoint': os.path.abspath(final_path),
                'optim_steps': int(optim_steps),
                'bank_path': str(final_test_oracle_path.resolve()),
                'bank_summary': final_test_oracle_bank['summary'],
                'metrics': final_test_metrics,
            }
            final_test_metrics_path = os.path.join(
                output_dir,
                'oracle_final_test_metrics.json',
            )
            with open(
                final_test_metrics_path,
                'w',
                encoding='utf-8',
            ) as handle:
                json.dump(
                    final_test_payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(
                "Final-test oracle: MAE=%.4f, RMSE=%.4f, Spearman=%.4f, "
                "coverage=%.3f；结果保存至 %s",
                float(final_test_metrics['mae']),
                float(final_test_metrics['rmse']),
                float(final_test_metrics['spearman_corr']),
                float(final_test_metrics['oracle_coverage']),
                final_test_metrics_path,
            )
        finally:
            if device != eval_device:
                agent.to(device)
            agent.train()

    if (
        args.env_type == 'comm_inspection_dubins_uav'
        and getattr(args, 'hierarchical_mode', 'none') == 'subgoal_actor'
        and int(getattr(args, 'subgoal_train_steps', 0)) > 0
    ):
        logger.info("开始第二阶段：冻结 critic，监督训练 SubgoalActor...")
        subgoal_actor = SubgoalActor(
            obs_dim=int(eval_env.observation_space.shape[0]),
            hidden_dim=int(args.subgoal_actor_hidden_dim),
        ).to(device)
        subgoal_ckpt_init = os.path.join(output_dir, 'subgoal_actor_checkpoint_00000.pth')
        save_subgoal_actor_checkpoint(
            subgoal_ckpt_init,
            subgoal_actor,
            train_step=0,
            metadata={
                'hierarchical_mode': args.hierarchical_mode,
                'subgoal_candidates': int(args.subgoal_candidates),
                'subgoal_lambda_final': float(args.subgoal_lambda_final),
                'subgoal_lambda_task': float(args.subgoal_lambda_task),
                'taskscore_beta_obs': float(args.taskscore_beta_obs),
                'taskscore_beta_comm': float(args.taskscore_beta_comm),
                'taskscore_beta_feas': float(args.taskscore_beta_feas),
                'taskscore_margin_clip': float(args.taskscore_margin_clip),
            },
        )
        logger.info(f"保存初始 SubgoalActor 检查点: {subgoal_ckpt_init}")

        agent.eval()
        qrl_value_agent = QRLGoalValueAdapter(agent, env=eval_env, device=device)

        def _log_subgoal_metrics(step: int, metrics: dict):
            global_step = optim_steps + step
            for key, value in metrics.items():
                writer.add_scalar(f'subgoal_actor/{key}', value, global_step)
            if step == 1 or step % max(1, args.log_interval) == 0:
                logger.info(
                    "SubgoalActor step %d: loss=%.4f pos=%.4f heading=%.4f raw_valid=%.3f repair=%.3f",
                    step,
                    float(metrics.get('loss', 0.0)),
                    float(metrics.get('pos_loss', 0.0)),
                    float(metrics.get('heading_loss', 0.0)),
                    float(metrics.get('raw_actor_output_valid_rate', 0.0)),
                    float(metrics.get('mean_repair_distance', 0.0)),
                )

        def _save_subgoal_checkpoint(step: int, metrics: dict):
            ckpt_path = os.path.join(output_dir, f'subgoal_actor_checkpoint_{step:05d}.pth')
            save_subgoal_actor_checkpoint(
                ckpt_path,
                subgoal_actor,
                train_step=step,
                metadata={
                    'metrics': metrics,
                    'hierarchical_mode': args.hierarchical_mode,
                    'subgoal_candidates': int(args.subgoal_candidates),
                    'subgoal_lambda_final': float(args.subgoal_lambda_final),
                    'subgoal_lambda_task': float(args.subgoal_lambda_task),
                    'high_level_period': int(args.high_level_period),
                    'taskscore_beta_obs': float(args.taskscore_beta_obs),
                    'taskscore_beta_comm': float(args.taskscore_beta_comm),
                    'taskscore_beta_feas': float(args.taskscore_beta_feas),
                    'taskscore_margin_clip': float(args.taskscore_margin_clip),
                },
            )
            logger.info(f"保存 SubgoalActor 检查点: {ckpt_path}")

        final_subgoal_metrics = train_subgoal_actor(
            actor=subgoal_actor,
            agent=qrl_value_agent,
            env_factory=create_env_fn,
            device=device,
            cfg=SubgoalActorTrainConfig(
                train_steps=int(args.subgoal_train_steps),
                batch_size=int(args.subgoal_batch_size),
                lr=float(args.subgoal_actor_lr),
                hidden_dim=int(args.subgoal_actor_hidden_dim),
                num_candidates=int(args.subgoal_candidates),
                lambda_final=float(args.subgoal_lambda_final),
                lambda_task=float(args.subgoal_lambda_task),
                seed=int(args.seed),
                save_interval=int(args.subgoal_save_interval),
            ),
            log_fn=_log_subgoal_metrics,
            checkpoint_fn=_save_subgoal_checkpoint,
        )
        subgoal_ckpt_final = os.path.join(output_dir, 'subgoal_actor_checkpoint_final.pth')
        save_subgoal_actor_checkpoint(
            subgoal_ckpt_final,
            subgoal_actor,
            train_step=int(args.subgoal_train_steps),
            metadata={
                'final_metrics': final_subgoal_metrics,
                'hierarchical_mode': args.hierarchical_mode,
                'subgoal_candidates': int(args.subgoal_candidates),
                'subgoal_lambda_final': float(args.subgoal_lambda_final),
                'subgoal_lambda_task': float(args.subgoal_lambda_task),
                'high_level_period': int(args.high_level_period),
                'taskscore_beta_obs': float(args.taskscore_beta_obs),
                'taskscore_beta_comm': float(args.taskscore_beta_comm),
                'taskscore_beta_feas': float(args.taskscore_beta_feas),
                'taskscore_margin_clip': float(args.taskscore_margin_clip),
            },
        )
        logger.info(f"保存最终 SubgoalActor 检查点: {subgoal_ckpt_final}")

    end_to_end_time_sec = float(prior_timing.get('end_to_end_time_sec', 0.0)) + (
        perf_counter() - end_to_end_start
    )
    timing_payload = {
        'seed': int(args.seed),
        'scenario_config': os.path.abspath(args.scenario_config) if getattr(args, 'scenario_config', None) else None,
        'data_time_sec': float(data_time_sec),
        'optimization_core_time_sec': float(
            optimization_core_time
        ),
        'optimization_phase_time_sec': float(optimization_elapsed),
        'evaluation_time_sec': float(evaluation_time),
        'oracle_generation_time_sec': float(oracle_generation_time),
        'checkpoint_io_time_sec': float(checkpoint_io_time),
        'end_to_end_time_sec': float(end_to_end_time_sec),
        'global_gradient_updates': int(optim_steps),
        'resumed_from_checkpoint': (
            os.path.abspath(args.init_checkpoint) if getattr(args, 'init_checkpoint', None) else None
        ),
        'dataset_total_transitions': int(len(dataset)),
        'collection_stats': collection_stats or None,
        'qrl_objective': {
            'training_mode': str(getattr(args, 'comm_dataset_mode', 'standard')),
            'global_push': True,
            'global_push_transform': 'linear_negative_mean',
            'global_push_abstract_goal_ratio': float(args.global_push_abstract_goal_ratio),
            'global_push_state_goal_ratio': float(args.global_push_state_goal_ratio),
            'global_push_source_distribution': (
                'uniform_goal_reachable_nonterminal_lattice_states'
                if full_graph_config is not None
                else 'dataset_transition_sources'
            ),
            'local_transition_constraint': True,
            'constraint_enforcement': (
                'three_families_separate_duals_all_goal_bound_edges_per_batch'
                if stratified_full_graph_constraints
                else 'unified_batch_mean_single_dual'
            ),
            'ordinary_edges_per_batch': (
                int(args.batch_size) if stratified_full_graph_constraints else None
            ),
            'direct_goal_edges_per_batch': (
                243 if stratified_full_graph_constraints else None
            ),
            'terminal_goal_edges_per_batch': (
                24 if stratified_full_graph_constraints else None
            ),
            'ordinary_epsilon': 0.25,
            'direct_goal_epsilon': float(args.full_graph_direct_goal_epsilon),
            'terminal_goal_epsilon': float(args.full_graph_terminal_goal_epsilon),
            'lagrange_optimization': True,
            'latent_transition_model': True,
            'abstract_goal_zero_edge': float(args.abstract_goal_edge_loss_weight),
            'temporal_multistep_weight': float(args.qrl_temporal_constraint_weight),
            'success_return_weight': float(args.qrl_goal_return_constraint_weight),
            'nstep_bootstrap_weight': float(args.qrl_nstep_goal_constraint_weight),
            'success_transition_sampling_weight': float(args.qrl_success_transition_weight),
            'oracle_value_labels': False,
        },
        'hardware': {
            'requested_device': str(args.device),
            'resolved_device': str(device),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': str(torch.__version__),
            'cuda_version': str(torch.version.cuda) if torch.version.cuda else None,
        },
    }
    with open(os.path.join(output_dir, 'timing.json'), 'w', encoding='utf-8') as f:
        json.dump(timing_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"保存训练计时: {os.path.join(output_dir, 'timing.json')}")
    writer.flush()
    writer.close()
    
    # 标记完成
    with open(os.path.join(output_dir, 'COMPLETE'), 'w') as f:
        f.write('')
    logger.info("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='最小可运行 QRL 训练脚本（环境无关版本）')
    
    # 训练参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='auto', 
                        help='设备 (auto/cpu/cuda/mps)，auto 会自动选择最佳设备')
    parser.add_argument('--output-dir', type=str, default='./results/minimal_qrl', help='输出目录')
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help='可选：加载已有 QRL checkpoint 作为初始化，再继续训练或仅训练 SubgoalActor')
    parser.add_argument('--skip-critic-training', action='store_true',
                        help='跳过前半段 QRL critic 训练；通常与 --init-checkpoint 配合，只训练 SubgoalActor')
    parser.add_argument('--scenario-config', type=str, default=None,
                        help='权威的工业园区场景 JSON；提供后覆盖所有环境参数')
    
    # 环境参数
    parser.add_argument('--env-type', type=str, default='simple_grid',
                        choices=['simple_grid', 'obstacle', 'maze2d', 'mountaincar', 'dubins_uav', 'comm_inspection_dubins_uav'],
                        help='环境类型: simple_grid, obstacle, maze2d, mountaincar, dubins_uav, 或 comm_inspection_dubins_uav')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                        help='网格大小 (height, width)，仅用于 simple_grid 环境')
    parser.add_argument('--grid-resolution', type=int, default=50,
                        help='A* 搜索的网格分辨率，仅用于 obstacle 环境（降低可加速评估）')
    parser.add_argument('--mountaincar-goal-position', type=float, default=0.5,
                        help='MountainCar 目标位置')
    parser.add_argument('--mountaincar-goal-velocity', type=float, default=0.0,
                        help='MountainCar 点目标速度；threshold 模式下仅用于可视化/点目标评估')
    parser.add_argument('--mountaincar-goal-tolerance-pos', type=float, default=0.015,
                        help='MountainCar 点目标位置容差')
    parser.add_argument('--mountaincar-goal-tolerance-vel', type=float, default=0.01,
                        help='MountainCar 点目标速度容差')
    parser.add_argument('--mountaincar-gt-pos-bins', type=int, default=160,
                        help='MountainCar ground-truth 图搜索的位置离散网格数')
    parser.add_argument('--mountaincar-gt-vel-bins', type=int, default=160,
                        help='MountainCar ground-truth 图搜索的速度离散网格数')
    parser.add_argument('--mountaincar-gt-goal-mode', type=str, default='threshold',
                        choices=['threshold', 'point'],
                        help='MountainCar ground-truth 目标定义：threshold=到达目标位置阈值，point=到达位置速度小区域')
    parser.add_argument('--mountaincar-dataset-mode', type=str, default='random_policy_paper',
                        choices=['random_policy_paper', 'discrete_graph', 'random_rollout'],
                        help='MountainCar 数据集模式：random_policy_paper=论文式随机策略离线数据并加入 abstract-goal transitions')
    parser.add_argument('--mountaincar-abstract-goal-transition-repeats', type=int, default=15,
                        help='MountainCar goal-set 到 abstract goal 的附加 transition 重复次数，用于近似论文中的 5% abstract-goal 采样')
    parser.add_argument('--mountaincar-encoder-arch', type=int, nargs='*', default=[1024, 1024, 1024],
                        help='MountainCar encoder hidden sizes')
    parser.add_argument('--mountaincar-transition-arch', type=int, nargs='*', default=[1024, 1024, 1024],
                        help='MountainCar latent transition hidden sizes')
    parser.add_argument('--mountaincar-projector-arch', type=int, nargs='*', default=[1024, 1024, 1024],
                        help='MountainCar quasimetric projector hidden sizes')
    parser.add_argument('--mountaincar-latent-size', type=int, default=256,
                        help='MountainCar latent dimension')
    parser.add_argument('--mountaincar-iqe-dim', type=int, default=512,
                        help='MountainCar IQE projected dimension; 16 components x 32 dim in the paper')
    parser.add_argument('--mountaincar-iqe-components', type=int, default=16,
                        help='MountainCar IQE component count')
    parser.add_argument('--mountaincar-transition-loss-weight', type=float, default=75.0,
                        help='MountainCar latent transition loss weight')
    parser.add_argument('--mountaincar-model-lr', type=float, default=5e-4,
                        help='MountainCar model parameter learning rate')
    parser.add_argument('--mountaincar-lambda-lr', type=float, default=0.3,
                        help='MountainCar Lagrange multiplier learning rate')
    parser.add_argument('--mountaincar-epsilon', type=float, default=0.25,
                        help='MountainCar local constraint epsilon')
    parser.add_argument('--mountaincar-lambda-init', type=float, default=0.01,
                        help='MountainCar initial Lagrange multiplier')
    parser.add_argument('--mountaincar-global-offset', type=float, default=500.0,
                        help='MountainCar global push softplus offset')
    parser.add_argument('--mountaincar-global-beta', type=float, default=0.01,
                        help='MountainCar global push softplus beta')
    
    # Dubins UAV 特定参数
    parser.add_argument('--bounds', type=float, nargs=4, default=None,
                        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
                        help='地图边界，仅用于 dubins_uav 环境')
    parser.add_argument('--omega-max', type=float, default=None,
                        help='最大角速度（弧度/秒），仅用于 dubins_uav 环境')
    parser.add_argument('--v', type=float, default=None,
                        help='固定前进速度（单位/秒），仅用于 dubins_uav 环境')
    parser.add_argument('--dt', type=float, default=None,
                        help='时间步长（秒），仅用于 dubins_uav 环境')
    parser.add_argument('--epsilon-pos', type=float, default=None,
                        help='位置到达目标的容差，仅用于 dubins_uav 环境')
    parser.add_argument('--epsilon-theta', type=float, default=None,
                        help='朝向到达目标的容差（弧度），仅用于 dubins_uav 环境')
    parser.add_argument('--collision-penalty', type=float, default=-10.0,
                        help='碰撞时的负奖励，仅用于 dubins_uav 环境')
    parser.add_argument('--obstacle-config', type=str, default='none',
                        choices=['none', 'simple', 'medium', 'hard'],
                        help='Dubins 障碍预设：none=无, simple/medium/hard 为圆形障碍')
    parser.add_argument('--obstacles', type=float, nargs='*', default=None,
                        help='Dubins 自定义圆障 (x1 y1 r1 x2 y2 r2 ...)，若提供则忽略 --obstacle-config')
    parser.add_argument('--use-cos-sin-obs', action='store_true', default=True,
                        help='Dubins 使用 (x,y,cosθ,sinθ) 作为观测（默认 True）')
    parser.add_argument('--no-use-cos-sin-obs', action='store_false', dest='use_cos_sin_obs',
                        help='禁用 cos/sin 观测，使用 (x,y,θ)')

    # 通信感知巡检 Dubins 环境特定参数
    parser.add_argument('--device-catalog', type=str, default=None,
                        help='工业设备目录 JSON；comm_inspection_dubins_uav 必需')
    parser.add_argument('--comm-alpha', type=float, default=2.0,
                        help='通信对数路径损耗系数 alpha，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--comm-bias', type=float, default=5.0,
                        help='通信质量偏置项，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--comm-occlusion-penalty', type=float, default=6.0,
                        help='通信 LOS 被遮挡时的额外惩罚，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--comm-threshold', type=float, default=0.0,
                        help='通信可行性阈值，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--require-ground-station-los', action='store_true',
                        help='要求到地面站具有 LOS，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--collision-cost', type=float, default=10.0,
                        help='碰撞阶段代价，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--out-of-bounds-cost', type=float, default=10.0,
                        help='越界阶段代价，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--communication-break-cost', type=float, default=1.0,
                        help='通信不可行时的固定阶段代价，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--observation-violation-cost-weight', type=float, default=1.0,
                        help='观测约束短缺的软代价权重，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--communication-violation-cost-weight', type=float, default=0.5,
                        help='通信约束短缺的软代价权重，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--observation-failure-cost', type=float, default=0.25,
                        help='观测不可行时的固定阶段代价，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--taskscore-beta-obs', type=float, default=1.0,
                        help='TaskScore 中 observation margin 的权重')
    parser.add_argument('--taskscore-beta-comm', type=float, default=1.0,
                        help='TaskScore 中 communication margin 的权重')
    parser.add_argument('--taskscore-beta-feas', type=float, default=0.5,
                        help='TaskScore 中 task feasible bonus 的权重')
    parser.add_argument('--taskscore-margin-clip', type=float, default=2.0,
                        help='TaskScore 对 obs/comm margin 的对称裁剪阈值')
    parser.add_argument('--qrl-cost-source', type=str, default='negative_reward',
                        choices=['negative_reward', 'fixed'],
                        help='comm_inspection_dubins_uav 的 QRL local constraint 单步代价来源：'
                             'negative_reward 使用环境 task cost；fixed 使用原始固定 step_cost=1.0')
    parser.add_argument('--global-push-softplus-offset', type=float, default=15.0,
                        help='goal-set GlobalPush softplus offset；控制 push 梯度开始衰减的距离尺度')
    parser.add_argument('--global-push-softplus-beta', type=float, default=0.1,
                        help='goal-set GlobalPush softplus beta；越小则大距离范围内的衰减越平滑')
    parser.add_argument('--global-push-abstract-goal-ratio', type=float, default=0.6,
                        help='goal-set GlobalPush 主项权重：普通状态到当前上下文抽象 G_task')
    parser.add_argument('--global-push-state-goal-ratio', type=float, default=0.4,
                        help='goal-set GlobalPush 辅助项权重：同上下文普通 state-state 几何结构')
    parser.add_argument('--abstract-goal-edge-loss-weight', type=float, default=1.0,
                        help='抽象零代价边 d(s_terminal, G_task)^2 的损失权重')
    parser.add_argument(
        '--qrl-temporal-constraint-weight',
        type=float,
        default=1.0,
        help='真实交互轨迹 state-to-future-state 多步上界损失权重',
    )
    parser.add_argument(
        '--qrl-temporal-min-future-steps',
        type=int,
        default=2,
        help='多步轨迹上界的最小时序跨度；2 表示排除已有单步约束',
    )
    parser.add_argument(
        '--qrl-goal-return-constraint-weight',
        type=float,
        default=1.0,
        help='自然成功交互轨迹到 G_task 的 remaining-cost 上界损失权重',
    )
    parser.add_argument(
        '--qrl-nstep-goal-constraint-weight',
        type=float,
        default=0.0,
        help='可选 EMA target-critic n-step task-goal 自举上界权重；默认关闭',
    )
    parser.add_argument(
        '--qrl-nstep-target-tau',
        type=float,
        default=0.005,
        help='n-step task-goal 自举中 EMA target critic 的软更新率',
    )
    parser.add_argument(
        '--qrl-success-transition-weight',
        type=float,
        default=4.0,
        help='自然成功轨迹 transition 在 offline dataloader 中的相对采样权重',
    )
    parser.add_argument('--task-aware-teacher-ratio', type=float, default=1.0,
                        help='通信巡检 goal-set 数据中额外追加的 Dubins guidance 成功轨迹比例；'
                             '1.0 表示每个 random rollout 额外收集 1 条 teacher 轨迹')
    parser.add_argument(
        '--comm-dataset-mode',
        choices=[
            'standard',
            'qrl_explore',
            'dense_transition_original',
            'full_graph_goal_set',
            'full_graph_goal_set_stratified_constraints',
        ],
        default='standard',
        help='通信巡检数据模式：standard=现有 random+teacher；'
             'qrl_explore=覆盖驱动、局部安全的无目标探索；'
             'dense_transition_original=global replay + exhaustive real U-trap lattice edges；'
             'full_graph_goal_set=validated full lattice macro-edges + explicit unified G；'
             'full_graph_goal_set_stratified_constraints=same graph with three constraint families',
    )
    parser.add_argument('--dense-transition-device-id', default='u_trap_target')
    parser.add_argument('--dense-transition-position-resolution', type=float, default=0.25)
    parser.add_argument('--dense-transition-heading-bins', type=int, default=24)
    parser.add_argument('--dense-transition-primitive-steps', type=int, default=5)
    parser.add_argument(
        '--dense-transition-primitive-scales',
        type=float,
        nargs='+',
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument('--dense-transition-local-fraction', type=float, default=0.5)
    parser.add_argument('--dense-transition-failure-results', default=None)
    parser.add_argument('--dense-transition-failure-position-radius', type=float, default=0.75)
    parser.add_argument('--dense-transition-failure-heading-radius', type=float, default=0.65)
    parser.add_argument('--full-graph-device-id', default='u_trap_target')
    parser.add_argument('--full-graph-position-resolution', type=float, default=0.25)
    parser.add_argument('--full-graph-heading-bins', type=int, default=24)
    parser.add_argument('--full-graph-primitive-steps', type=int, default=5)
    parser.add_argument(
        '--full-graph-primitive-scales',
        type=float,
        nargs='+',
        default=[-1.0, -0.5, 0.0, 0.5, 1.0],
    )
    parser.add_argument('--full-graph-uniform-push-seed', type=int, default=20260824)
    parser.add_argument('--full-graph-direct-goal-epsilon', type=float, default=0.25)
    parser.add_argument('--full-graph-terminal-goal-epsilon', type=float, default=0.0)
    parser.add_argument(
        '--explore-attempted-env-steps',
        type=int,
        default=200_000,
        help='QRL-explore 实际执行且全部保留的环境步预算',
    )
    parser.add_argument(
        '--explore-start-position-resolution',
        type=float,
        default=1.0,
        help='QRL-explore 固定自由空间起点库的位置网格分辨率',
    )
    parser.add_argument(
        '--explore-start-heading-bins',
        type=int,
        default=12,
        help='QRL-explore 固定起点库的航向分层数',
    )
    parser.add_argument(
        '--explore-action-hold-min-steps',
        type=int,
        default=3,
        help='QRL-explore 每段持久随机角速度的最短保持步数',
    )
    parser.add_argument(
        '--explore-action-hold-max-steps',
        type=int,
        default=10,
        help='QRL-explore 每段持久随机角速度的最长保持步数',
    )
    parser.add_argument(
        '--explore-straight-action-probability',
        type=float,
        default=0.5,
        help='QRL-explore 每个动作段选择直行的概率；其余均匀选择左右大小曲率',
    )
    parser.add_argument(
        '--explore-exclusion-task-bank',
        type=str,
        default=None,
        help='QRL-explore 起点库需要排除的 validation/test task-bank JSON',
    )
    parser.add_argument(
        '--explore-exclusion-radius',
        type=float,
        default=0.25,
        help='QRL-explore 起点与评估 task-bank 起点的最小平面距离',
    )
    parser.add_argument(
        '--explore-start-boundary-margin',
        type=float,
        default=0.5,
        help='QRL-explore 固定起点相对地图边界的安全缓冲',
    )
    parser.add_argument(
        '--explore-local-safety-lookahead-steps',
        type=int,
        default=10,
        help='QRL-explore 目标无关局部碰撞重采样的前视步数',
    )
    
    parser.add_argument('--num-episodes', type=int, default=100, help='数据集中的 episode 数量')
    parser.add_argument('--target-env-transitions', type=int, default=None,
                        help='通信巡检数据集的精确真实转移预算；抽象目标边不计入')
    parser.add_argument('--max-steps-per-episode', type=int, default=200, help='每个 episode 的最大步数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小（增大可提升GPU利用率）')
    parser.add_argument('--total-steps', type=int, default=10000, help='总训练步数')
    parser.add_argument('--num-critics', type=int, default=2, help='Critic 数量')
    
    # 日志和保存
    parser.add_argument('--log-interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔')
    
    # 评估参数
    parser.add_argument('--eval-interval', type=int, default=1000, help='评估间隔（步数）')
    parser.add_argument(
        '--eval-n-pairs',
        type=int,
        default=500,
        help='传统随机评估的状态-目标对数；启用 --oracle-bank-eval 时忽略',
    )
    parser.add_argument('--visualization-interval', type=int, default=1000,
                        help='可视化间隔（步数），设为0禁用可视化')
    parser.add_argument(
        '--oracle-bank-eval',
        action='store_true',
        help='通信巡检使用固定 Hybrid A* validation/final-test bank 评估 QRL',
    )
    parser.add_argument(
        '--oracle-bank-dir',
        type=str,
        default=None,
        help='Oracle bank 目录；默认 OUTPUT_DIR/oracle_banks',
    )
    parser.add_argument(
        '--oracle-validation-bank',
        type=str,
        default=None,
        help='可选的 validation bank JSON 路径',
    )
    parser.add_argument(
        '--oracle-final-test-bank',
        type=str,
        default=None,
        help='可选的 final-test bank JSON 路径',
    )
    parser.add_argument(
        '--oracle-bank-size',
        type=int,
        default=192,
        help='每个 oracle split 的样本数；24 个设备时 192=每设备 8 个',
    )
    parser.add_argument(
        '--oracle-bank-seed',
        type=int,
        default=20260729,
        help='独立于训练 seed 的固定 oracle bank 生成 seed',
    )
    parser.add_argument(
        '--oracle-candidate-multiplier',
        type=int,
        default=16,
        help='每个入选起点的分层候选数倍数',
    )
    parser.add_argument(
        '--oracle-astar-timeout-sec',
        type=float,
        default=60.0,
        help='Hybrid A* 每个 oracle 样本的超时秒数',
    )
    parser.add_argument(
        '--oracle-astar-position-resolution',
        type=float,
        default=0.25,
        help='Oracle Hybrid A* 位置离散分辨率',
    )
    parser.add_argument(
        '--oracle-astar-heading-bins',
        type=int,
        default=24,
        help='Oracle Hybrid A* 朝向离散数',
    )
    parser.add_argument(
        '--oracle-astar-primitive-steps',
        type=int,
        default=5,
        help='Oracle Hybrid A* 每个运动原语的环境步数',
    )
    parser.add_argument(
        '--oracle-astar-heuristic-weight',
        type=float,
        default=1.0,
        help='Oracle Hybrid A* 启发函数权重',
    )
    parser.add_argument(
        '--oracle-astar-max-expansions',
        type=int,
        default=50000,
        help='Oracle Hybrid A* 最大扩展节点数',
    )
    parser.add_argument(
        '--oracle-astar-terminal-samples',
        type=int,
        default=128,
        help='Oracle Hybrid A* goal-set 启发函数终态样本数',
    )
    parser.add_argument(
        '--oracle-final-bootstrap-samples',
        type=int,
        default=2000,
        help='Final-test 按设备聚类 bootstrap 的重复次数',
    )
    
    # Planning / Reachability 评估参数
    parser.add_argument('--planning-eval-interval', type=int, default=1000,
                        help='Planning 评估间隔（步数），设为0禁用 Planning 评估')
    parser.add_argument('--planning-eval-n-trials', type=int, default=100,
                        help='Planning 评估时的测试次数')
    parser.add_argument('--planning-num-action-candidates', type=int, default=32,
                        help='Planning 评估时每步候选动作数量')
    parser.add_argument('--planning-execution-modes', type=str, default='greedy',
                        help='执行机制对比：逗号分隔，例如 "greedy,lookahead"（仅影响评估，不影响训练）')
    parser.add_argument('--lookahead-horizon', type=int, default=5,
                        help='lookahead 规划步长（仅 lookahead 模式）')
    parser.add_argument('--lookahead-num-sequences', type=int, default=64,
                        help='lookahead 序列数量（仅 lookahead 模式）')
    parser.add_argument('--lookahead-step-cost-weight', type=float, default=0.0,
                        help='lookahead 步长惩罚权重（抑制抖动/绕圈，默认 0）')
    parser.add_argument('--lookahead-collision-penalty', type=float, default=0.0,
                        help='lookahead 碰撞惩罚（默认 0；ContinuousObstacle2D 碰撞 reward=-0.1）')
    parser.add_argument('--lookahead-distance-types', type=str, default='qrl',
                        help='lookahead 终端代价的 distance 类型，逗号分隔："qrl" 或 "euclidean"；默认仅使用 QRL')
    parser.add_argument('--planning-visualize-failures', action='store_true',
                        help='是否可视化失败案例')
    parser.add_argument('--planning-visualize-interval', type=int, default=2000,
                        help='Failure mode 可视化间隔（步数）')
    parser.add_argument('--hierarchical-mode', type=str, default='none',
                        choices=['none', 'subgoal_actor'],
                        help='是否在 critic 训练后追加高层 SubgoalActor 训练')
    parser.add_argument('--subgoal-train-steps', type=int, default=0,
                        help='SubgoalActor 监督训练步数')
    parser.add_argument('--subgoal-batch-size', type=int, default=32,
                        help='SubgoalActor 训练批大小')
    parser.add_argument('--subgoal-actor-lr', type=float, default=3e-4,
                        help='SubgoalActor 学习率')
    parser.add_argument('--subgoal-actor-hidden-dim', type=int, default=256,
                        help='SubgoalActor 隐层宽度')
    parser.add_argument('--subgoal-save-interval', type=int, default=1000,
                        help='SubgoalActor checkpoint 保存间隔；设为0表示只保存首尾')
    parser.add_argument('--subgoal-candidates', type=int, default=64,
                        help='每个 teacher 打分时的 subgoal 候选数')
    parser.add_argument('--high-level-period', type=int, default=5,
                        help='hierarchical 执行时的高层重规划周期')
    parser.add_argument('--subgoal-lambda-final', type=float, default=0.3,
                        help='高层 teacher 中 final goal critic 项系数')
    parser.add_argument('--subgoal-lambda-task', type=float, default=1.0,
                        help='高层 teacher 中 TaskScore 奖励项系数')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
