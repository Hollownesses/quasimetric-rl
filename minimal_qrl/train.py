#!/usr/bin/env python3
"""
最小可运行 QRL 核心训练脚本（环境无关版本）
使用核心 QRL 模块，不依赖 d4rl/mujoco/gym 等复杂环境
支持多种环境：SimpleGrid2D, ContinuousObstacle2D 等
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import *
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses
from quasimetric_rl.modules.optim import AdamWSpec
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from quasimetric_rl.data import BatchData, Dataset, EpisodeData, register_offline_env
from minimal_qrl.envs import (
    SimpleGrid2D,
    ContinuousObstacle2D,
    DubinsUAV2D,
    CircleObstacle,
    CommInspectionDubinsUAV2D,
)
from minimal_qrl.dataset import create_dataset
from minimal_qrl.eval import evaluate_quasimetric, visualize_distance_field_heatmap, evaluate_planning, LookaheadConfig
from minimal_qrl.eval.dubins_execution_mode_eval import DubinsLookaheadConfig
from minimal_qrl.gc_agents import QRLGoalValueAdapter
from minimal_qrl.subgoal_actor import (
    SubgoalActor,
    SubgoalActorTrainConfig,
    save_subgoal_actor_checkpoint,
    train_subgoal_actor,
)
from minimal_qrl.cost_aware_subgoal_scorer import (
    CostAwareSubgoalScorer,
    CostAwareSubgoalScorerTrainConfig,
    TOP_MODEL_COST_CONTEXT_KEYS,
    build_top_model_cost_context,
    save_cost_aware_subgoal_scorer_checkpoint,
    train_cost_aware_subgoal_scorer,
)


def setup_logging(output_dir: str):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_env_factory(env_type: str, **env_kwargs):
    """
    创建环境工厂函数
    
    Args:
        env_type: 环境类型 ('simple_grid', 'obstacle', 'dubins_uav', 或 'comm_inspection_dubins_uav')
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
        observation_mode = getattr(args, 'observation_mode', None)
        if not observation_mode:
            observation_mode = 'cos_sin' if getattr(args, 'use_cos_sin_obs', True) else 'state'
        kwargs = {
            'max_steps': args.max_steps_per_episode,
            'bounds': tuple(args.bounds) if (hasattr(args, 'bounds') and args.bounds) else (0.0, 0.0, 10.0, 10.0),
            'omega_max': args.omega_max if (hasattr(args, 'omega_max') and args.omega_max is not None) else 3.0,
            'v': args.v if (hasattr(args, 'v') and args.v is not None) else 1.0,
            'dt': args.dt if (hasattr(args, 'dt') and args.dt is not None) else 0.1,
            'obstacles': obstacles,
            'observation_mode': observation_mode,
            'inspection_target': tuple(args.inspection_target) if getattr(args, 'inspection_target', None) else None,
            'ground_station': tuple(args.ground_station) if getattr(args, 'ground_station', None) else None,
            'randomize_inspection_target': getattr(args, 'randomize_inspection_target', False),
            'randomize_ground_station': getattr(args, 'randomize_ground_station', False),
            'observation_radius': getattr(args, 'observation_radius', 1.5),
            'fov_angle': getattr(args, 'fov_angle', np.pi / 2.0),
            'require_target_los': getattr(args, 'require_target_los', False),
            'comm_alpha': getattr(args, 'comm_alpha', 2.0),
            'comm_bias': getattr(args, 'comm_bias', 5.0),
            'comm_occlusion_penalty': getattr(args, 'comm_occlusion_penalty', 6.0),
            'comm_threshold': getattr(args, 'comm_threshold', 0.0),
            'require_ground_station_los': getattr(args, 'require_ground_station_los', False),
            'goal_sampling_mode': getattr(args, 'goal_sampling_mode', 'task_feasible'),
            'goal_position_tolerance': getattr(args, 'goal_position_tolerance', 0.15),
            'goal_heading_tolerance': getattr(args, 'goal_heading_tolerance', 0.2),
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


def train(args):
    """训练主函数"""
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
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
            probe_goal = precheck_env.sample_goal(seed=args.seed)
            logger.info(f"通信巡检环境预检查通过，示例任务可行目标: {probe_goal}")
        except RuntimeError as e:
            raise ValueError(
                "当前通信巡检环境配置下不存在可采样的任务可行目标。"
                "请检查 inspection_target / ground_station / obstacle_config / "
                "observation_radius / fov_angle / require_target_los / "
                "comm_threshold 等参数是否过于严格。"
            ) from e
    
    # 创建环境工厂函数
    create_env_fn = create_env_factory(args.env_type, **env_kwargs)
    
    # 注册环境（如果还没注册）
    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
    env_key = (args.env_type, args.env_type)  # 使用 env_type 作为 name
    if env_key not in CREATE_ENV_REGISTRY:
        def load_episodes():
            env = create_env_fn()
            return create_dataset(
                env=env,
                num_episodes=args.num_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
                sample_valid_states=True,
                seed=args.seed,
            )
        
        register_offline_env(
            args.env_type, args.env_type,  # 使用 env_type 作为 name
            create_env_fn=create_env_fn,
            load_episodes_fn=load_episodes,
        )
        logger.info(f"已注册环境: {env_key}")
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset_conf = Dataset.Conf(
        kind=args.env_type,
        name=args.env_type,  # 使用 env_type 作为 name
        future_observation_discount=0.99,
    )
    dataset = dataset_conf.make(dummy=False)
    logger.info(f"数据集大小: {len(dataset)} 个转移")
    
    # 创建 QRL Agent 和 Losses（Dubins 用 step_cost=1.0 使约束与网络输出尺度一致，评估时再乘 dt 得时间）
    logger.info("创建 QRL Agent 和 Losses...")
    step_cost = 1.0
    if args.env_type in {'dubins_uav', 'comm_inspection_dubins_uav'}:
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
    )
    
    # 创建 TensorBoard writer（使用带时间戳的子目录，便于区分不同训练）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard', timestamp)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"TensorBoard 日志目录: {tensorboard_dir}")
    
    # 创建环境实例用于评估
    eval_env = create_env_fn()
    
    # 初始化训练状态
    logger.info("开始训练...")
    optim_steps = loaded_optim_steps
    critic_training_enabled = not bool(getattr(args, 'skip_critic_training', False))
    critic_steps_remaining = max(0, int(args.total_steps) - int(optim_steps))
    pbar = tqdm(total=critic_steps_remaining, desc="训练进度")

    if not getattr(args, 'init_checkpoint', None):
        checkpoint_path = os.path.join(output_dir, 'checkpoint_00000.pth')
        torch.save({
            'optim_steps': optim_steps,
            'agent': agent.state_dict(),
            'losses': losses.state_dict(),
        }, checkpoint_path)
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
            
            batch_data = batch_data.to(device)
            
            # 前向传播和反向传播
            loss_result = losses(agent, batch_data, optimize=True)
            
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
                # 监控一步距离与 TD 类偏差（便于验证收敛与震荡）
                try:
                    lc = loss_result.info.get('critic_00', {}).get('local_constraint', {})
                    if lc and 'dist' in lc:
                        d = lc['dist']
                        one_step_dist = d.mean().item() if hasattr(d, 'mean') else (d.item() if hasattr(d, 'item') else float(d))
                        writer.add_scalar('train/one_step_dist', one_step_dist, optim_steps)
                        step_cost_val = getattr(losses.critic_losses[0].local_constraint, 'step_cost', 1.0)
                        writer.add_scalar('train/td_like_error', one_step_dist - step_cost_val, optim_steps)
                except Exception:
                    pass
                
                # 打印到控制台（包含详细信息）
                loss_str = f"Step {optim_steps}: total_loss={loss_result.loss.item():.4f}"
                
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
                                    loss_str += f" | violation={violation:.4f}" if isinstance(violation, (int, float)) else ""
                                    loss_str += f" | sq_dev={sq_dev:.4f}" if isinstance(sq_dev, (int, float)) else ""
                                    loss_str += f" | dist={dist:.4f}" if isinstance(dist, (int, float)) else ""
                
                pbar.set_postfix_str(loss_str)
                logger.info(loss_str)
            
            # 评估和可视化
            if optim_steps % args.eval_interval == 0:
                logger.info(f"Step {optim_steps}: 开始评估...")
                agent.eval()
                # MPS 时临时迁到 CPU 做评估与热力图，避免 Metal buffer 错误
                if device != eval_device:
                    agent.to(eval_device)
                try:
                    # 评估 quasimetric
                    eval_metrics = evaluate_quasimetric(
                        agent=agent,
                        env=eval_env,
                        n_pairs=args.eval_n_pairs,
                        device=eval_device_str,
                        seed=args.seed + optim_steps,
                    )
                    
                    # 记录到 TensorBoard
                    for key, value in eval_metrics.items():
                        writer.add_scalar(f'eval/{key}', value, optim_steps)
                    
                    # 打印评估结果
                    eval_str = f"评估结果: MSE={eval_metrics['mse']:.4f}, "
                    eval_str += f"MAE={eval_metrics['mae']:.4f}, "
                    eval_str += f"Spearman={eval_metrics['spearman_corr']:.4f}, "
                    eval_str += f"Pearson={eval_metrics['pearson_corr']:.4f}"
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
                    
                    # 可视化距离场（按照 visualization_interval 间隔执行）
                    if args.visualization_interval > 0 and optim_steps % args.visualization_interval == 0:
                        try:
                            heatmap_path = visualize_distance_field_heatmap(
                                agent=agent,
                                env=eval_env,
                                goal=None,
                                step=optim_steps,
                                output_dir=output_dir,
                                device=eval_device_str,
                            )
                            logger.info(f"已保存距离场热力图: {heatmap_path}")
                            # 将图像添加到 TensorBoard
                            try:
                                from PIL import Image
                                img = Image.open(heatmap_path)
                                img_array = np.array(img)
                                writer.add_image('eval/distance_heatmap', img_array, optim_steps, dataformats='HWC')
                            except ImportError:
                                # 如果没有 PIL，使用 matplotlib 读取
                                import matplotlib.image as mpimg
                                img_array = mpimg.imread(heatmap_path)
                                writer.add_image('eval/distance_heatmap', img_array, optim_steps, dataformats='HWC')
                        except Exception as e:
                            logger.warning(f"可视化失败: {e}")
                    
                except Exception as e:
                    logger.warning(f"评估失败: {e}")
                finally:
                    if device != eval_device:
                        agent.to(device)
                    agent.train()
            
            # 保存检查点
            if optim_steps % args.save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{optim_steps:05d}.pth')
                torch.save({
                    'optim_steps': optim_steps,
                    'agent': agent.state_dict(),
                    'losses': losses.state_dict(),
                }, checkpoint_path)
                logger.info(f"保存检查点: {checkpoint_path}")
        
        epoch += 1
    
    pbar.close()
    
    final_path = os.path.join(output_dir, 'checkpoint_final.pth')
    torch.save({
        'optim_steps': optim_steps,
        'agent': agent.state_dict(),
        'losses': losses.state_dict(),
    }, final_path)
    logger.info(f"保存最终模型: {final_path}")

    qrl_value_agent = None
    subgoal_actor = None
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

    if (
        args.env_type == 'comm_inspection_dubins_uav'
        and getattr(args, 'subgoal_selector_mode', 'heuristic') == 'cost_aware'
        and int(getattr(args, 'top_model_train_steps', 0)) > 0
    ):
        if subgoal_actor is None or qrl_value_agent is None:
            raise ValueError(
                "训练 cost-aware 顶层模型前，需要先完成 SubgoalActor 训练。"
                "请确保 hierarchical_mode=subgoal_actor 且 subgoal_train_steps > 0。"
            )
        logger.info("开始第三阶段：冻结 QRL critic + SubgoalActor，训练 CostAwareSubgoalScorer...")
        top_model_rollout_steps = (
            int(args.top_model_rollout_steps)
            if getattr(args, 'top_model_rollout_steps', None) is not None
            else int(args.high_level_period)
        )
        top_model = CostAwareSubgoalScorer(
            obs_dim=int(eval_env.observation_space.shape[0]),
            hidden_dim=int(args.top_model_hidden_dim),
        ).to(device)
        top_model_ckpt_init = os.path.join(output_dir, 'cost_aware_subgoal_scorer_checkpoint_00000.pth')
        lookahead_cfg = DubinsLookaheadConfig(
            horizon=int(args.lookahead_horizon),
            num_sequences=int(args.lookahead_num_sequences),
            step_cost_weight=float(args.lookahead_step_cost_weight),
            collision_penalty=float(args.lookahead_collision_penalty),
            alpha_subgoal=float(args.planner_alpha_subgoal),
            alpha_final=float(args.planner_alpha_final),
            alpha_task_terminal=float(args.planner_alpha_task_terminal),
            use_env_stage_cost=bool(args.planner_use_env_stage_cost),
        )
        top_model_metadata = {
            'subgoal_selector_mode': args.subgoal_selector_mode,
            'top_model_rollout_steps': int(top_model_rollout_steps),
            'planner_alpha_subgoal': float(args.planner_alpha_subgoal),
            'planner_alpha_final': float(args.planner_alpha_final),
            'planner_alpha_task_terminal': float(args.planner_alpha_task_terminal),
            'planner_use_env_stage_cost': bool(args.planner_use_env_stage_cost),
            'cost_context_keys': list(TOP_MODEL_COST_CONTEXT_KEYS),
            'taskscore_beta_obs': float(args.taskscore_beta_obs),
            'taskscore_beta_comm': float(args.taskscore_beta_comm),
            'taskscore_beta_feas': float(args.taskscore_beta_feas),
            'taskscore_margin_clip': float(args.taskscore_margin_clip),
            'cost_context_default': build_top_model_cost_context(eval_env, lookahead_cfg).tolist(),
        }
        save_cost_aware_subgoal_scorer_checkpoint(
            top_model_ckpt_init,
            top_model,
            train_step=0,
            metadata=top_model_metadata,
        )
        logger.info(f"保存初始 CostAwareSubgoalScorer 检查点: {top_model_ckpt_init}")

        def _log_top_model_metrics(step: int, metrics: dict):
            global_step = optim_steps + int(args.subgoal_train_steps) + step
            for key, value in metrics.items():
                writer.add_scalar(f'cost_aware_top_model/{key}', value, global_step)
            if step == 1 or step % max(1, args.log_interval) == 0:
                logger.info(
                    "CostAwareTopModel step %d: loss=%.4f top1=%.3f pred=%.3f label=%.3f",
                    step,
                    float(metrics.get('loss', 0.0)),
                    float(metrics.get('top1_match_rate', 0.0)),
                    float(metrics.get('mean_selected_pred_cost', 0.0)),
                    float(metrics.get('mean_selected_rollout_cost_label', 0.0)),
                )

        def _save_top_model_checkpoint(step: int, metrics: dict):
            ckpt_path = os.path.join(output_dir, f'cost_aware_subgoal_scorer_checkpoint_{step:05d}.pth')
            save_cost_aware_subgoal_scorer_checkpoint(
                ckpt_path,
                top_model,
                train_step=step,
                metadata={
                    **top_model_metadata,
                    'metrics': metrics,
                    'high_level_period': int(args.high_level_period),
                },
            )
            logger.info(f"保存 CostAwareSubgoalScorer 检查点: {ckpt_path}")

        final_top_model_metrics = train_cost_aware_subgoal_scorer(
            scorer=top_model,
            actor=subgoal_actor,
            agent=qrl_value_agent,
            env_factory=create_env_fn,
            actor_device=device,
            scorer_device=device,
            lookahead_cfg=lookahead_cfg,
            cfg=CostAwareSubgoalScorerTrainConfig(
                train_steps=int(args.top_model_train_steps),
                batch_size=int(args.top_model_batch_size),
                lr=float(args.top_model_lr),
                hidden_dim=int(args.top_model_hidden_dim),
                num_candidates=int(args.subgoal_candidates),
                rollout_steps=int(top_model_rollout_steps),
                seed=int(args.seed),
                save_interval=int(args.top_model_save_interval),
            ),
            log_fn=_log_top_model_metrics,
            checkpoint_fn=_save_top_model_checkpoint,
        )
        top_model_ckpt_final = os.path.join(output_dir, 'cost_aware_subgoal_scorer_checkpoint_final.pth')
        save_cost_aware_subgoal_scorer_checkpoint(
            top_model_ckpt_final,
            top_model,
            train_step=int(args.top_model_train_steps),
            metadata={
                **top_model_metadata,
                'final_metrics': final_top_model_metrics,
                'high_level_period': int(args.high_level_period),
            },
        )
        logger.info(f"保存最终 CostAwareSubgoalScorer 检查点: {top_model_ckpt_final}")
    
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
    
    # 环境参数
    parser.add_argument('--env-type', type=str, default='simple_grid',
                        choices=['simple_grid', 'obstacle', 'dubins_uav', 'comm_inspection_dubins_uav'],
                        help='环境类型: simple_grid, obstacle, dubins_uav, 或 comm_inspection_dubins_uav')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                        help='网格大小 (height, width)，仅用于 simple_grid 环境')
    parser.add_argument('--grid-resolution', type=int, default=50,
                        help='A* 搜索的网格分辨率，仅用于 obstacle 环境（降低可加速评估）')
    
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
    parser.add_argument('--inspection-target', type=float, nargs=2, default=None,
                        metavar=('X_T', 'Y_T'),
                        help='巡检目标位置，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--ground-station', type=float, nargs=2, default=None,
                        metavar=('X_BS', 'Y_BS'),
                        help='地面站位置，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--randomize-inspection-target', action='store_true',
                        help='每次 reset 随机采样巡检目标位置')
    parser.add_argument('--randomize-ground-station', action='store_true',
                        help='每次 reset 随机采样地面站位置')
    parser.add_argument('--observation-radius', type=float, default=1.5,
                        help='观测半径，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--fov-angle', type=float, default=float(np.pi / 2.0),
                        help='视场角全角（弧度），仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--require-target-los', action='store_true',
                        help='要求到巡检目标具有 LOS，仅用于 comm_inspection_dubins_uav')
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
    parser.add_argument('--observation-mode', type=str, default='task_context',
                        choices=['task_context', 'cos_sin', 'state'],
                        help='通信巡检 Dubins 的观测模式，默认 task_context')
    parser.add_argument('--goal-sampling-mode', type=str, default='task_feasible',
                        choices=['task_feasible', 'valid'],
                        help='目标采样方式，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--goal-position-tolerance', type=float, default=0.15,
                        help='目标位置容差，仅用于 comm_inspection_dubins_uav')
    parser.add_argument('--goal-heading-tolerance', type=float, default=0.2,
                        help='目标朝向容差，仅用于 comm_inspection_dubins_uav')
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
    
    parser.add_argument('--num-episodes', type=int, default=100, help='数据集中的 episode 数量')
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
    parser.add_argument('--eval-n-pairs', type=int, default=500, help='评估时采样的状态-目标对数（减少可加速评估）')
    parser.add_argument('--visualization-interval', type=int, default=1000,
                        help='可视化间隔（步数），设为0禁用可视化')
    
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
    parser.add_argument('--planner-alpha-subgoal', type=float, default=1.0,
                        help='cost-aware 顶层训练中 low-level planner 的 subgoal 终端项系数')
    parser.add_argument('--planner-alpha-final', type=float, default=0.3,
                        help='cost-aware 顶层训练中 low-level planner 的 final goal 终端项系数')
    parser.add_argument('--planner-alpha-task-terminal', type=float, default=0.5,
                        help='cost-aware 顶层训练中 low-level planner 的 terminal task score 系数')
    parser.add_argument('--planner-use-env-stage-cost', dest='planner_use_env_stage_cost', action='store_true', default=True,
                        help='cost-aware 顶层训练标签是否累计环境真实 stage cost（默认开启）')
    parser.add_argument('--no-planner-use-env-stage-cost', dest='planner_use_env_stage_cost', action='store_false',
                        help='关闭环境真实 stage cost，退回 lookahead 的动作/碰撞代价近似')
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
    parser.add_argument('--subgoal-selector-mode', type=str, default='heuristic',
                        choices=['heuristic', 'cost_aware'],
                        help='高层 subgoal 选择器模式；heuristic 使用旧 teacher 目标，cost_aware 训练额外顶层 scorer')
    parser.add_argument('--top-model-train-steps', type=int, default=0,
                        help='CostAwareSubgoalScorer 训练步数；0 表示不训练')
    parser.add_argument('--top-model-batch-size', type=int, default=16,
                        help='CostAwareSubgoalScorer 训练批大小（按状态-目标组计）')
    parser.add_argument('--top-model-lr', type=float, default=3e-4,
                        help='CostAwareSubgoalScorer 学习率')
    parser.add_argument('--top-model-hidden-dim', type=int, default=256,
                        help='CostAwareSubgoalScorer 隐层宽度')
    parser.add_argument('--top-model-save-interval', type=int, default=1000,
                        help='CostAwareSubgoalScorer checkpoint 保存间隔；设为0表示只保存首尾')
    parser.add_argument('--top-model-rollout-steps', type=int, default=None,
                        help='CostAwareSubgoalScorer 标签 rollout 步数；默认等于 high_level_period')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
