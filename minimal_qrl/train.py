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
from quasimetric_rl.modules.quasimetric_critic import QuasimetricCriticConf
from quasimetric_rl.modules.quasimetric_critic.losses import QuasimetricCriticLosses
from quasimetric_rl.modules.quasimetric_critic.losses.local_constraint import LocalConstraintLoss
from quasimetric_rl.data import BatchData, Dataset, EpisodeData, register_offline_env
from minimal_qrl.envs import SimpleGrid2D, ContinuousObstacle2D, DubinsUAV2D
from minimal_qrl.dataset import create_dataset
from minimal_qrl.eval import evaluate_quasimetric, visualize_distance_field_heatmap, evaluate_planning, LookaheadConfig


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
        env_type: 环境类型 ('simple_grid', 'obstacle', 或 'dubins_uav')
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
    else:
        raise ValueError(f"未知的环境类型: {env_type}")


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
        # 初步训练默认：小地图、无障碍、固定 v/dt、大 omega_max、cos/sin 观测
        kwargs = {
            'max_episode_steps': args.max_steps_per_episode,
            'bounds': tuple(args.bounds) if (hasattr(args, 'bounds') and args.bounds) else (0.0, 0.0, 5.0, 5.0),
            'omega_max': args.omega_max if (hasattr(args, 'omega_max') and args.omega_max is not None) else 3.0,
            'v': args.v if (hasattr(args, 'v') and args.v is not None) else 1.0,
            'dt': args.dt if (hasattr(args, 'dt') and args.dt is not None) else 0.1,
            'epsilon_pos': args.epsilon_pos if (hasattr(args, 'epsilon_pos') and args.epsilon_pos is not None) else 0.15,
            'epsilon_theta': args.epsilon_theta if (hasattr(args, 'epsilon_theta') and args.epsilon_theta is not None) else 0.2,
            'obstacles': [],  # 初步训练无障碍
            'use_cos_sin_obs': getattr(args, 'use_cos_sin_obs', True),
        }
        if hasattr(args, 'collision_penalty') and args.collision_penalty is not None:
            kwargs['collision_penalty'] = args.collision_penalty
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
    
    # 创建 QRL Agent 和 Losses（Dubins 时 step_cost=dt 以学习 time-to-go）
    logger.info("创建 QRL Agent 和 Losses...")
    step_cost = 1.0
    if args.env_type == 'dubins_uav':
        step_cost = env_kwargs.get('dt', 0.1)
        agent_conf = QRLConf(
            actor=None,
            num_critics=args.num_critics,
            quasimetric_critic=QuasimetricCriticConf(
                losses=QuasimetricCriticLosses.Conf(
                    local_constraint=LocalConstraintLoss.Conf(step_cost=step_cost),
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
        total_optim_steps=args.total_steps,
    )
    agent.to(device)
    losses.to(device)
    logger.info(f"Agent: {agent}")
    logger.info(f"Losses: {losses}")
    
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
    optim_steps = 0
    pbar = tqdm(total=args.total_steps, desc="训练进度")
    
    # 保存初始检查点
    checkpoint_path = os.path.join(output_dir, 'checkpoint_00000.pth')
    torch.save({
        'optim_steps': optim_steps,
        'agent': agent.state_dict(),
        'losses': losses.state_dict(),
    }, checkpoint_path)
    logger.info(f"保存初始检查点: {checkpoint_path}")
    
    # 训练循环
    epoch = 0
    while optim_steps < args.total_steps:
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
    
    # 保存最终模型
    final_path = os.path.join(output_dir, 'checkpoint_final.pth')
    torch.save({
        'optim_steps': optim_steps,
        'agent': agent.state_dict(),
        'losses': losses.state_dict(),
    }, final_path)
    logger.info(f"保存最终模型: {final_path}")
    
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
    
    # 环境参数
    parser.add_argument('--env-type', type=str, default='simple_grid',
                        choices=['simple_grid', 'obstacle', 'dubins_uav'],
                        help='环境类型: simple_grid (简单网格), obstacle (障碍物环境), 或 dubins_uav (Dubins UAV)')
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
    parser.add_argument('--collision-penalty', type=float, default=None,
                        help='碰撞时的负奖励，仅用于 dubins_uav 环境')
    parser.add_argument('--use-cos-sin-obs', action='store_true', default=True,
                        help='Dubins 使用 (x,y,cosθ,sinθ) 作为观测（默认 True）')
    parser.add_argument('--no-use-cos-sin-obs', action='store_false', dest='use_cos_sin_obs',
                        help='禁用 cos/sin 观测，使用 (x,y,θ)')
    
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
    parser.add_argument('--lookahead-distance-types', type=str, default='qrl',
                        help='lookahead 终端代价的 distance 类型，逗号分隔："qrl" 或 "euclidean"；默认仅使用 QRL')
    parser.add_argument('--planning-visualize-failures', action='store_true',
                        help='是否可视化失败案例')
    parser.add_argument('--planning-visualize-interval', type=int, default=2000,
                        help='Failure mode 可视化间隔（步数）')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

