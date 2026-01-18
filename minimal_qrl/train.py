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

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses
from quasimetric_rl.data import BatchData, Dataset, EpisodeData, register_offline_env
from minimal_qrl.envs import SimpleGrid2D, ContinuousObstacle2D
from minimal_qrl.dataset import create_dataset
from minimal_qrl.evaluation import evaluate_quasimetric, visualize_distance_field_heatmap


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
        env_type: 环境类型 ('simple_grid' 或 'obstacle')
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
            'grid_resolution': getattr(args, 'grid_resolution', 100),
        }
    else:
        raise ValueError(f"未知的环境类型: {args.env_type}")


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
    
    # 获取环境参数
    env_kwargs = get_env_kwargs(args)
    logger.info(f"环境参数: {env_kwargs}")
    
    # 创建环境工厂函数
    create_env_fn = create_env_factory(args.env_type, **env_kwargs)
    
    # 注册环境（如果还没注册）
    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
    env_key = (args.env_type, args.env_name)
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
            args.env_type, args.env_name,
            create_env_fn=create_env_fn,
            load_episodes_fn=load_episodes,
        )
        logger.info(f"已注册环境: {env_key}")
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset_conf = Dataset.Conf(
        kind=args.env_type,
        name=args.env_name,
        future_observation_discount=0.99,
    )
    dataset = dataset_conf.make(dummy=False)
    logger.info(f"数据集大小: {len(dataset)} 个转移")
    
    # 创建 QRL Agent 和 Losses
    logger.info("创建 QRL Agent 和 Losses...")
    agent_conf = QRLConf(
        actor=None,  # 不训练 actor，只训练 critic
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
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # 创建环境实例用于评估
    eval_env = create_env_fn()
    
    # 训练循环
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
                # 记录损失信息
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
                
                try:
                    # 评估 quasimetric
                    eval_metrics = evaluate_quasimetric(
                        agent=agent,
                        env=eval_env,
                        n_pairs=args.eval_n_pairs,
                        device=str(device),  # 使用实际设备而非 args.device（可能是 'auto'）
                        seed=args.seed + optim_steps,  # 使用不同的种子
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
                    
                    # 可视化距离场（按照 visualization_interval 间隔执行）
                    if args.visualization_interval > 0 and optim_steps % args.visualization_interval == 0:
                        try:
                            heatmap_path = visualize_distance_field_heatmap(
                                agent=agent,
                                env=eval_env,
                                goal=None,  # 使用 env.goal_pos
                                step=optim_steps,
                                output_dir=output_dir,
                                device=str(device),
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
                        choices=['simple_grid', 'obstacle'],
                        help='环境类型: simple_grid (简单网格) 或 obstacle (障碍物环境)')
    parser.add_argument('--env-name', type=str, default='grid2d',
                        help='环境名称（用于注册，如 grid2d, obstacle2d）')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                        help='网格大小 (height, width)，仅用于 simple_grid 环境')
    parser.add_argument('--grid-resolution', type=int, default=50,
                        help='A* 搜索的网格分辨率，仅用于 obstacle 环境（降低可加速评估）')
    parser.add_argument('--num-episodes', type=int, default=100, help='数据集中的 episode 数量')
    parser.add_argument('--max-steps-per-episode', type=int, default=200, help='每个 episode 的最大步数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小（增大可提升GPU利用率）')
    parser.add_argument('--total-steps', type=int, default=5000, help='总训练步数')
    parser.add_argument('--num-critics', type=int, default=2, help='Critic 数量')
    
    # 日志和保存
    parser.add_argument('--log-interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔')
    
    # 评估参数
    parser.add_argument('--eval-interval', type=int, default=1000, help='评估间隔（步数）')
    parser.add_argument('--eval-n-pairs', type=int, default=500, help='评估时采样的状态-目标对数（减少可加速评估）')
    parser.add_argument('--visualization-interval', type=int, default=1000,
                        help='可视化间隔（步数），设为0禁用可视化')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

