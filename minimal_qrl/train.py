#!/usr/bin/env python3
"""
最小可运行 QRL 核心训练脚本
使用核心 QRL 模块，不依赖 d4rl/mujoco/gym 等复杂环境
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
from minimal_qrl.simple_env import SimpleGrid2D
from minimal_qrl.dataset import create_simple_dataset


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
    logger.info(f"参数: {args}")
    
    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 注册环境（如果还没注册）
    from quasimetric_rl.data.base import CREATE_ENV_REGISTRY
    if ('simple_grid', 'grid2d') not in CREATE_ENV_REGISTRY:
        def create_env():
            return SimpleGrid2D(grid_size=args.grid_size)
        
        def load_episodes():
            return create_simple_dataset(
                env=SimpleGrid2D(grid_size=args.grid_size),
                num_episodes=args.num_episodes,
                max_steps_per_episode=args.max_steps_per_episode,
            )
        
        register_offline_env(
            'simple_grid', 'grid2d',
            create_env_fn=create_env,
            load_episodes_fn=load_episodes,
        )
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset_conf = Dataset.Conf(
        kind='simple_grid',
        name='grid2d',
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
    dataloader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # 简化版本，不使用多进程
    )
    
    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
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
    parser = argparse.ArgumentParser(description='最小可运行 QRL 训练脚本')
    
    # 训练参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--output-dir', type=str, default='./results/minimal_qrl', help='输出目录')
    
    # 环境参数
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10], help='网格大小 (height, width)')
    parser.add_argument('--num-episodes', type=int, default=100, help='数据集中的 episode 数量')
    parser.add_argument('--max-steps-per-episode', type=int, default=200, help='每个 episode 的最大步数')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=256, help='批次大小')
    parser.add_argument('--total-steps', type=int, default=10000, help='总训练步数')
    parser.add_argument('--num-critics', type=int, default=2, help='Critic 数量')
    
    # 日志和保存
    parser.add_argument('--log-interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

