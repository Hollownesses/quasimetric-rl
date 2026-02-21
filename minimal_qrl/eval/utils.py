"""
评估工具函数：共享的辅助函数
"""
import torch
from quasimetric_rl.data import register_offline_env
from quasimetric_rl.data.base import CREATE_ENV_REGISTRY, LOAD_EPISODES_REGISTRY


def auto_device(device_str: str) -> torch.device:
    """
    自动选择设备
    
    Args:
        device_str: 设备字符串 ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        torch.device 对象
    """
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_registered_env(kind: str, name: str, *, create_env_fn, load_episodes_fn):
    """
    确保环境已注册（如果已注册则跳过）
    
    Args:
        kind: 环境类型
        name: 环境名称
        create_env_fn: 创建环境的函数
        load_episodes_fn: 加载 episodes 的函数
    """
    key = (kind, name)
    if key in CREATE_ENV_REGISTRY and key in LOAD_EPISODES_REGISTRY:
        return
    register_offline_env(kind, name, create_env_fn=create_env_fn, load_episodes_fn=load_episodes_fn)
