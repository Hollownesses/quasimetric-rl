#!/usr/bin/env python3
"""
测试 Dubins UAV 2D 环境
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from minimal_qrl.envs import DubinsUAV2D, Obstacle, CircleObstacle

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试 1: 基本功能（无障碍物）")
    print("=" * 60)
    
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=100,
        epsilon_pos=0.2,
        epsilon_theta=0.3,
    )
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 测试 reset
    obs, info = env.reset(seed=42)
    print(f"初始观察: {obs}")
    print(f"初始状态: x={obs[0]:.2f}, y={obs[1]:.2f}, theta={obs[2]:.3f}")
    print(f"目标状态: x={env.goal[0]:.2f}, y={env.goal[1]:.2f}, theta={env.goal[2]:.3f}")
    
    # 测试 step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, "
              f"pos_dist={info['pos_dist']:.3f}, theta_diff={info['theta_diff']:.3f}")
        if terminated or truncated:
            break
    
    print("\n✓ 基本功能测试通过\n")


def test_with_obstacles():
    """测试障碍物功能"""
    print("=" * 60)
    print("测试 2: 障碍物功能")
    print("=" * 60)
    
    obstacles = [
        Obstacle(x_min=3.0, x_max=5.0, y_min=3.0, y_max=5.0),
        Obstacle(x_min=7.0, x_max=8.0, y_min=2.0, y_max=4.0),
    ]
    
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=100,
        obstacles=obstacles,
        epsilon_pos=0.2,
        epsilon_theta=0.3,
    )
    
    # 测试状态合法性检查
    valid_state = np.array([1.0, 1.0, 0.0])
    invalid_state = np.array([4.0, 4.0, 0.0])  # 在障碍物内
    
    print(f"状态 {valid_state} 是否合法: {env.is_valid_state(valid_state)}")
    print(f"状态 {invalid_state} 是否合法: {env.is_valid_state(invalid_state)}")
    
    # 测试采样合法状态
    valid_sample = env.sample_valid_state(seed=42)
    print(f"采样合法状态: {valid_sample}")
    print(f"采样状态是否合法: {env.is_valid_state(valid_sample)}")
    
    # 测试碰撞检测
    obs, _ = env.reset(seed=42)
    print(f"\n初始状态: x={obs[0]:.2f}, y={obs[1]:.2f}, theta={obs[2]:.3f}")
    
    # 尝试向障碍物移动
    for i in range(5):
        action = np.array([0.0])  # 直行
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: x={obs[0]:.2f}, y={obs[1]:.2f}, "
              f"collision={info['collision']}, reward={reward:.3f}")
        if terminated or truncated:
            break
    
    print("\n✓ 障碍物功能测试通过\n")


def test_circle_obstacles():
    """测试圆形障碍物 (x, y, radius)"""
    print("=" * 60)
    print("测试 2b: 圆形障碍物")
    print("=" * 60)
    obstacles = [
        CircleObstacle(x=5.0, y=5.0, radius=1.0),
        CircleObstacle(x=2.0, y=8.0, radius=0.5),
    ]
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=100,
        obstacles=obstacles,
        epsilon_pos=0.2,
        epsilon_theta=0.3,
    )
    assert not env.is_valid_state(np.array([5.0, 5.0, 0.0]))
    assert env.is_valid_state(np.array([1.0, 1.0, 0.0]))
    obs, _ = env.reset(seed=123)
    for _ in range(3):
        obs, reward, term, trunc, info = env.step(np.array([0.0]))
        if term or trunc:
            break
    print("圆形障碍物 contains/intersects 与 step 正常")
    print("\n✓ 圆形障碍物测试通过\n")


def test_custom_start_goal():
    """测试自定义起点和目标"""
    print("=" * 60)
    print("测试 3: 自定义起点和目标")
    print("=" * 60)
    
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=100,
        epsilon_pos=0.2,
        epsilon_theta=0.3,
    )
    
    # 使用自定义起点和目标
    start = (1.0, 1.0, 0.0)
    goal = (9.0, 9.0, np.pi / 4)
    
    obs, _ = env.reset(seed=42, options={'start': start, 'goal': goal})
    print(f"起点: {start}")
    print(f"目标: {goal}")
    print(f"实际起点: x={obs[0]:.2f}, y={obs[1]:.2f}, theta={obs[2]:.3f}")
    print(f"实际目标: x={env.goal[0]:.2f}, y={env.goal[1]:.2f}, theta={env.goal[2]:.3f}")
    
    print("\n✓ 自定义起点和目标测试通过\n")


def test_state_save_restore():
    """测试状态保存和恢复"""
    print("=" * 60)
    print("测试 4: 状态保存和恢复")
    print("=" * 60)
    
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=1.0,
        v=1.0,
        dt=0.1,
        max_episode_steps=100,
    )
    
    obs, _ = env.reset(seed=42)
    
    # 执行几步
    for i in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
    
    # 保存状态
    state = env.get_state()
    print(f"保存状态: step={state['t']}, state={state['state']}")
    
    # 继续执行几步
    for i in range(3):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
    
    print(f"继续执行后: step={env._t}, state={env.state}")
    
    # 恢复状态
    env.set_state(state)
    print(f"恢复状态后: step={env._t}, state={env.state}")
    
    assert env._t == state['t'], "步数不匹配"
    assert np.allclose(env.state, np.array(state['state'])), "状态不匹配"
    
    print("\n✓ 状态保存和恢复测试通过\n")


def test_reach_goal():
    """测试到达目标"""
    print("=" * 60)
    print("测试 5: 到达目标")
    print("=" * 60)
    
    env = DubinsUAV2D(
        bounds=(0.0, 0.0, 10.0, 10.0),
        omega_max=2.0,  # 增大角速度以便更快转向
        v=1.0,
        dt=0.1,
        max_episode_steps=200,
        epsilon_pos=0.3,
        epsilon_theta=0.5,
    )
    
    # 设置简单的起点和目标（距离较近）
    start = (1.0, 1.0, 0.0)
    goal = (2.0, 2.0, np.pi / 4)
    
    obs, _ = env.reset(seed=42, options={'start': start, 'goal': goal})
    print(f"起点: {start}")
    print(f"目标: {goal}")
    
    # 简单的控制策略：转向目标方向
    for i in range(50):
        # 计算到目标的方向
        dx = env.goal[0] - obs[0]
        dy = env.goal[1] - obs[1]
        target_theta = np.arctan2(dy, dx)
        
        # 计算角度差
        theta_diff = target_theta - obs[2]
        # 归一化到 [-pi, pi]
        while theta_diff > np.pi:
            theta_diff -= 2 * np.pi
        while theta_diff < -np.pi:
            theta_diff += 2 * np.pi
        
        # 简单的比例控制
        omega = np.clip(theta_diff * 2.0, -env.omega_max, env.omega_max)
        action = np.array([omega])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 5 == 0:
            print(f"Step {i}: x={obs[0]:.2f}, y={obs[1]:.2f}, theta={obs[2]:.3f}, "
                  f"pos_dist={info['pos_dist']:.3f}, theta_diff={info['theta_diff']:.3f}")
        
        if terminated:
            print(f"\n✓ 成功到达目标！步数: {i+1}")
            break
        if truncated:
            print(f"\n✗ 达到最大步数，未到达目标")
            break
    
    print("\n✓ 到达目标测试完成\n")


if __name__ == '__main__':
    print("\n开始测试 Dubins UAV 2D 环境\n")
    
    try:
        test_basic_functionality()
        test_with_obstacles()
        test_circle_obstacles()
        test_custom_start_goal()
        test_state_save_restore()
        test_reach_goal()
        
        print("=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
