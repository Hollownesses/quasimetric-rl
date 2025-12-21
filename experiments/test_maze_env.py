# 测试 SimpleMaze2D 环境
"""快速测试 2D maze 环境是否正常工作"""

import numpy as np
from maze2d_qrl import SimpleMaze2D

def test_maze_env():
    print("测试 SimpleMaze2D 环境...")
    
    # 创建环境
    env = SimpleMaze2D(maze_size=(10, 10), render_mode="human")
    
    # 测试 reset
    obs, info = env.reset()
    print(f"初始观察: {obs}")
    print(f"起始位置: {env.start_pos}")
    print(f"目标位置: {env.goal_pos}")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 渲染初始状态
    print("\n初始迷宫状态:")
    env.render()
    
    # 测试几步随机动作
    print("\n执行 10 步随机动作:")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, "
              f"obs={obs}, terminated={terminated}")
        if terminated:
            print("到达目标！")
            break
    
    # 测试到达目标
    print("\n测试到达目标:")
    obs, _ = env.reset()
    steps = 0
    max_steps = 1000
    while steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated:
            print(f"在 {steps} 步后到达目标！")
            break
    else:
        print(f"在 {max_steps} 步内未到达目标")
    
    print("\n环境测试完成！")

if __name__ == "__main__":
    test_maze_env()

