# 2D Maze QRL 实验说明

## 概述

这个实验将 QRL (Quasimetric Reinforcement Learning) 框架应用到 2D maze 环境中，并通过 2D heatmap 可视化准度量距离，验证准度量的几何特性。

## 文件说明

- `maze2d_qrl.py`: 主要的训练和可视化代码
- `SimpleMaze2D`: 简单的 2D grid maze 环境实现

## 环境特性

### SimpleMaze2D 环境

- **状态空间**: 连续坐标 (x, y)，归一化到 [0, 1]
- **动作空间**: 离散 4 个动作 (0=上, 1=右, 2=下, 3=左)
- **奖励**: 
  - 到达目标: +1.0
  - 每步成本: -0.01
- **默认迷宫**: 10x10 网格，包含外圈墙壁和中间障碍物

## 使用方法

### 基本运行

```bash
python experiments/maze2d_qrl.py
```

### 自定义配置

可以在代码中修改 `Config` 类来调整参数：

```python
@dataclass
class Config:
    maze_size: Tuple[int, int] = (10, 10)  # 迷宫大小
    total_steps: int = 10000  # 训练步数
    viz_interval: int = 1000  # 可视化间隔
    heatmap_resolution: int = 50  # heatmap 分辨率
    # ... 其他参数
```

## 可视化输出

训练过程中会生成以下可视化：

1. **2D Heatmap**: 显示从所有位置到目标的准度量距离
   - 颜色越深表示距离越远
   - 叠加了迷宫墙壁（黑色方块）
   - 标记了起始位置（绿色圆点）和目标位置（红色星号）

2. **3D 表面图**: 3D 可视化准度量距离场

3. **准度量性质验证**: 每 1000 步输出验证结果
   - 非负性: d(s, g) >= 0
   - 自反性: d(s, s) ≈ 0
   - 三角不等式: d(s1, s3) <= d(s1, s2) + d(s2, s3)

## 输出文件

训练完成后，会在 `figs_maze2d_qrl/` 目录下生成：

- `heatmap_step{step}.png`: 每个可视化步骤的 heatmap
- `quasimetric_net.pth`: 训练好的模型权重

## 准度量几何特性验证

代码会自动验证以下准度量性质：

1. **非负性** (Non-negativity): d(s, g) >= 0
   - 所有距离值应该非负

2. **自反性** (Reflexivity): d(s, s) = 0
   - 从状态到自身的距离应该接近 0

3. **三角不等式** (Triangle Inequality): d(s1, s3) <= d(s1, s2) + d(s2, s3)
   - 违反率应该尽可能低

## 预期结果

训练过程中，你应该看到：

1. **距离场逐渐收敛**: 
   - 初始时距离场可能比较随机
   - 随着训练，距离场应该反映出到达目标的最短路径

2. **准度量性质改善**:
   - 自反性: d(s, s) 应该接近 0
   - 三角不等式违反率应该逐渐降低

3. **损失函数收敛**:
   - `sq_dev`: 应该逐渐减小
   - `violation`: 应该接近 0（满足约束）
   - `global_loss`: 应该稳定
   - `tri_loss`: 应该逐渐减小

## 自定义迷宫

你可以通过修改 `SimpleMaze2D` 的 `_create_default_walls()` 方法或直接传入 `walls` 参数来创建自定义迷宫：

```python
# 创建自定义墙壁
custom_walls = [
    (0, 0), (0, 1), ...  # 墙壁坐标列表
]

env = SimpleMaze2D(
    maze_size=(15, 15),
    walls=custom_walls,
    start_pos=(1, 1),
    goal_pos=(13, 13)
)
```

## 与 MountainCar 实验的对比

| 特性 | MountainCar | 2D Maze |
|------|-------------|---------|
| 状态空间 | 连续 2D (位置, 速度) | 连续 2D (x, y) |
| 动作空间 | 离散 3 个 | 离散 4 个 |
| 可视化 | 1D 曲线 | 2D Heatmap + 3D 表面 |
| 几何验证 | 基础验证 | 完整验证 |

## 故障排除

1. **ImportError: No module named 'mpl_toolkits'**
   - 确保安装了 matplotlib: `pip install matplotlib`

2. **内存不足**
   - 降低 `heatmap_resolution` 参数
   - 减少 `batch_size`

3. **训练不收敛**
   - 检查 `step_cost` 是否与环境奖励匹配
   - 调整学习率 `lr`
   - 增加训练步数 `total_steps`

## 下一步

- 尝试不同的迷宫大小和结构
- 实验不同的准度量网络架构
- 添加策略可视化（显示最优路径）
- 对比不同损失权重的影响

