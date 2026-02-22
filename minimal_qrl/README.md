# 最小可运行 QRL 核心训练版本

这是一个最小可运行版本的 QRL（Quasimetric RL）核心训练脚本，**尽量复用现有 QRL 核心实现**，不依赖 d4rl/mujoco/gym 等复杂环境。

## 特性

- ✅ **复用核心 QRL 模块**：使用 `QRLAgent`, `QRLLosses`, `QRLConf` 等核心实现
- ✅ **简单环境**：使用简单的 2D 网格环境，无需 mujoco/d4rl
- ✅ **完整训练闭环**：包含数据收集、训练、日志记录、模型保存
- ✅ **可观测结果**：输出 loss 曲线（TensorBoard）和训练统计
- ✅ **评估指标**：自动计算真实距离、Spearman/Pearson 相关系数、MSE/MAE 等
- ✅ **可视化**：距离场热力图，对比预测距离与真实距离
- ✅ **Planning / Reachability 评估**：专门面向 obstacle navigation 的评估功能
  - Greedy Navigation Success Rate：使用 QRL distance 进行 greedy navigation
  - Path Efficiency：计算实际路径与最短路径的竞争比
  - Failure Mode 可视化：分析 QRL 在 obstacle 场景下的典型失效模式

## 快速开始

### 一条命令运行

```bash
python minimal_qrl/train.py
```

或使用脚本：

```bash
chmod +x minimal_qrl/run.sh
./minimal_qrl/run.sh
```

### 自定义参数

```bash
python minimal_qrl/train.py \
    --seed 42 \
    --device cpu \
    --output-dir ./results/minimal_qrl \
    --grid-size 10 10 \
    --num-episodes 100 \
    --max-steps-per-episode 200 \
    --batch-size 256 \
    --total-steps 10000 \
    --num-critics 2 \
    --log-interval 100 \
    --save-interval 1000
```

## 参数说明

### 训练参数
- `--seed`: 随机种子（默认: 42）
- `--device`: 设备，cpu 或 cuda（默认: cpu）
- `--output-dir`: 输出目录（默认: ./results/minimal_qrl）

### 环境参数
- `--grid-size`: 网格大小，格式为 height width（默认: 10 10）
- `--num-episodes`: 数据集中的 episode 数量（默认: 100）
- `--max-steps-per-episode`: 每个 episode 的最大步数（默认: 200）

### 训练参数
- `--batch-size`: 批次大小（默认: 256）
- `--total-steps`: 总训练步数（默认: 10000）
- `--num-critics`: Critic 数量（默认: 2）

### 日志和保存
- `--log-interval`: 日志记录间隔（默认: 100）
- `--save-interval`: 模型保存间隔（默认: 1000）

### 评估参数
- `--eval-interval`: 评估间隔（默认: 1000）
- `--eval-n-pairs`: 评估时采样的状态-目标对数（默认: 500）
- `--visualization-interval`: 可视化间隔（默认: 1000，设为 0 禁用）

### Planning / Reachability 评估参数（仅 obstacle 环境）
- `--planning-eval-interval`: Planning 评估间隔（默认: 1000，设为 0 禁用）
- `--planning-eval-n-trials`: Planning 评估时的测试次数（默认: 100）
- `--planning-num-action-candidates`: 每步候选动作数量（默认: 32）
- `--planning-execution-modes`: 执行机制对比（逗号分隔），例如 `greedy` 或 `greedy,lookahead`（默认: greedy）
- `--lookahead-horizon`: lookahead 规划步长（默认: 5，仅 lookahead 模式）
- `--lookahead-num-sequences`: lookahead 序列数量（默认: 128，仅 lookahead 模式）
- `--lookahead-step-cost-weight`: 步长惩罚权重（默认: 0，仅 lookahead 模式）
- `--lookahead-collision-penalty`: 碰撞惩罚（默认: 0，仅 lookahead 模式）
- `--planning-visualize-failures`: 是否可视化失败案例（需要配合 `--planning-visualize-interval` 使用）
- `--planning-visualize-interval`: Failure mode 可视化间隔（默认: 2000）

## 输出

训练完成后，输出目录包含：

- `train.log`: 训练日志
- `tensorboard/`: TensorBoard 日志（可用 `tensorboard --logdir=results/minimal_qrl/tensorboard` 查看）
- `checkpoint_*.pth`: 训练检查点
- `checkpoint_final.pth`: 最终模型
- `distance_heatmap/`: 距离场热力图目录，内含 `distance_heatmap_step*.png`（每个评估间隔生成一次）
- `failure_mode/`: Failure 可视化目录，内含 `failure_mode_{mode}_*_step*.png`（若启用；mode=greedy/lookahead）及 `failure_start_distribution_step*.png`（当同时评估 greedy 与 lookahead 且启用可视化时）
- `COMPLETE`: 完成标记文件

### 评估指标

在 TensorBoard 的 `eval/` 标签下可以查看：
- `mse`: 均方误差
- `mae`: 平均绝对误差
- `spearman_corr`: Spearman 相关系数（衡量排序一致性）
- `pearson_corr`: Pearson 相关系数（衡量线性相关性）
- `relative_error`: 相对误差
- `distance_heatmap`: 距离场热力图可视化

在 TensorBoard 的 `planning/` 标签下可以查看（仅 obstacle 环境）：
- **旧标签（兼容）**：
  - `success_rate`, `avg_steps`, `avg_path_length`, `avg_efficiency_ratio`, `median_efficiency_ratio`（默认对应 greedy）
- **新标签（按执行机制分组，推荐用于对比）**：
  - `planning/greedy/*`
  - `planning/lookahead/*`

## 查看训练结果

### TensorBoard

```bash
tensorboard --logdir=results/minimal_qrl/tensorboard
```

然后在浏览器中打开 `http://localhost:6006` 查看 loss 曲线。

### 日志文件

```bash
tail -f results/minimal_qrl/train.log
```

## 代码结构

```
minimal_qrl/
├── __init__.py              # 包初始化
├── envs/                    # 环境模块
│   ├── base.py             # 环境基类
│   ├── simple_grid_2d.py   # 简单的 2D 网格环境
│   └── continuous_obstacle_2d.py  # 2D 连续障碍物环境
├── dataset.py              # 数据集创建
├── train.py                # 主训练脚本
├── eval/                   # 评估脚本/模块（集中管理）
│   ├── __init__.py         # 评估 API 导出（供 train.py 调用）
│   ├── evaluation.py       # 基础评估模块
│   ├── planning_evaluation.py  # Planning / Reachability 评估模块
│   └── qualitative_multigoal_eval.py  # 定性评估可视化脚本（展示用）
│   └── execution_mode_eval.py  # 执行机制对比评估（greedy vs lookahead）
├── run.sh                  # 运行脚本
└── README.md               # 本文件
```

## 核心实现复用

本实现完全复用现有 QRL 核心模块：

- `quasimetric_rl.modules.QRLConf`: QRL 配置
- `quasimetric_rl.modules.QRLAgent`: QRL Agent（包含 Critic）
- `quasimetric_rl.modules.QRLLosses`: QRL 损失函数（包含 Local Constraint 和 Global Push）
- `quasimetric_rl.data.Dataset`: 数据集接口
- `quasimetric_rl.data.register_offline_env`: 环境注册

## 与完整版本的区别

- **环境**：使用简单的 2D 网格环境，而不是 d4rl/mujoco 环境
- **Actor**：不训练 Actor（`actor=None`），只训练 Critic
- **配置**：使用命令行参数，而不是 Hydra 配置系统
- **依赖**：不需要 d4rl、mujoco 等复杂依赖

## 故障排除

### 导入错误

如果遇到导入错误，确保在项目根目录运行：

```bash
cd /path/to/quasimetric-rl
python minimal_qrl/train.py
```

### 设备错误

如果使用 CPU，确保 `--device cpu`。如果使用 GPU，确保安装了 PyTorch 的 CUDA 版本。

### 内存不足

如果内存不足，可以：
- 减小 `--batch-size`
- 减小 `--num-episodes`
- 减小 `--grid-size`

## Planning / Reachability 评估

对于 `obstacle` 环境，系统会自动进行 Planning / Reachability 评估（如果启用），包括：

### 1. Greedy Navigation Success Rate

使用 QRL 学到的 quasimetric 作为距离函数，在给定 start 和 goal 的情况下，每一步选择能最小化下一状态到 goal 的 QRL distance 的动作，进行 greedy navigation。统计：
- 成功率（Success Rate）
- 平均步数（仅成功案例）
- 平均路径长度（仅成功案例）

### 2. Path Efficiency

对成功到达目标的 rollout，计算实际轨迹长度与 `compute_shortest_path_distance` 得到的最短路径长度之比，统计：
- 平均效率比
- 中位数效率比
- 最小/最大效率比

### 3. Failure Mode 可视化

自动收集 greedy rollout 失败的起点，对这些 failure case 可视化：
- QRL learned distance heatmap
- shortest-path distance heatmap
- 实际 rollout 轨迹（叠加在环境上）

用于分析 QRL 在 obstacle 场景下的典型失效模式。

### 使用示例

```bash
# 启用 Planning 评估（默认已启用，每 1000 步评估一次）
python minimal_qrl/train.py \
    --env-type obstacle \
    --planning-eval-interval 1000 \
    --planning-eval-n-trials 100 \
    --planning-visualize-failures \
    --planning-visualize-interval 2000
```

评估结果会：
- 记录到 TensorBoard（`planning/success_rate`, `planning/avg_efficiency_ratio` 等）
- 打印到日志
- 失败案例可视化保存到输出目录下的 `failure_mode/` 子目录（`failure_mode_*_step*.png`）

## 下一步

- 可以修改 `simple_env.py` 创建更复杂的环境
- 可以启用 Actor 训练（修改 `train.py` 中的 `actor=None`）
- 可以调整 QRL 超参数（通过 `QRLConf`）
- 可以使用 Planning / Reachability 评估分析 QRL 在 obstacle navigation 中的表现

