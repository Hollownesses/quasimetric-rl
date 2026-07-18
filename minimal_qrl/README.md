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

## 工业园区米制扩展实验

空间/设备扩展实验使用 `1 环境单位 = 10 m`：`10×10` 即
`100 m×100 m = 10,000 m²`。默认生成6个场景、18个训练 job，并在
20k/40k/60k 检查点评估。

```bash
# 只生成场景、固定任务库和 manifest
python -m minimal_qrl.industry_exp.scalability generate \
  --output-root results/qrl_scalability_metric

# 运行全部训练与评估（必须固定同一计算设备）
python -m minimal_qrl.industry_exp.scalability run \
  --output-root results/qrl_scalability_metric \
  --device cpu \
  --seeds 0,1,2

# 从已有结果重新生成 CSV、JSON、图表和 Markdown 报告
python -m minimal_qrl.industry_exp.scalability report \
  --output-root results/qrl_scalability_metric
```

运行器会跳过已有 `COMPLETE` 的 job，并从未完成训练目录中最新的
checkpoint 继续。汇总报告位于 `results/qrl_scalability_metric/report/`。
训练默认每2,000 step保存一次模型，但仍只在由 `--checkpoints`
指定的节点运行验证；保存间隔可用 `--save-interval` 单独调整。

可以用 `--area-sides` 和 `--device-counts` 只选择部分场景。面积边长目前支持
`100,200,300,500,1000` 米。例如，单种子趋势实验：

```bash
bash minimal_qrl/run_qrl_scalability_pilot.sh
```

该脚本选择 `100/300/1000 m` 和 `K=4/12/24`，公共基准不重复，
因此每个种子共训练5个模型。单种子报告仅用于趋势观察，
`usable` 的正式判定仍保持三种子要求。

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

## Goal-set 通信巡检 Baseline

通信巡检环境由 `--device-catalog` 指定工业设备 JSON 目录。任务目标仅从目录设备产生；Global Push 的 state–state 项则使用同一设备上下文下独立采样的全自由空间状态对。

统一 benchmark 包含 Hybrid A*、no-terminal MPPI、model-only MPPI、goal-set SAC、QRL，以及三个目标上下文 GCRL 基线：

- `context_her_ddpg`：设备任务上下文重标记的 HER+DDPG；
- `context_contrastive_rl`：以抽象设备任务为目标表示的 Contrastive RL；
- `mrn_context_her_ddpg`：和 Context HER-DDPG 共享训练管线、使用 MRN critic。

三者都同时输出原生 actor 和统一价值校准 MPPI 结果。Context HER 只会重标记到未来状态真实满足的目录设备任务，并在该上下文中重算观察、稠密任务代价和终止标记。

```bash
# 快速闭环检查
STAGE=smoke bash minimal_qrl/run_comm_inspection_baselines.sh

# 单 seed pilot；默认训练 SAC 和三个 Context GCRL 方法 300k 环境步
STAGE=pilot bash minimal_qrl/run_comm_inspection_baselines.sh

# 正式实验；QRL_CHECKPOINTS 用空格分隔多个训练 seed 的 checkpoint
STAGE=final \
QRL_CHECKPOINTS="path/to/qrl_seed0.pth path/to/qrl_seed1.pth path/to/qrl_seed2.pth path/to/qrl_seed3.pth path/to/qrl_seed4.pth" \
bash minimal_qrl/run_comm_inspection_baselines.sh
```

可以通过 `TRAIN_CONTEXT_AGENTS=0` 禁用现场训练，并用空格分隔的 `CONTEXT_CHECKPOINTS` 加载已有 checkpoint。`CONTEXT_TOTAL_ENV_STEPS`、`CONTEXT_SEEDS`、`CONTEXT_HER_K` 和 `CONTEXT_TEACHER_RATIO` 分别控制训练预算、seed、future relabel 比例和 teacher episode 比例。每个训练目录包含 `train_metrics.csv`、`validation_metrics.csv`、TensorBoard 日志及每 50k 步 checkpoint。

结果写入 `baseline_results.json` 和 `baseline_results.csv`，包含逐 episode 指标、训练/teacher 步数、HER 有效率、参数量、bootstrap 95% 区间，以及原生策略对 QRL greedy、统一 MPPI 对 QRL-MPPI、MRN 对普通 DDPG critic 的配对比较。所有 MPPI 变体使用相同 rollout 预算；Context GCRL 价值在训练分布的固定校准集上按 5%/95% 分位映射到 `[0,1]`。
