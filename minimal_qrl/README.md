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

### 2×2密度保持拼接园区

更真实的联合规模实验将4个 `100 m×100 m` 子园区平移拼接为
`200 m×200 m` 园区，共96台设备和12个障碍物。设备、障碍物半径和
`3.5–8.5 m` 巡检距离均保持原物理尺寸；完整园区使用一个中心基站，
通信偏置按边长调整。验证和测试起点直接从完整200米园区独立随机采样。

```bash
bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
```

脚本默认只运行2×2单种子场景。固定数据密度以24设备场景的50k真实转移
和50k更新为基准，因此96设备场景自动使用200k真实转移和200k更新，
并在40k/80k/120k/160k/200k进行验证。若要重新训练具有相同中心基站
口径的1×1基准，可运行：

```bash
TILE_GRIDS=1,2 \
OUTPUT_ROOT=results/qrl_tiled_scalability_1x1_2x2 \
bash minimal_qrl/run_qrl_tiled_scalability_2x2.sh
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

## Goal-set 通信巡检 Baseline

### 可行方向 QP 消融

使用诊断脚本的独立训练阶段：

```bash
PHASE=train_qrl_feasible_direction DEVICE=cpu \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

该阶段沿用 `train_qrl_kkt_functional` 的诊断场景与 replay 协议，
默认将训练结果写入 `results/diagnostic_u_shadow_corridors/qrl_training_feasible_direction`。
可以用 `OUTPUT_ROOT` 或 `FEASIBLE_DIRECTION_TRAIN_DIR` 指定独立输出目录。
该阶段默认只跑 `TOTAL_STEPS=2000` 的 smoke；需要更长实验时显式设置该变量。
默认每个 critic 从 replay 取 32 条一步成本边组成工作集；可用
`QRL_FEASIBLE_WORK_EDGES`、`QRL_FEASIBLE_KAPPA`、
`QRL_FEASIBLE_STEP_SIZE`、`QRL_FEASIBLE_RESTORATION_STEP_SIZE`、
`QRL_FEASIBLE_TRUST_RADIUS` 和 `QRL_FEASIBLE_AUDIT_INTERVAL` 环境变量调整。
恢复阶段默认尝试系数 1.0，由参数空间 trust radius 和真实残差回溯限幅；
可行阶段的 Global Push 方向默认尝试系数 0.01。二者分开是因为不可行初始化
需要快速消除约束违反，而普通学习率不足以完成恢复。恢复阶段的 QP 仍使用
Global Push 梯度，但只接受工作集真实违反下降的有限更新；这阶段不保证目标单调。
通信巡检任务中，编码器将
`task_feasible=1` 的终态与同一任务的抽象目标映射到同一个规范输入，因而零代价
terminal-to-goal 距离按构造等于零。IQE 仍在映射后的输入上定义准距离；所有零代价边
仍留在 replay 约束和审计中，不按成本归一化。此映射依赖通信巡检环境的 42 维
任务观测布局，checkpoint 保存开关以供评估时使用。

此模式的完整 critic 更新是 Global Push 梯度的 QP 方向，并直接写入参数；
不执行 AdamW、weight decay 或 functional dual 更新。为保持目标和约束定义清楚，
训练配置关闭 latent dynamics、temporal、goal-return 和 abstract-edge 的辅助损失；
抽象目标边仍在 replay 约束中。该模式拒绝非零 n-step/MQE 权重。
`train_metrics.csv` 记录工作集真实残差、QP elastic slack、恢复阶段、方向范数、
回溯拒绝率、边访问数和 wall time；全 replay 审计单列普通边与零代价边违反率，
并在训练结束时
追加 `final_replay_audit` 行。工作集之外的边只有审计时才检查，离散更新不具有
连续时间可行性证明的保证。比较实验应同时对齐 replay、Global Push 抽样支持、
访问边数和总 wall time。

通信巡检环境由 `--device-catalog` 指定工业设备 JSON 目录。任务目标仅从目录设备产生；Global Push 的 state–state 项则使用同一设备上下文下独立采样的全自由空间状态对。

### Global Push 支持对照（主线 B）

以下四个 `PHASE` 复用相同的 qrl_explore 数据、线性 push、task/state=.6/.4，
默认采用 `kkt_functional` 单步约束配置；关闭 temporal、goal-return、n-step 和 MQE
路径辅助 loss。可统一设置 `QRL_LOCAL_CONSTRAINT_MODE=goal_set_barrier` 等切换
优化器，四组应使用相同值。其他训练参数、seed、budget、成功转移权重也应对齐。

```bash
for phase in train_qrl_push_support_control train_qrl_push_support_b0 \
             train_qrl_push_support_b1 train_qrl_push_support_witnessed; do
  PHASE="$phase" DEVICE=mps SEED=42 \
    OUTPUT_ROOT=results/diagnostic_push_support \
    bash minimal_qrl/run_comm_inspection_diagnostic.sh
done
```

各组写入独立的 `qrl_training_push_support_{control,b0,b1,witnessed}` 目录。
`PUSH_SUPPORT_TRAIN_DIR` 可覆盖单次实验目录。

- control：保留环境合法状态的独立随机配对。
- B0：两端独立抽自相同 `context_id` 和相同 task goal 的物理 replay 状态池，
  包含末端状态、排除抽象伪边；允许跨 episode、逆时间与相同状态配对，不能保证有路径。
- B1：每个物理转移源状态与同一连续片段的未来状态配对；均匀抽取
  `k ∈ [1, 剩余片段步数]`，包含失败和成功轨迹，不跨终止、timeout、context 变化或伪边。
  task 分支保持原样，因此本组只检验 state-state 支持错位。
- witnessed：在 B1 基础上，只保留有完整成功后缀且确实连接到 replay 中同 context
  合法零代价 anchor 的 task 源状态。预算截断的成功轨迹、失败轨迹和独立抽取的
  anchor 不算成功见证。没有见证时 task push 为空并明确记录，不用 teacher 补齐。

所有模式沿用每个 replay row 一个固定 pair 的 bank，使用独立 seed；配对改变不修改
replay 转移或其采样权重。片段只决定 push 状态对，不使用累计成本 target、最优轨迹
或最短路。`--global-push-state-sampling environment|replay|trajectory` 和
`--global-push-task-sampling all|witnessed` 也可直接组合使用。默认值保留原训练行为。

每组输出 `global_push_sampling_stats.json`（context/device/heading 覆盖、成功/失败
pair 数、时间跨度直方图及分位数、位置/heading 差分位数、task 保留比例及成功加权后的
采样质量比例）和 `global_push_pair_coverage.csv`（逐位置/heading/context 的 replay、
pair 两端与 task 计数，包含零信号网格）。诊断区域计数可检查 U 内信号是否丢失。
报告中的 `physical_replay_sha256` 可确认四组使用相同 replay 转移与任务上下文。
`GLOBAL_PUSH_COVERAGE_POSITION_RESOLUTION` / `GLOBAL_PUSH_COVERAGE_HEADING_BINS`
默认 1.0 / 12。覆盖率以物理 replay 网格为参照，不表示未访问环境状态也获得了约束。
B0 与 control 没有路径跨度，报告空间和 heading 差，时间跨度明确为空。

见证路径只有在对应单步约束及 anchor 被满足时才提供距离上界；此实验没有让软约束
优化器变成硬可行求解器，也不保证有限网络的未覆盖状态泛化。若尺度改善伴随关键区域
失去信号，应结合覆盖报告判断；覆盖基本保持而 topology 无改善则不支持支持错位是主因。

### Goal-set quotient + replay margin barrier

可行方向 QP 在严格可行边界附近可能长期拒绝有限参数更新。替代诊断阶段使用同一个
终态规范映射，并在高残差和近边界 replay 边上优化平滑 margin 惩罚：

```bash
PHASE=train_qrl_goal_set_barrier DEVICE=cpu \
  OUTPUT_ROOT=results/diagnostic_goal_set_barrier \
  bash minimal_qrl/run_comm_inspection_diagnostic.sh
```

默认 `TOTAL_STEPS=2000`，每 critic 工作集 64 条边，其中最多一半来自最近一次审计的
高违反边；每 25 step 审计全 replay。可用 `QRL_BARRIER_WORK_EDGES` 和
`QRL_BARRIER_AUDIT_INTERVAL` 调整覆盖与计算量。
`QRL_BARRIER_WEIGHT`、`QRL_BARRIER_MARGIN`、`QRL_BARRIER_TEMPERATURE` 和
`QRL_BARRIER_TOPK` 控制近边界惩罚；默认权重为 500，学习率为 2e-5，
可通过 `QRL_BARRIER_LR` 调整。此模式使用 AdamW 优化
Global Push 与惩罚之和；
审计单列普通边和零成本 anchor 的违反数量、比例及最大残差。它是软约束实验，
不提供硬可行保证。应根据完整 smoke 的约束尾部、价值尺度与执行成功决定是否继续。

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
