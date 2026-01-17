# 最小可运行 QRL 核心训练版本

这是一个最小可运行版本的 QRL（Quasimetric RL）核心训练脚本，**尽量复用现有 QRL 核心实现**，不依赖 d4rl/mujoco/gym 等复杂环境。

## 特性

- ✅ **复用核心 QRL 模块**：使用 `QRLAgent`, `QRLLosses`, `QRLConf` 等核心实现
- ✅ **简单环境**：使用简单的 2D 网格环境，无需 mujoco/d4rl
- ✅ **完整训练闭环**：包含数据收集、训练、日志记录、模型保存
- ✅ **可观测结果**：输出 loss 曲线（TensorBoard）和训练统计
- ✅ **评估指标**：自动计算真实距离、Spearman/Pearson 相关系数、MSE/MAE 等
- ✅ **可视化**：距离场热力图，对比预测距离与真实距离

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
- `--eval-n-pairs`: 评估时采样的状态-目标对数（默认: 2000）

## 输出

训练完成后，输出目录包含：

- `train.log`: 训练日志
- `tensorboard/`: TensorBoard 日志（可用 `tensorboard --logdir=results/minimal_qrl/tensorboard` 查看）
- `checkpoint_*.pth`: 训练检查点
- `checkpoint_final.pth`: 最终模型
- `distance_heatmap_step*.png`: 距离场热力图（每个评估间隔生成一次）
- `COMPLETE`: 完成标记文件

### 评估指标

在 TensorBoard 的 `eval/` 标签下可以查看：
- `mse`: 均方误差
- `mae`: 平均绝对误差
- `spearman_corr`: Spearman 相关系数（衡量排序一致性）
- `pearson_corr`: Pearson 相关系数（衡量线性相关性）
- `relative_error`: 相对误差
- `distance_heatmap`: 距离场热力图可视化

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
├── __init__.py          # 包初始化
├── simple_env.py        # 简单的 2D 网格环境
├── dataset.py           # 数据集创建
├── train.py             # 主训练脚本
├── run.sh               # 运行脚本
└── README.md            # 本文件
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

## 下一步

- 可以修改 `simple_env.py` 创建更复杂的环境
- 可以启用 Actor 训练（修改 `train.py` 中的 `actor=None`）
- 可以调整 QRL 超参数（通过 `QRLConf`）

