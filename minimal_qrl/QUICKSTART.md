# 快速开始指南

## 一条命令运行

```bash
cd /Users/z./Documents/博士/本科毕设/quasimetric-rl
python minimal_qrl/train.py
```

## 预期输出

训练开始后，你会看到：

1. **日志输出**：训练进度、loss 值等
2. **TensorBoard 日志**：保存在 `results/minimal_qrl/tensorboard/`
3. **模型检查点**：保存在 `results/minimal_qrl/checkpoint_*.pth`
4. **训练日志文件**：`results/minimal_qrl/train.log`

## 查看训练曲线

### 方法一：查看最新训练结果（推荐）

每次训练都会在 `results/minimal_qrl/tensorboard/` 下创建一个带时间戳的子目录，使用以下脚本可以自动启动最新训练的 TensorBoard：

```bash
./minimal_qrl/view_latest_tensorboard.sh
```

然后在浏览器打开 `http://localhost:6006`

### 方法二：查看所有训练结果

如果想同时查看所有训练结果进行对比：

```bash
tensorboard --logdir=results/minimal_qrl/tensorboard
```

### 方法三：查看特定训练结果

如果想查看特定时间的训练结果，可以指定具体的子目录：

```bash
tensorboard --logdir=results/minimal_qrl/tensorboard/20240101_120000
```

**注意**：每次训练都会创建独立的子目录，旧的训练结果会被保留，但默认只显示最新的一次训练结果，避免曲线混乱。

## 验证训练是否成功

训练成功的标志：
1. ✅ Loss 值逐渐下降（在 TensorBoard 中查看 `train/total_loss`）
2. ✅ 没有报错，训练完成并生成 `COMPLETE` 文件
3. ✅ 生成了多个检查点文件

## 常见问题

### Q: 导入错误 `ModuleNotFoundError: No module named 'minimal_qrl'`
A: 确保在项目根目录运行，而不是在 `minimal_qrl` 目录内

### Q: 训练很慢
A: 可以减小 `--total-steps` 或 `--batch-size` 来快速测试

### Q: 内存不足
A: 减小 `--batch-size` 或 `--num-episodes`

## 通信巡检环境成功率评估

如果你已经用 `run_comm_inspection_train.sh` 训练好了通信感知巡检 Dubins 环境，可以直接运行：

```bash
bash minimal_qrl/run_comm_inspection_execution_eval.sh
```

默认会读取：

- `results/minimal_qrl_inspection_dubins/checkpoint_final.pth`
- 固定的 `inspection_target` / `ground_station` / `obstacle_config`
- 随机起点和随机 task terminal goal
- 同时评估 `greedy` 与 `lookahead`

评估结果会保存到：

```bash
results/minimal_qrl_inspection_dubins/comm_inspection_execution_eval.json
```

重点可以查看这些字段：

- `success_rate`: 真正完成任务的 episode 比例
- `avg_steps_success`: 成功 episode 的平均步数
- `ever_task_feasible_rate`: rollout 中曾进入联合任务可行区域的比例
- `collision_rate`: episode 级别的碰撞比例
- `out_of_bounds_rate`: episode 级别的越界比例

## 下一步

- 查看 `README.md` 了解详细参数
- 修改 `simple_env.py` 创建更复杂的环境
- 调整超参数来优化训练效果
