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

## 下一步

- 查看 `README.md` 了解详细参数
- 修改 `simple_env.py` 创建更复杂的环境
- 调整超参数来优化训练效果

