# QRL Loss 分析：为什么 total_loss 没有下降？

## 问题分析

从训练结果看，`total_loss` 从 15.6 上升到 22-23 后稳定，这**在 QRL 中可能是正常的**，原因如下：

## QRL Loss 的特殊性

QRL 的优化目标与传统 RL 不同，它包含**相互冲突的目标**：

### 1. Global Push Loss（全局推拉损失）
- **目标**：**最大化**状态-目标对之间的距离
- **公式**：`F.softplus(offset - dists, beta=beta).mean()`
- **行为**：当距离 < offset 时，惩罚较大；当距离 > offset 时，惩罚较小
- **效果**：鼓励模型学习**更大的距离值**

### 2. Local Constraint Loss（局部约束损失）
- **目标**：确保不**高估**观察到的局部距离/成本
- **公式**：`(sq_deviation - epsilon^2) * lagrange_mult`
- **约束**：`E[relu(d(s,s') - step_cost)^2] <= epsilon^2`
- **效果**：限制距离不能太大，必须满足局部约束

### 3. Latent Dynamics Loss（潜在动力学损失）
- **目标**：确保潜在空间动力学一致性
- **公式**：`sq_dists * weight`
- **效果**：学习状态转移

## 为什么 Loss 会上升？

1. **Global Push 的目标是最大化距离**：
   - 模型学习增大距离值 → Global Push Loss 可能上升
   - 但受 Local Constraint 限制，不能无限增大

2. **平衡状态**：
   - 当模型在 Global Push 和 Local Constraint 之间达到平衡时
   - Loss 会稳定在一个值附近，而不是单调下降

3. **这是正常的**：
   - QRL 不是简单的损失最小化问题
   - 而是**约束优化问题**（最大化距离，受局部约束限制）

## 应该关注什么指标？

### ✅ 正确的指标

1. **Local Constraint Violation**：
   - `train/critic_XX/local_constraint/violation`
   - 应该接近 0 或为负（表示约束满足）

2. **Squared Deviation**：
   - `train/critic_XX/local_constraint/sq_deviation`
   - 应该接近 `epsilon^2 = 0.25^2 = 0.0625`

3. **Distance 值**：
   - `train/critic_XX/local_constraint/dist`
   - 应该合理（对于 10x10 网格，最大距离约 18 步）

4. **Global Push Distance**：
   - `train/critic_XX/global_push/dist`
   - 应该增大，但受约束限制

### ❌ 不应该只看 total_loss

- `total_loss` 的绝对值不重要
- 重要的是各个损失组件的**平衡**和**约束满足**

## 如何验证训练是否成功？

### 方法 1：检查 TensorBoard 中的各个损失组件

```bash
tensorboard --logdir=results/minimal_qrl/tensorboard
```

查看：
- `train/critic_00/local_constraint/violation` - 应该接近 0
- `train/critic_00/local_constraint/sq_deviation` - 应该接近 0.0625
- `train/critic_00/global_push/tsfm_dist` - 应该稳定

### 方法 2：评估学习到的 Quasimetric

可以添加评估代码，检查：
1. 距离是否满足局部约束
2. 距离是否合理（例如，从起点到终点的距离应该接近真实最短路径）

### 方法 3：可视化距离场

类似 `experiments/maze2d_qrl.py` 中的可视化，检查学习到的距离是否合理。

## 改进建议

1. **调整 Global Push 参数**：
   - 对于简单环境，`softplus_offset=15` 可能太大
   - 可以尝试减小到 5-10

2. **检查数据质量**：
   - 随机收集的数据可能不够好
   - 可以增加数据收集的 episode 数量

3. **调整学习率**：
   - 默认 `lr=1e-4` 可能偏小
   - 可以尝试 `lr=1e-3`

4. **添加评估指标**：
   - 计算真实最短路径距离
   - 比较学习到的距离与真实距离

## 结论

**Loss 上升后稳定是正常的**，因为：
- QRL 是约束优化问题，不是简单的损失最小化
- Global Push 鼓励增大距离，可能导致 loss 上升
- 重要的是约束是否满足，而不是 loss 是否下降

**建议**：查看 TensorBoard 中各个损失组件的详细曲线，而不是只看 total_loss。

