# M2：原始100k模型的完整官方value梯度核验

2026-10-29，在结果前冻结。补齐M1仅测value_loss的解释缺口，直接检查官方dynamics_loss对value编码器的即时作用；不重新做数据覆盖调查，不修改E1/E2/MQE或其他正在运行的预算。

- 严格加载原始100k checkpoint及Adam状态。每个反事实从同一原始状态出发，不串联更新，不保存新模型。
- 使用与M1相同seed4400、32个官方batch1024、相同采样调用顺序与固定probe（E0路线节点0/15、同task3目标）。核对M1已记录的批次流hash与value-only反事实，避免换样本后比较。
- 分别计算官方value_loss、dynamics_loss、actor_loss及total_loss对modules_value的梯度；断言分量之和等于真实total梯度，实测actor对value的梯度是否为零，不根据代码表面猜测。
- FP32核验采用逐元素rtol2e-5、atol5e-5，并记录实际最大绝对误差。首次只用max绝对误差<5e-5，在第二个批次（batch=1）5.3406e-5处中止；失败目录full_gradient保留。修复仅为考虑梯度尺度的数值一致性检查，新目录full_gradient_fp32重复原32个批次，不改损失或样本。不得以此调整上坡方向或显著性判据。运行后仅对失败批次的文字记录作勘误（最初误写第一批），原运行版本在source_snapshot内按哈希保留。
- 测量原尺度gap与相对gap的一阶SGD投影、梯度范数及value/dynamics夹角。相对gap定义和M1相同，用于排除统一正缩放的直接影响；局部两个probe不能代表全图。
- 同批次分别执行M1的官方value-only一次Adam和原生agent.update的完整联合一次Adam，比较实际probe变化及一阶预测。完整更新允许actor/dynamics一起变化，只读取更新后的value场；不把它称为冻结actor的训练实验。
- 测量零新梯度但继承Adam矩的value运动，作为动量参照。不得把Adam反事实差简单当作可加的梯度贡献，不将一次更新线性外推到10k/1M。
- 报告所有32个批次的分布、正负比例、均值和配对差，不按结果增加批次或换seed。不根据结果调整正在运行的训练。

本项只能完善原始模型附近的梯度机制证据；它不证明整个训练历史的唯一根因，不代替固定终点的通道/闭环执行检验。原始JSON与日志保留本地，源码、测试、协议、报告及引用PNG单独提交。

复现命令（先按README完成原始模型场复核与M1，所引用的原始checkpoint和M1逐批analysis必须存在；输出目录使用新名称）：

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
R=results/ogbench_critic_20261029
$P "$E/test_full_gradient.py"
$P "$E/full_gradient.py" "$R" --output "$R/full_gradient_reproduction"
$P "$E/report_full_gradient.py" "$R/full_gradient_reproduction"
```
