# Connected expert 原 QRL 机制诊断与 PHR 求解对照

## 第一项：只读 checkpoint 诊断（运行前协议）

使用 `connected_expert_qrl_20261023/main/train/{phr,final}.pt` 的原 task+B1 45k/50k checkpoint，保持共享 encoder + IQE、0.6 task + 0.4 B1 push、原局部约束 `d(s,s') <= c` 不变。数据为 `connected_expert_utrap_20261021/main`，参考为已修正并通过目标集合与 Bellman 最小值检查的 `connected_reference_fix_20261024/main`。参考只用于离线诊断。

沿记录图的确定性参考最短路径，精确分解

`D*(s,G)-d(s,G) = sum[c(e)-d(e)] + [sum d(e)-d(s,G)]`。

第一项保留符号，负数意味着路径上的累计违反，不能截成零。第二项是沿路径累积的三角松弛。输出记录权重、去重 source、全部/去重起点与长程四分位统计。此分解不证明哪项是训练失败的因果原因，也不代表环境连续空间距离。

对全部物理记录按原均匀记录权重精确累计梯度，分别计算 task、B1、PHR、普通 Lagrangian、总增广目标梯度及夹角；同时报告可行性、`abs(lambda*h)`、负松弛正乘子数量和投影 dual 残差。PHR 使用 rho=100，保存乘子固定，float32 前向/导数、float64 向量累加。梯度范数依赖参数坐标，不能单凭绝对值宣布达到/未达到普适 KKT 阈值。均值目标约定下，若约束写成未平均的求和，实际乘子为保存 lambda/N。

测试分解的正负符号、错误参考拒绝、全量/分块梯度一致和模型/乘子只读性。checkpoint 文件前后 SHA256 必须一致。

```sh
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m pytest experiments/connected_phr_solver/test_solver.py -q
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.connected_phr_solver.diagnostic --data results/connected_expert_utrap_20261021/main --reference results/connected_reference_fix_20261024/main --run results/connected_expert_qrl_20261023/main/train --output results/connected_phr_diagnostic_20261024/main
```

只提交源码、测试、协议、报告和报告引用图；JSON/NPZ、checkpoint、日志不提交。
