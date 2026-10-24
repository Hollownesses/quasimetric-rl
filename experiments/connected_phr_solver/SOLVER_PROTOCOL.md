# 原数学问题下的 PHR 时钟对照：运行前冻结

基于只读诊断提交 `926a1ed`，不替换原局部约束、不删除 B1、不调整数据权重。第一项诊断只决定开展此项求解器比较，不用于生成训练标签。

## 固定项

- 数据：`results/connected_expert_utrap_20261021/main`；42536 物理记录，重复段原样保留；原 B1 配对、task mask、节点与成本保持不变。
- 共同父 checkpoint：`results/connected_expert_qrl_20261023/main/train/final.pt`；SHA256 `e40171f8c4d23f4d894dc64f37f29ca5d4f9e42a9f5846e4664d197afbe3e37d`，step50000，初始化 seed606。
- 两组都恢复 model、Adam 一二阶矩、逐记录 lambda、访问计数、sampler 及 RNG。共享 encoder+IQE，目标 `−0.6 mean(task)−0.4 mean(B1)+mean(PHR)`；约束 `d(s,s')−c<=0`；rho100；batch256，原无放回 epoch 采样。
- 各追加 5000 primal 更新。共同重新启动低步长衰减 `2e-6 * 0.1**((j-1)/4999)`，j=1..5000，保留 Adam 状态。该 A 是相同续训设置下的按访问更新基线，不是保持最后学习率常数的基线。
- 两组固定 torch CPU 单线程，便于确定性复现；父训练为4线程，此运行设置变化对两组相同。没有新 seed 搜索、QP 或参考驱动停止。

## 唯一求解策略差异

|组|primal 更新期间|dual 更新|
|---|---|---|
|A visit|沿用当前 advance|每次采样以 primal 更新前的残差 `lambda_i <- relu(lambda_i+h_i)`|
|B outer|每500步内固定 lambda|块结束后全量重算当前 h，`lambda_i <- relu(lambda_i+n_i*h_i)`，n_i为该块真实访问次数|

每个完整块中每条记录至少访问两次。两组每条记录的累计 nominal dual 步长严格相同；不是把 B 的 eta 直接设为 rho100。B 同时改变了冻结/聚合、投影时机与残差新鲜度，因此不能将差异单独归因于其中一项。它是阻尼外层 PHR 对照，不是 eta=rho 的经典精确子问题 ALM。

每500步进行同样的全量前向审计，B 在更新前保存固定 lambda 的增广目标，更新后重新计算同一模型下的目标，并作为下一块起点。由相邻块起点/终点得到固定乘子的内层目标下降量；不将跨 lambda 目标差误解为下降。最后一块在4500步（B外层更新后）及5000步（B外层更新前）额外计算全量梯度。A也在相同步数计算。5000步的最终外层更新没有后续 primal 响应，明确保留并报告更新前/后乘子状态；执行评估只由模型决定。

## 评价、监督与停止

- 固定预算停止；训练模块不加载最短路、Oracle、参考目标或执行标签。每5000步（本次0、5000）写入并检查监督记录；500步的原约束审计只用于记录与非有限值检测，不做自适应调参。
- 完成后统一评估父模型、A、B：全边可行性、互补性、全量梯度、task/B1 push、长程有符号分解、去重 source/起点误差、58个分支与307/165起点路径 regret。
- 对齐 initial model/Adam/dual/RNG/sampler，最终 sampler/visits/push counters/逐记录 dual nominal预算必须完全一致；记录训练与全量审计总时间，B额外求解成本单列。
- 本次只有一个共同父模型，是求解机制诊断，不是从头多seed复现或环境闭环导航验证。500步内层不能被预先称为充分收敛，依据固定乘子目标及梯度实测解释。
- 提交源码、测试、本协议、报告、引用图；不提交 checkpoint、日志、JSON/NPZ、源码快照。

```sh
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.connected_phr_solver.train --arm A --output results/connected_phr_solver_20261024/main/A
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.connected_phr_solver.train --arm B --output results/connected_phr_solver_20261024/main/B
```
