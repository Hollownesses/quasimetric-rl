# 大 replay PHR 校准：同一 witnessed final 的 A/B 5k 续训

共同父 checkpoint 固定为 `results/witnessed_task_push_20261019/main/train/final.pt`，SHA256 `5d2094784b6ea07ee115970e20a5c950db7fb907ffa4a55b6ced7731809cc7c1`，原训练 step 25,000、初始化 seed 606。两组都完整继承模型、AdamW 动量与步数、逐边 dual、访问次数、采样顺序/游标/RNG 和 witnessed push 计数；不重置优化器或乘子。

- A：直接调用已有 `advance`。原始边按采样访问执行 `lambda = relu(lambda + h_pre_primal)`，步长 1；四路径仍每步相同更新。
- B：primal loss、采样、梯度更新与 A 相同，取消原始边按访问更新。每 250 次 primal 更新后，用当前模型分块扫描全量边；原始 200,000 条物理边同步执行 `lambda = relu(lambda + 100*h_latest)`。期间原始 dual 固定；四路径 dual 继续每步按更新前残差、步长 1 更新。

外层步长选 100，与现有 PHR 的 rho=100 相等，采用标准 PHR 外层上升系数；周期 250 给 primal 留出适应时间，在固定 5k 预算内产生 20 次全量更新。没有第 0 步外层更新。第 20 次发生在第 5,000 步末，保存供恢复使用，对本轮已完成的模型没有反馈；本轮后续 primal 实际受前 19 次外层更新影响。这不是等 dual 预算或仅刷新频率对照：B 每条原始边名义累计步长 2,000（影响本轮 primal 的为 1,900），A 约 6.4；投影与残差变化使其不能直接换算成实际乘子比例。

共同训练配置保持 witnessed task 条件均值与空 batch 处理、全部有效 B1、全部真实边约束、两个来源各 0.5 权重、task/B1 为 0.6/0.4、原始 batch 256 无放回、每步全四路径、共享 encoder/IQE、AdamW weight_decay=0、4 CPU 线程。学习率共同重启此前使用的低步长阶段：第 1 步 2e-6，指数下降到第 5,000 步 2e-7；父模型末步为 2e-7，但两组均做同样重启，并保留 Adam 状态。A 提供额外预算及共同学习率重启的控制。

两组每 250 步都扫描全部 201,087 条边，避免只有 B 得到额外诊断。原始 748 个 quotient anchor 恒为零，扫描保留、无需优化；四路径 4 个 anchor 继续沿用旧实现。B 扫描后才更新 dual，日志中 residual 属于当前模型，dual/complementarity 属于更新后的乘子。保存原子 latest、初始、250 步、2,500 步及最终 checkpoint，恢复时校验 arm/config，并避免重做 checkpoint 已包含的外层更新。

固定各 5k primal 更新，不按图或 Oracle 指标选择停止时刻。两组 COMPLETE 后才比较全量残差、入终态边、各任务、四路径图距离/起点/余量分解及记录 B1 上界。所有失败保留。原始数据、环境成本、任务成功定义、task push、模型架构均不改，无 QP、新数据或闭环评估。沿用 4 线程的微小数值非确定性，单个父 checkpoint 的对照不代表跨 seed 结论。

复现（仓库根目录，Python `../quasimetric-rl/.venv/bin/python`）：

```sh
export MPLCONFIGDIR=/tmp/qrl-execution-mpl
PY=../quasimetric-rl/.venv/bin/python
RUN=results/large_phr_calibration_reproduction/main
$PY -m pytest -q experiments/large_phr_calibration/test_calibration.py
$PY -m experiments.large_phr_calibration.prepare --root "$RUN"
$PY -m experiments.large_phr_calibration.train --output "$RUN/A" --arm A
$PY -m experiments.large_phr_calibration.train --output "$RUN/B" --arm B
$PY -m experiments.large_phr_calibration.analyze --root "$RUN"
```

两条正式训练命令也可独立并行，各用 4 CPU 线程。恢复某组时保持相同命令并加 `--resume "$RUN/A/latest.pt"`（B 同理）。恢复日志保留重复片段，分析按 continuation_step 的最后一条全量审计读取。

本次结果目录为 `results/large_phr_calibration_20261019`。正式训练前，单元验证原始/四路径 dual 分离、A/B 同状态 primal 一致、全量最新残差投影和优化器恢复；B 使用 260 步 smoke 跨过一个真实 250 步边界，再从该边界恢复 10 步，确认原始 dual 不按访问变化且不重复外层更新。smoke 和恢复记录均保留，不作为新增正式组或模型选择依据。
