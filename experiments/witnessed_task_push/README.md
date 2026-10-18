# 完整 replay 的 witnessed task push 单 seed 配对实验

本轮只改 task 最大化方向的支持：复用 `task_mode="witnessed"` 产生的 `global_push_task_mask`，训练 loss 必须实际消费该 mask。原始 200,000 物理边＋748 anchors、四路径 335 物理边＋4 anchors、任务条件、一步成本、终态 quotient、固定 B1 配对、共享 encoder/IQE 全部不变。原始 task 仅保留 2,575 个合法成功后缀源；四路径 335 个全部保留。

每步原始 batch 256 无放回遍历，四路径全量，两来源各占 0.5；各来源 push 是 `0.6 * mean(task[mask]) + 0.4 * mean(B1[pair_mask])`。空 task 选择返回可微零，不回退、不重新分配权重。这里是保留条目的条件均值，并非将掩码后总和除以原 batch 大小；与已有 production witnessed 语义一致。全部 200,335 有效 B1 配对及真实边局部约束继续参与。原始 quotient 恒等 anchors 不必反复优化，但每次全量扫描仍检查。

从同一 seed 606 初始模型重新训练，与旧 all 模型配对；不是从旧 final 微调。沿用 AdamW、rho=100、逐边 dual、20,000 步 2e-5 与 5,000 步 2e-6→2e-7。保留原始/路径采样权重及 dual 时钟；不加 QP。每 1,000 步扫描全部 201,087 边，并评估各原始任务固定 512 源与全部四路径源的模型距离。探针 RNG 独立，不改变训练 RNG；只读模型值，没有 Oracle 或图标签。

停止时刻及最终模型事先固定。只有 `train/COMPLETE` 写入后，分析器才使用旧结果中的记录图距离与 B1 片段成本上界。分析器校验数据、初始参数、采样器最终状态、访问次数、输入代码及旧产物哈希。输出包括 corridor 数值轨迹、全量各任务分布、四路径长程误差与余量分解、全量局部约束、B1 上界违反、空见证 batch 数、最坏 50 边。保存全部结果；不因早期曲线或参考误差改变训练预算。

此轮优先回答用户指定的三个训练诊断问题，不新增闭环或候选评分，也不由记录图恢复推断导航改善。所有既有 MPPI 与失败结果保留；环境成本、成功定义均未改。

复现（仓库根目录，依赖两个既有 replay 缓存及旧完整实验）：

```sh
export MPLCONFIGDIR=/tmp/qrl-execution-mpl
PY=../quasimetric-rl/.venv/bin/python
RUN=results/witnessed_task_push_reproduction
$PY -m pytest -q experiments/witnessed_task_push/test_witnessed.py experiments/full_replay_four_path/test_full_replay.py minimal_qrl/test_global_push_sampling.py
$PY -m experiments.witnessed_task_push.prepare --root "$RUN"
$PY -m experiments.full_replay_four_path.train --output "$RUN/train" --task-mode witnessed --seed 606 --phr-steps 20000 --decay-steps 5000 --batch 256 --threads 4 --audit-interval 1000
$PY -m experiments.witnessed_task_push.analyze --root "$RUN"
```

恢复时重复训练命令，附加 `--resume "$RUN/train/latest.pt"`，保持其余参数相同。checkpoint 原子保存模型、优化器、逐边 dual、访问次数、采样顺序/游标/RNG、task mode/mask 指纹与 push 计数；禁止 all/witnessed 间恢复。恢复日志保留全部已写记录，可能出现同一步的重放段；分析和绘图不应将其视为独立样本。

本次实际结果：`results/witnessed_task_push_20261019/main`；旧配对模型：`results/full_replay_four_path_20261017/main`。先用 seed 9003、100＋10 步做独立 smoke，结果也保留。`--task-mode` 默认仍是 `all`，旧命令语义兼容；回归测试将 all 模式单步参数和 dual 与旧源码快照逐位比较。空 task、物理/anchor mask 对齐、所有 B1 梯度及优化器恢复都有定向验证。

数值复现范围：与旧实验相同的 4 CPU 线程下，同一初始状态、同一 batch 的两份模型也会在更新后出现微小数值差异；5 步诊断的最大参数差约 2.16e-5。同样诊断改为单线程时逐位相同，优化器恢复单元测试也使用单线程。端到端 110 步恢复 smoke 的采样与访问计数一致，但模型不保证逐位复现，原结果和差异均保留在 `smoke_resume`、`resume_numeric_diagnostic.json`。正式实验保持旧 4 线程配置且未恢复，不为追求数值一致改变模型或训练机制。相同 seed/初始化和采样不消除这一数值噪声，因果结论仍限于单次配对运行。
