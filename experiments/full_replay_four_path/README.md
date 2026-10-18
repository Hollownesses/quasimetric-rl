# 原始 20 万物理转移 + 四条成功路径，单 seed 实验

后续仅限制 task push 支持的配对实验见 [witnessed_task_push](../witnessed_task_push/README.md)。训练器新增 `--task-mode witnessed` 并实际消费 task mask；默认 `all` 保持本文原实验语义。旧产物不修改。

本实验只训练合并数据一组，seed 606。保留所有旧实验产物；不新增轨迹，不训练原始 replay 单独对照。原始 200,000 条物理转移及 748 条合法 anchor，与四路径 335 条物理边及 4 条 anchor 合并。context 编号偏移以避免冲突，观测的真实任务字段、奖励与终态映射保持不变；两个来源各自按历史 seed 42 重建固定 B1 bank，随后偏移索引，四路径配对不改变。

训练共享 production encoder/projector/IQE，从独立初始化开始。每步使用一批 256 原始物理边与全部 339 四路径边；两部分的 push/PHR 各占 50%。push 内部保持 0.6 task(all) + 0.4 B1。原始部分无放回遍历，轮末采用较小 batch，避免跨轮 batch 中出现重复行。rho=100，逐记录行独立 multiplier，访问时以更新前残差执行 lambda <- relu(lambda+h)。原始 anchor quotient 后为恒等边，其约束恒为零，全量审计中保留。

固定 20,000 次 lr=2e-5，再 5,000 次指数下降 2e-6 到 2e-7，所有模型参数使用同一 AdamW 参数组，无 weight decay。每 1,000 步分块扫描全部 201,087 条约束。没有全量 QP；不因图/Oracle 指标改变停止时刻或挑选 checkpoint。原始 edge 的 dual 访问频次低于路径 edge，报告实际访问次数；这种随机约束优化不等同于小图全量逐步 PHR，不能隔离“纯数据量”的因果影响。

四路径物理边在均匀合并采样中仅占约 0.167%，故使用明确的 50% 分层权重并每步保留全部路径。原始 replay 包含不同任务，保留所有任务条件和全部数据，不能将它们错误改写为 U-trap 任务。未见证任务 push 保持 all 策略，可能在记录图不连通的分量上存在无有限图上界的方向；这是需在评估中诊断的限制。

冻结 H10/N128 MPPI、环境成本、成功定义。原有 32 状态×2 RNG 的七个基线复用原 Oracle 配对实验；新模型在相同实例上执行。新增 24 个 U 内独立状态、8 个 heading 层、每状态 2 RNG，所有八组重新执行。新状态使用独立随机种子，不按模型或 Oracle 分数筛选；“独立”不意味着远离原始 replay，另报告支持距离。两个集合不合并成一个泛化指标。

所有 Oracle、记录图参考、固定候选与闭环评分均在固定最终模型保存之后使用。记录图 infinity 是没有记录链，不能当作连续环境不可达；B1 链成本是路径上界，不能自动叫最短路；Oracle 是离散环境模型参考，不是连续最优值或训练标签。

复现（仓库根目录；Python 使用 `../quasimetric-rl/.venv/bin/python`）：

```sh
export MPLCONFIGDIR=/tmp/qrl-execution-mpl
PY=../quasimetric-rl/.venv/bin/python
RUN=results/full_replay_four_path_reproduction
$PY -m pytest -q experiments/full_replay_four_path/test_full_replay.py experiments/four_path_oracle_execution/test_oracle.py minimal_qrl/test_comm_inspection_baselines.py
$PY -m experiments.full_replay_four_path.evaluate prepare --root "$RUN"
$PY -m experiments.full_replay_four_path.train --output "$RUN/train" --seed 606 --threads 4
$PY -m experiments.full_replay_four_path.run_evaluation --root "$RUN" --workers 4
$PY -m experiments.full_replay_four_path.analyze_values --root "$RUN"
$PY -m experiments.full_replay_four_path.evaluate candidates --root "$RUN"
$PY -m experiments.full_replay_four_path.independent_candidates --root "$RUN"
$PY -m experiments.four_path_execution.audit_replay --root "$RUN/original"
$PY -m experiments.four_path_execution.audit_replay --root "$RUN/independent"
$PY -m experiments.full_replay_four_path.objective_diagnostic --root "$RUN"
$PY -m experiments.full_replay_four_path.report --root "$RUN"
```

恢复训练：同一 output 和所有原配置参数，加 `--resume "$RUN/train/latest.pt"`。checkpoint 含模型、Adam、完整逐边乘子、访问计数、采样排列/游标、Torch RNG 与数据指纹；原子替换 latest。恢复后已写训练日志可能保留中断到 checkpoint 之间的重复诊断，按 step 和恢复段区分，不丢弃失败历史。final 固定对应 step 25,000；COMPLETE 表示预算完成，不表示通过约束或导航。

输入依赖：`data.py` 固定指向两个既有 `verified_replay.pt` 缓存，并校验完整 replay 指纹；四路径 production 构造/适配器继续复用原目录的 bank 与既有模型文件。历史七个基线依赖 `results/four_path_oracle_execution_20261017/main` 的原始 episode 记录，以及五个 `qp_numerical_fix` checkpoint。所有这些旧文件均未修改。报告区分“新增执行 448 次”与“复用 448 次”，不重复制造基线样本。
