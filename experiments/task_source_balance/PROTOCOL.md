# task push 去重源均衡：冻结配对协议

仅改变 task 经验目标分布。R0 为 `-.6 mean(d(s,G))`；R1 为
`-.6 mean(w(s)d(s,G))`，全局 `w(s)=N/(M n(s))`，不做 minibatch 权重和归一化。
N/M 从实际启用 task 的物理行、训练加载器的完整观测/任务 canonical ID 计算。
不删除重复行，不改变记录采样，不改变 `.4 B1`、逐行 PHR 或 dual。

原训练器与 50452ef 完全一致；R0 直接调用其 advance，测试另用 e06e2ad
源代码做数值更新对照。eb2967d 的参考修正只用于完成预算后的评价。
不读取 e162dd3/b7c940f 的优化状态，也不继承它们的训练入口。

共同设置：原 encoder/projector/IQE，seed606，AdamW weight_decay=0，
batch256，sampler seed2307，B1 seed42，task_mode=all，rho100，
逐访问用 primal 前残差以 eta1 更新对应乘子。前45000步 lr2e-5；后5000步
从2e-6几何下降至2e-7；4线程，每1000步全量审计，50000步固定停止。
保留原 epoch 尾部40条小批次，50k共12,735,416次记录访问。

每组独立重建零步状态，逐项核对原 initial.pt 的 model/optimizer/dual/
visits/sampler/RNG。初始状态一致，不要求最终网络/乘子一致。
保存有序采样 hash chain、逐行访问计数、完整 sampler/RNG/optimizer/dual、
权重及模式、配置和源文件哈希。resume 仅接受相同冻结协议与权重模式；
初始、45k、50k checkpoint 全部保留。四线程不承诺训练结果逐位复现。

可复现命令（在仓库根目录，输出必须是新目录）：

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-source-balance
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/task_source_balance/test_balance.py experiments/connected_expert_utrap/test_task_only.py experiments/connected_expert_utrap/test_reference.py -q
$PY -m experiments.task_source_balance.prepare --output results/task_source_balance_20261025/main
$PY -m experiments.task_source_balance.train --protocol results/task_source_balance_20261025/main/protocol.json --mode record --output results/task_source_balance_20261025/main/R0 > results/task_source_balance_20261025/main/R0.log 2>&1 &
$PY -m experiments.task_source_balance.train --protocol results/task_source_balance_20261025/main/protocol.json --mode source --output results/task_source_balance_20261025/main/R1 > results/task_source_balance_20261025/main/R1.log 2>&1 &
wait
$PY -m experiments.task_source_balance.analyze --root results/task_source_balance_20261025/main
$PY -m experiments.task_source_balance.validate_results --root results/task_source_balance_20261025/main
```

若中断，使用完全相同命令追加 `--resume <该组目录>/latest.pt`。
不跨模式、不从历史终点恢复。所有测试与冻结先于正式训练。

评价固定 final.pt；45k 仅事后过程分析，历史50k同一评价器复算。
报告 records/unique_sources/longest_quartile/165 unique_starts/307 starts
任务 MAE、RMSE、NRMSE、bias；58分支选择、严格排序、regret；独立起点完整
路径regret；记录/去重边局部违反；参考最短路径上的累计边项及三角松弛。
`bias=prediction-reference`；分解使用相反符号
`reference-prediction=sum(cost-edge_distance)+(sum(edge_distance)-task_distance)`。
该图无环，唯一汇点目标，最长275步，307/307到达仅为完整性检查。
单 seed、有限固定图、多线程；不声称连续控制泛化或 MPPI 能力。

冻结清单覆盖229个仓库Python源文件；后续新增分析/监督代码不参与训练，
其源码另随分析结果保存。实际运行环境与外部IQE源码另见 `runtime.json`、
`environment_packages.json`、`external_source_manifest.json`。
主试验不使用resume；独立4步烟雾运行验证两种模式完整恢复以及跨模式拒绝，
见 `results/task_source_balance_20261025/integration_smoke.py` 和对应日志。
