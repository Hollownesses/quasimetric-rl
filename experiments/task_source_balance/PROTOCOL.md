# task push 去重源均衡：冻结配对协议

仅改变 task 经验目标分布。R0 为 `-.6 mean(d(s,G))`；R1 为
`-.6 mean(w(s)d(s,G))`，全局 `w(s)=N/(M n(s))`，不做 minibatch 权重和归一化。
N/M 从实际启用 task 的物理行、训练加载器的完整观测/任务 canonical ID 计算。
不删除重复行，不改变记录采样，不改变 `.4 B1`、逐行 PHR 或 dual。

原训练器与 50452ef 完全一致；R0 直接调用其 advance，测试另用 e06e2ad
源代码做数值更新对照。eb2967d 的参考修正只用于完成预算后的评价。
不读取 e162dd3/b7c940f 的优化状态，也不继承它们的训练入口。

原seed606实验的共同设置：原 encoder/projector/IQE，seed606，AdamW weight_decay=0，
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

原运行的冻结清单覆盖229个仓库Python源文件；新运行按当前源码重新冻结清单。后续新增分析/监督代码不参与训练，
其源码另随分析结果保存。实际运行环境与外部IQE源码另见 `runtime.json`、
`environment_packages.json`、`external_source_manifest.json`。
主试验不使用resume；独立4步烟雾运行验证两种模式完整恢复以及跨模式拒绝，
见 `results/task_source_balance_20261025/integration_smoke.py` 和对应日志。

## 多初始化复现（707、808）

新增两个初始化seed707、808，与已有606组成三个初始化。仅改变初始化seed；
sampler seed固定2307、B1 seed固定42，数据、模型、task/B1系数、PHR/dual、
batch256、4线程、45k+5k学习率和50k正式终点均沿用原实验。
采样器seed与模型初始化独立，可由 `prepare --sampler-seed` 显式指定；
这轮复现保持默认2307。新seed之间及其R0/R1的记录访问序列相同。

`prepare --seed` 为该seed生成并保存 `common_initial.pt`；两组分别重建初始化，
与该文件逐项核对。新seed不要求与历史606模型或全局torch RNG一致；
分析继续核对两组之间的RNG、完整采样序列、访问次数及预算。
历史606参数和配置仍用于协议一致性检查，历史模型仅作背景参考，
不能当作新seed的配对基线。默认 `prepare` 仍使用606并验证历史初始化一致。
旧版协议由训练器的历史初始checkpoint回退逻辑识别；旧输出目录及结果不改写。

在仓库根目录，以下每条命令都会完成一次预检查、并行启动该seed的R0/R1，
等待两组固定预算完成，再执行统一分析和独立结果核验。输出目录必须不存在；
两组日志为各自目录上一级的 `R0.log`、`R1.log`。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-source-balance ../quasimetric-rl/.venv/bin/python -m experiments.task_source_balance.pair --seed 707 --sampler-seed 2307 --output results/task_source_balance_multiseed_20261025/seed707/main
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-source-balance ../quasimetric-rl/.venv/bin/python -m experiments.task_source_balance.pair --seed 808 --sampler-seed 2307 --output results/task_source_balance_multiseed_20261025/seed808/main
```

任一训练子进程失败会使启动器报错，且不执行终点评价；中断后可沿用上面的
单组train入口和已有 `protocol.json`，以 `--resume .../latest.pt` 恢复。
启动器不覆盖已有目录、不更改预算，也不执行参考驱动的配置选择。
新运行的环境/测试日志若存在则纳入结果清单；缺失时不会伪造通过声明。
本次代码改动仅用临时目录的短预算测试初始化、两种模式及完整恢复，
不预先启动这两组50k正式实验，也不产生新的正式结果报告。
