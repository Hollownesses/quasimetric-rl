# Rhalf：固定 alpha=0.5 的部分源均衡

唯一新增正式训练为 Rhalf，alpha=0.5。已有 R0/R1 在核对配置、数据/模型文件指纹、
完整零步状态、50k预算与采样哈希后直接复用；不重跑、不增加seed、不搜索alpha。
新增独立入口，旧代码、模式与全部旧产物保持原样。

加载 `results/connected_expert_utrap_20261021/main`，按实际启用task的物理行和既有
完整观测/任务canonical node计算 N、M、n(s)。全局 w=N/(M n(s))，
w_half=.5+.5w。损失 `-.6 mean(w_half*d(s,G))`；不除minibatch权重和。
全量task目标是记录均值与去重源均值的等权混合，明确改变经验目标分布。
非task行权重为0。重复记录、轨迹、固定B1及一步成本全部保留。

B1 `-.4 mean(d(s,g_B1))`、逐记录PHR rho100及每访问用primal前残差eta1更新dual，
直接复用原更新函数。alpha=0直接调用原R0，alpha=1调用原R1；端点只作单元测试。
alpha与权重digest均进入checkpoint签名；resume校验alpha、模式、权重及完整冻结协议。

共同协议严格沿用旧实验：seed606、sampler seed2307、B1 seed42、batch256、
task_mode=all、每组4线程、AdamW/weight_decay0；45k×2e-5，后5k从2e-6几何衰减至2e-7。
每1000步全量审计，50k固定停止。保留epoch末尾40条小批次；50k共有12,735,416次记录访问。
重新生成零步model/optimizer/zero dual/visits/sampler/RNG/counters，逐项匹配旧R0/R1 initial.pt。
不得使用旧45k或50k训练状态。4线程不承诺最终模型逐位复现。

运行前核对旧runtime（Python/PyTorch/NumPy/平台/build）、torchqmet源码指纹。
旧代码后来新增的多seed初始化支持不改变损失/更新函数，核心AST与旧源码快照逐项相等。
当前源码全量快照和digest单独冻结；数据fingerprint仍为旧值。

重复次数分组预先固定为 1、2–4、5–16、17–64、65+，仅基于task源计数，
不按图参考/结果选边界。每组同时报告去重源及记录加权的MAE/RMSE/NRMSE/bias，
以及每源相对R0/R1改善/退化数和误差差值。共享后缀下这些不是独立统计样本。

全部训练结束后用现有evaluate/decomposition及修正参考
`results/connected_reference_fix_20261024/main` 统一重算R0/Rhalf/R1，核对R0/R1与旧报告一致。
包括records、unique_sources、最远四分位、165独立起点、307原始起点、分支/完整路径、
记录/去重边局部违反、累计边项和三角松弛。正式终点为50k，45k仅事后分析。
图参考/Oracle不参与权重、训练、停止或模型选择。到达率主要由DAG结构保证。

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-partial-balance
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/task_partial_balance/test_partial.py -q
$PY -m experiments.task_partial_balance.prepare --output results/task_partial_balance_20261025/main
$PY -m experiments.task_partial_balance.train --protocol results/task_partial_balance_20261025/main/protocol.json --alpha 0.5 --output results/task_partial_balance_20261025/main/Rhalf > results/task_partial_balance_20261025/main/Rhalf.log 2>&1
$PY -m experiments.task_partial_balance.analyze --root results/task_partial_balance_20261025/main
```

以上输出目录须全新。若确有中断，原训练命令追加 `--resume .../Rhalf/latest.pt`；
不得更换alpha、协议或初始checkpoint来绕过检查。正式训练完成后不再追加预算。
最终报告分别判断距离、高估、决策与可行性，并保留单seed、4线程、固定图限制。
