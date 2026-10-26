# R2：task源均衡固定，PHR按唯一物理边均衡（seed606）

本轮仅增加R2。数据 `results/connected_expert_utrap_20261021/main`；从加载的全部物理记录，
以完整canonical source、target、原始action字节及原始一步cost定义唯一边。无位置/近邻合并。
实测N、K及m(e)，预期N=42536、K=6958仅作为检查。全局v=N/(K m(e))，平均v=1，
每条唯一边的总权重相等；所有记录、轨迹和逐记录multiplier保留。

唯一目标变化：保留R1的全局task源权重w=N/(M n(s))与固定B1，
`loss = -.6 mean(w*d_task) - .4 mean(d_B1) + mean(v*PHR_i)`。
代码先计算逐记录 `num_i=relu(lambda_i+rho*h_i)^2-lambda_i^2`，
再 `(v*num).mean()/(2*rho)`；权重在均值之前作用，绝不对phr_terms返回的标量加权，
不按minibatch权重和归一化。共同分母最后除，保证v全1时与原R1舍入/更新行为一致。
同一边不同副本的lambda分别进入各自PHR；等价于每条边内部对各副本PHR取平均，再对边均匀平均，
不等价于取一个代表副本的PHR。

网络、AdamW/weight_decay0、seed606、sampler seed2307、B1 seed42、batch256、4线程、
task_mode=all、rho100均同R1。dual完全不加权：每次物理记录访问，以primal前残差执行
`lambda_i=relu(lambda_i+h_i)`，eta1。不合并multiplier，不调整采样。
45k×lr2e-5，随后5k由2e-6几何衰减到2e-7，固定50k停止；每1000步全量审计。
保留每epoch末尾40条小批次；总记录访问12,735,416次。45k仅作事后分析。

从与旧R1一致的零步model/optimizer/zero dual/visits/sampler/RNG/counters重新初始化，
逐项核对initial.pt。旧R0/R1/Rhalf配置、数据、完整初始化、预算和产物hash通过核验后复用。
旧代码/产物不修改，不回滚，不从旧45k/50k恢复。Python/PyTorch/NumPy/build/平台与外部IQE源码核对旧环境。
源码/data/配置在正式训练前冻结；保存initial/phr(45k)/final(50k)、日志、边分组、权重及完成标记。
resume校验实验模式、task/edge权重、edge partition和完整协议签名，恢复优化器/逐记录dual/访问/RNG/sampler/采样hash chain。

后验分析统一调用既有图评价器、修正参考 `results/connected_reference_fix_20261024/main`。
训练/权重生成/停止/选择模型不读取图参考或Oracle。固定R0路径指seed606 R0的50k冻结贪心路径，
165去重起点等权为主、307原始起点为辅助；每个模型的正残差在完全相同的R0路径上求和。
另报告各自路径上的正违反，但不能以路线更换代替固定路径比较。
分支后继误差差分使用 `epsilon(t_choice)-epsilon(t_reference_best)`，
验证 `predicted_score(choice)-predicted_score(best)=regret+error_contrast`，并报告严格参考候选对的排序误差。
边副本分组预先固定为1、2–4、5–16、17–64、65+；记录/唯一边均报告正违反均值、p90/p95/p99和最大值。
违反数阈值1e-5。累计正违反是模型残差诊断，不是额外环境成本，也不等价于路径regret。

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qrl-phr-edge
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest experiments/phr_edge_balance/test_edge_balance.py -q
$PY -m experiments.phr_edge_balance.prepare --output results/phr_edge_balance_20261026/main
$PY -m experiments.phr_edge_balance.train --protocol results/phr_edge_balance_20261026/main/protocol.json --output results/phr_edge_balance_20261026/main/R2 > results/phr_edge_balance_20261026/main/R2.log 2>&1
$PY -m experiments.phr_edge_balance.analyze --root results/phr_edge_balance_20261026/main
```

输出目录必须全新；如中断，相同训练命令追加 `--resume .../R2/latest.pt`。
本轮只seed606，不追加预算、不搜索配置。结论分别判读局部可行性、远端紧致性和路径决策；
DAG上307/307到达仅是完整性检查。保留单seed、4线程非逐位复现和固定图范围限制。
