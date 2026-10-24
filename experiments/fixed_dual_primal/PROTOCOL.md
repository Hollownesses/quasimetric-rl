# 固定乘子：小批次与全量 primal 优化小规模对照（运行前冻结）

## 问题与边界

检验在原QRL增广目标下，当前网络参数的更新方式是否影响下降。使用原始50k父checkpoint `results/connected_expert_qrl_20261023/main/train/final.pt`，SHA256 `e40171f8c4d23f4d894dc64f37f29ca5d4f9e42a9f5846e4664d197afbe3e37d`；不从上一轮A/B终点续训。

两组共享encoder/IQE、数据、重复物理记录权重、0.6 task+0.4 B1 push、原局部约束`d(s,s')<=c`、rho100。固定整个lambda向量（包括非物理锚点），从父模型恢复相同参数、Adam全部状态及torch RNG。没有辅助loss、QP、最短路标签或新乘子更新。

## 冻结预算和学习率

- 每组20次完整数据遍历（pass），每遍42536条物理记录，训练前向850720条、训练反向850720条。每个记录同时提供原edge/task/B1查询；此处按记录计数，不把三项距离查询再算成三倍样本。
- mini：batch256，最后40条短batch照常取均值并更新；每遍167次Adam更新，总3340次。
- full：同一完整排列分成batch256，网络参数在167个chunk间保持不变；每个chunk loss乘以`len(chunk)/42536`，累计梯度完成后只做一次Adam更新，共20次。梯度用float64累加后转回参数dtype；禁止把167个均值等权相加或每个chunk更新网络。
- 两组恒定lr=2e−6，不按各自optimizer更新次数衰减；Adam一二阶矩、step计数、beta、eps、weight decay从同一父模型继承。lr不是为全量组调优后的值。更新频率、Adam记忆以记录数计的时间尺度、累计参数位移均不同，不能将差异全部归因于梯度噪声。
- 为精确配平全量遍历，使用共同的新无放回epoch序列，seed=2026102401；不是继续父sampler剩余半个epoch。两组每遍的排列哈希必须一致，每条记录最终恰好追加20次访问。mini仍沿用现有batch均值网络更新方式；其短batch不被人为放大或缩小学习率，full的短chunk严格按记录数加权。
- CPU单线程，两组可并发。报告实际更新次数、训练及审计时间；耗时受并发影响，只作观测，不作严格速度结论。

## 审计与监督

每遍结束计算原固定lambda下的全量目标、task/B1 push、边可行性和互补残差，保存原始记录。pass0/1/5/10/20额外计算全量梯度各分量、总范数和相对抵消比例。全量梯度审计每次共用一轮前向，分task/B1/PHR/普通Lagrangian四次反向；单列为审计开销，不混入训练预算。两组审计频率与记录量一致。

在pass0/1/5/10/15/20写监督记录并检查全有限值、lambda逐位不变、访问预算正确、终态锚点距离0。所有停止只按固定20遍预算，不因图指标或目标曲线提前停止。模型及Adam、RNG、sampler在pass边界支持确定性恢复，配置或源码漂移拒绝。

## 预先规定的失败排查（不影响主实验）

若full最终全量目标未比父模型降低超过1e−4，追加只读一步探针：分别在父模型与full最终模型固定当前lambda和完整全量梯度，比较继承Adam与仅清零一阶矩的Adam，学习率分别为2e−6、2e−7、2e−8。保留二阶矩和step计数，隔离历史一阶动量影响。探针每个候选都从同一模型/优化器状态独立出发，仅评价原全量目标，不串联、不用于修改正式checkpoint或选停点。另报前向/反向记录量和耗时。若有限探针仍无下降，只说明这些方向/步长未找到下降，不宣布PHR不适配或网络不能表达。

## 事后评价与提交

两组COMPLETE后才加载修正参考 `results/connected_reference_fix_20261024/main`，验证目标零集合与Bellman最小值等式。比较父模型、mini、full的去重source/起点/最长四分位误差、路径边项与三角松弛、58分支及307/165起点的图上决策。图上到达率不代表MPPI闭环导航成功率。

对比以相同累计训练记录数为横轴。全量目标下降也不自动等于原约束满足或最优值恢复，必须分开报告。当前固定乘子只检验该增广子问题，不宣称解出完整primal-dual问题。

必要测试包括：全量分块/稠密梯度与单步更新一致、短chunk加权、chunk期间参数不变、lambda不变、小批次单步与既有实现一致、两组遍历预算配平、确定性恢复和配置拒绝。完成后提交中文报告与源码、测试、协议、引用图片；不提交checkpoint、JSON/NPZ、日志、中间状态。

```sh
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.fixed_dual_primal.train --arm mini --output results/fixed_dual_primal_20261024/main/mini
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.fixed_dual_primal.train --arm full --output results/fixed_dual_primal_20261024/main/full
MPLCONFIGDIR=/tmp/qrl-oct24 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../quasimetric-rl/.venv/bin/python -m experiments.fixed_dual_primal.analyze --root results/fixed_dual_primal_20261024/main
```
