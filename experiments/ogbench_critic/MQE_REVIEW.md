# MQE 论文与官方实现核对（用于下一轮设计）

核对日期：2026-10-29。仅查阅资料，尚未安装或启动MQE；E1协议不变。

官方仓库 [WJ2003B/mqe-release](https://github.com/WJ2003B/mqe-release)，查阅版本固定为 [04c2697405233e4313fab43e05bd4d95f1b3e111](https://github.com/WJ2003B/mqe-release/commit/04c2697405233e4313fab43e05bd4d95f1b3e111)。

| 方面 | 当前QRL / E1 | MQE固定源码 |
|---|---|---|
| 表示 | 单个状态IQE编码器 | 状态/状态动作表示、双critic、MRN |
| 价值目标 | 随机目标，距离最大化与乘子约束 | 未来目标及批内非对角目标，备份与动作不变性 |
| 中间状态 | 原始一步转移 | 几何采样并截到目标索引（可相等），另混合一步后继 |
| 备份梯度 | E1 goal差两端均有梯度 | 后继距离stop-gradient，截断LINEX形式 |
| 主要代码默认 | batch1024，gamma0.99 | batch256，gamma0.995，lambda0.95，next-state概率0.2 |

上述事实见 [critic与actor实现](https://github.com/WJ2003B/mqe-release/blob/04c2697405233e4313fab43e05bd4d95f1b3e111/impls/agents/mqe.py#L55)、[默认配置](https://github.com/WJ2003B/mqe-release/blob/04c2697405233e4313fab43e05bd4d95f1b3e111/impls/agents/mqe.py#L311)及[目标/中间状态采样](https://github.com/WJ2003B/mqe-release/blob/04c2697405233e4313fab43e05bd4d95f1b3e111/impls/utils/datasets.py#L234)。

一个实际复现差异：代码默认actor alpha=0.1，而论文附录表3的pointmaze值为0.03。后续MQE方法对照应显式使用并记录论文对应值，不能把代码默认直接称作pointmaze论文配置。固定版本的hyperparameters.sh共1105行，没有MQE命令；不能凭该文件宣称找到MQE逐任务配置。[论文表2/3，第15页](https://mqe-paper.github.io/static/figures/mqe.pdf#page=15)

代码diag_backup默认0.5，注释另建议locomotion为0.2–0.4；尚未据此选参。任何后续运行须在训练前声明实际取值与来源，不按task3结果搜索。

设计边界：完整MQE方法对照会改变多个变量，不能算“只给QRL加多步”的单变量实验。若在QRL内做长程消融，须另设同表示、同预算的一步对照，并明确离线目标；不得把任意行为路线长度当作最优成本等式。本笔记不决定E2方案，须先分析E1固定终点，且多步本身不是新颖贡献。
