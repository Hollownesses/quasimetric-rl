# 连通专家图参考修正

仅修正事后评估，不重新建图、不训练模型、不改写旧参考。训练目标仍由物理行索引读取，未受此错误影响。

`goal` 的非物理行默认值 0 被旧审计误当作目标。现在只从物理行提取目标，并检查零距离节点与合法目标一致、每个非终态满足 Bellman 最小值等式。

合法目标：[175]。变化节点数：1。

| 节点 | 原参考 | 修正参考 |
|---|---:|---:|
| 0 | 0.000000000 | 595.484283336 |

原始 replay、状态编号、动作、成本、B1 与 checkpoint 均保留。新参考供后续统一评估；该修正不改变模型输出。

复现：`python -m experiments.connected_expert_utrap.reference --root results/connected_expert_utrap_20261021/main --output results/NEW_REFERENCE`。输出目录必须不存在。
