# 专家辅助 U-trap 精确转移网络

本目录只构造和审计机制诊断数据，不启动正式训练。原环境、成本、成功条件、生产终态 canonicalization、QRL 网络和 MPPI 均不修改。专家辅助来源为本轮助手编写的显式 waypoint/控制设计及端点几何 shooting，没有真人示教、外部专家标注、Oracle value 或最短路标签。

默认数据：`results/expert_utrap_network_20261019/main`。报告：该目录下 `audit/report.md`；可加载数据：`verified_replay.pt`。

## 构造与审计

在仓库根目录执行（输出目录必须不存在，拒绝覆盖旧产物）：

```sh
export MPLCONFIGDIR=/tmp/qrl-expert-mpl
PY=../quasimetric-rl/.venv/bin/python
$PY -m pytest -q experiments/expert_utrap_network/test_network.py
$PY -m experiments.expert_utrap_network.build --output results/NEW_EXPERT_NETWORK
$PY -m experiments.expert_utrap_network.audit --root results/NEW_EXPERT_NETWORK
```

`build` 先执行 64 个 U 内前缀、4 条上下绕行片段、2 条终态接近片段，再将 64 个指定组合各从头重放到合法成功。出口与东侧交汇的全部 `(x,y,heading)` 及完整 42 维观测精确相等才允许接续。shooting 仅产生动作建议，绝不写入吸附后的状态。`audit` 独立重放所有段和全部轨迹，并实际执行 512 种组合（448 种没有存为完整训练轨迹），检查未改动的 300 步上限。审计没有把额外组合加入 replay。

四路径报告指出独立链恢复不代表分支和拼接；完整 replay 含大量无成功链源；witnessed 限制抑制膨胀但没有完全恢复长程距离；PHR 外层更新改善局部残差同时压缩价值。本数据把所有源的成功链支持补足，以便后续分别诊断优化可行性、距离紧致性和分支排序。它不直接证明其中任何机制已经修复。

文件契约：

- `verified_replay.pt`：沿用 `schema_version / sha256 / raw` 和 `MultiEpisodeData` 格式，含真实转移与分开存放的零成本合法终态 anchor。环境 reward 的相反数是一步成本。兼容字段 `goal_return_cost` 全零、`goal_return_mask` 全假；无累计成本/Oracle/最短路标签。
- `segments.json`、`trajectories.json`：真实状态、动作、观测、一步 reward，以及片段组合与 context 索引，用于逐位复现；不提供学习用路径成本目标。
- `construction_attempts.json`：保留 shooting 长度候选的接受/失败及原因。失败提议不进入数据。
- `manifest.json`、`source_snapshot/`：环境签名、原始输入及源码哈希、运行版本、数据指纹与源码副本。
- `audit/`：可达性、分支/交汇度、成本核验、重组成功记录、图最短路和 B1 路径上界，仅供事后审计；训练加载器完全不读取。

按原生产采样器，task push 使用物理源；每个源在自己的 context 内有完整成功后缀和合法 anchor，因此 `all` 与 `witnessed` 等价。B1 仍是固定 seed=42 的同轨迹未来状态配对；B1 路径成本仅是上界，不自动等于图最短路。不同 context 只表示采集 episode，不是环境任务；物理交汇按相同完整观测和任务判断。

## 后续 QRL 接入（本轮不执行训练）

```python
from experiments.expert_utrap_network.data import load_data

data = load_data('results/expert_utrap_network_20261019/main', task_mode='all', seed=42)
# data.raw: MultiEpisodeData
# data.nodes, src, dst, cost: 生产 quotient 后的节点、逐记录边、一步成本
# data.rows, task_src, goal, pair: 物理源与固定 task/B1 配对
```

此接口兼容现有 `experiments.four_path_factorial.run.IndexedIQE` 及 `training_objective`：后续应显式使用 `IndexedIQE(data, 'encoder', 0, initialization_seed=<预先固定的 seed>)` 从生产网络重新初始化，再为**本数据的记录行数**创建 PHR dual。不要读历史四路径 checkpoint 或复用其 339 条 multiplier。`test_network.py` 仅做一次生产模型前向兼容性检查，没有优化器、反向传播、参数更新或模型文件。

现有 factorial CLI 的 `load_data()` 和完整 replay CLI 的 `load_union()` 固定旧数据，不能直接把旧命令输出路径换成本数据。后续需在独立训练 harness 中显式调用本加载器，复用共享 encoder/IQE、原 task/B1 push 与 PHR 实现；不修改环境或 MPPI。旧 `graph_reference()` 假定每条边都在最短成功链上，不适用于本分支图；参考只应在独立分析中读取本 `audit/graph_reference_audit_only.npz`。本次不提供会误启动训练的命令。

未来应预先固定训练预算、采样/权重和终点，再同时报告全量一步约束、合法终态映射、长程误差和同出口不同分支排序。按记录行采样会重复共享中后段；去重边数与重复数已单列。若改变为去重采样或更换 B1 支持，须作为新协议，而不是声称与当前数据目标完全相同。

这是专家辅助、设计起点和有限网络上的诊断；不继承旧四路径的 held-out 排除协议，不宣称无专家学习、连续状态最优性或独立泛化。不同控制段的浮点逐位复现需兼容的环境源码和数值运行时；冻结动作重放为主要审计依据。
