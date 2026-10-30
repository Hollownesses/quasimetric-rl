# 模型独立的物理诊断与精确评价缓存

`core.py`保留原E0/E1的几何、完整状态恢复及1000步episode语义；历史实验依赖其固定摘要，后续入口不改该文件。

2026-10-30新增的pointmaze专用实现，在固定QRL同25例完整观测/动作/距离/成功/步数上逐位复现。先前单/四线程物理批量核验各超时120秒并保留，不当通过；加入固定模型精确输入评分缓存后76.76秒完成全量。性能工具没有训练或改变成功率，当前仍16/25、task3为0/5。

使用独立官方JAX环境。原数据/模型与baseline_greedy、evaluation_bank及E6保存的完整结束integration state必须存在。详细门槛见[缓存核验协议](CACHED_SCORE_PROTOCOL.md)，当前原始结果位于`results/ogbench_critic_20261029/cached_pointmaze_verification`。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P experiments/ogbench_critic/test_exact_score_cache.py
# 若在全新研究根目录复现，先准备相同bank/baseline_greedy/E6结束状态，
# 并用本入口生成35组原生物理候选；它可能按120秒上限退出，不能当完整通过。
$P experiments/ogbench_diagnostics/verify_fast_pointmaze.py "$G" --output "$G/fast_pointmaze_parallel_verification" --threads 4
# 当前目录已有上述候选，不必重复前一个命令；以下始终使用全新输出目录。
$P experiments/ogbench_diagnostics/verify_cached_pointmaze.py "$G" --output "$G/cached_pointmaze_new_verification"
```

新实验显式使用方式（必须先核对完整verification.json的passed、25例和当前源码摘要；原core默认不自动切换）：

```python
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
from exact_score_cache import ExactScoreCache  # 加入experiments/ogbench_critic导入路径
engine = FastPointmaze(env, nthread=4)
score = ExactScoreCache(frozen_agent, audit=True)
try:
    info, trace = engine.rollout(case, score)
finally:
    engine.close()
```

每个新agent重新创建评分缓存，不能在训练过程中替换agent。默认首次hit重新计算并逐位核对；完整episode始终通过原env.step，选中后继始终精确核对。`encoder=None`、x64关闭及固定状态pointmaze适用，不支持teleport/图像/其他动力学。缓存容量只影响开销，不影响输出。

原始模型、JSON/NPZ/日志不提交Git。后续研究遵循`../ogbench_critic/CURRENT_LIMITS.md`，不换seed证明非偶然性，不启动1M；评价器性能不代表critic稳定改善。
