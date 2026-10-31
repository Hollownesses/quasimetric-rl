# 正向方法的独立复现入口

2026-10-31。原官方100k基线入口仍在[官方README](../ogbench_official_qrl/README.md)。本文件打包E10教师、E16价值学习和E17-B官方策略提取；这是改进实验，不替代或改写原官方基线。当前禁止换训练seed证明非偶然性、禁止启动官方1M，见[运行约束](CURRENT_LIMITS.md)。

## 输入与固定预算

只需要原seed0官方100k checkpoint及其同目录flags.json、固定官方源码/独立虚拟环境、完整官方训练数据。无需先运行E6/E12/E15等历史控制组，也不依赖既有评价bank。安装和生成原100k的命令见官方README；本次已存在，不重复训练。

固定方法：从全部1,000,000有效转移建E10数据支持图及抽象距离教师；原IQE网络/原Adam状态续接，batch256，价值1000步、动力学1000步、actor1000步。价值阶段半批原查询，半批固定8候选中当前双向拟合误差最大目标，使用同一Sobol查询扩展；冻结actor/dynamics/lam及其Adam矩。随后冻结critic/lam，先使用官方dynamics_loss适配，再冻结动力学并使用官方ddpgbc actor_loss。保持alpha=0.0003、lr=0.0003、const_std、512×3网络及100%未来actor目标。

价值阶段NumPy采样2001；动力学阶段重置同一采样2001，actor继续该流，与E16/E17-B一致。不是更换初始化seed；全程从同一个官方训练seed0终点开始。新增教师包含局部可逆先验及连续边代价，标签不是严格MDP最优成本；价值目标已改变，阶段化优化也不是官方main联合训练。原flags中的batch_size=1024描述基线，本研究实际sample(256)由protocol.json另行明确记录。

## 命令

从仓库根目录执行，每次指定全新目录，入口拒绝覆盖。以下只有显式选择improve才执行完整改进配方；没有默认训练模式。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
BASE_MODEL=results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628/params_100000.pkl
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/ogbench_critic_mplcache
# 只冻结计划/源码/配置，不加载数据、建图或更新。
$P "$E/run_improvement.py" --base-checkpoint "$BASE_MODEL" --mode plan --output results/improvement_plan_new
# 各阶段2次更新，官方五任务各1完整episode，上限120秒。
$P "$E/run_improvement.py" --base-checkpoint "$BASE_MODEL" --mode smoke --output results/improvement_smoke_new
# 仅提供复现命令，本次不再启动：各阶段1000次更新，总上限600秒。
$P "$E/run_improvement.py" --base-checkpoint "$BASE_MODEL" --mode improve --output results/improvement_formal_new
```

独立评价子进程直接调用官方actor/evaluate，正式模式全部五任务各50episode，temperature=0、gaussian=None、video=0；不导入教师/图/真实后继候选控制。评价流rng0只是匹配评价输入，不是另换训练seed。阶段计数依次101000/102000/103000，不冒称103k官方联合训练；不做中途成功率评价、挑checkpoint或追加更新。

输出包括protocol.json、agent_config.json、source_snapshot、各阶段CSV、图/距离缓存、三个完整checkpoint、独立评价逐episode轨迹、analysis.json。每阶段核验非活动模块参数/Adam矩冻结、有限性、计数和完整保存加载；异常保留failure.json及日志，不换任务/缩数据/自动续跑。

## 本次验证及性能来源

本次只执行standalone_smoke，18.93秒：value/dynamics/actor各2次，count100006/TrainState.step100007，三阶段冻结及完整保存加载通过。独立官方五任务各1个完整episode，步数126/124/1000/144/93；task3超时是有效完成的数值评价，不将4/5当正式性能。原checkpoint SHA未变，子进程无教师导入。

原始记录：results/ogbench_critic_20261029/standalone_smoke及同级.log。打包入口的完整1000+1000+1000模式未再次运行；正式性能证据来自已完成的[E16](../../results/ogbench_critic_20261029/hard_goal_1000/report.md)和[E17配对对照](../../results/ogbench_critic_20261029/extraction_1000/report.md)。因此这里只声称入口smoke可用，不声称打包后又独立复现一次92.4%，也不承诺跨平台逐位一致。

相关单元测试沿用test_hard_goal.py（三项）、test_extraction.py（三项）；本入口实际端到端smoke覆盖组合、数据建图、三阶段保存加载和独立官方评价。无需为了交付再重复正式训练或跑seed1/1M。
