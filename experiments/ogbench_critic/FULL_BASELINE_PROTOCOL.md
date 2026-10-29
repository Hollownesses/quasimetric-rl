# E3：官方完整预算基线核验（预先冻结）

目的：区分100k模型的距离场失败是短预算学习/优化困难，还是原方法在官方完整预算下仍然存在的恢复问题。此对照是用户总目标所需的结论边界检查，不代替E1的配对critic因果对照。

- 官方 commit a977253ce5b88434f4e4f531c50047cc95993a51。
- pointmaze-medium-navigate-v0，seed0，从头初始化，完整1,000,000次更新；不加载100k模型。
- 采用 hyperparameters.sh 对应QRL配置：alpha=0.0003；其余agent配置全部官方默认（包括actor、latent dynamics、采样、batch1024、学习率3e-4和eps0.05）。
- 官方完整预算频率：每5000更新训练/验证日志，每100000更新评价，全五任务各50个episode；第1步评价保留；1M保存官方checkpoint。固定1M终点，不挑选最好评价或延长预算。
- 原始官方train/val流程，独立运行目录 results/ogbench_critic_20261029/official_full_seed0；CPU JAX0.4.35，离线W&B，video_episodes=0。本机环境和源码与100k基线一致。
- 正式完成后验证Adam count=1M、checkpoint加载、有限性及官方终点评价，另用E1同一固定bank检查真实后继贪心和参考图。官方actor与诊断控制分开报告。
- 本运行采用官方完整预算的评价频率，与原100k试跑每10k评价不同。官方评价消耗全局NumPy RNG，故二者虽相同初始化seed0，但后续训练批次流不完全相同。不能将差异全部归因于额外更新次数；它能检验原算法在官方完整协议下是否恢复。
- 独立初始化核验仍需另行完成；本seed0完整预算不能替代多初始化结果，也不能凭某次task3成功率称稳定改善。

执行命令（仓库根目录）：

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
$P experiments/ogbench_official_qrl/protocol.py navigate full results/ogbench_critic_20261029/official_full_seed0
$P experiments/ogbench_official_qrl/launch.py results/ogbench_critic_20261029/official_full_seed0
$P experiments/ogbench_official_qrl/check_checkpoint.py results/ogbench_critic_20261029/official_full_seed0 1000000
```

launch.py拒绝重复启动。恢复工作时须检查已有进程/句柄及runtime.json，不能因观察超时重新启动。
