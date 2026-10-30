# M6：官方固定评价样本的critic/actor反事实分离

2026-10-30，E17提交0cd7234后冻结。官方actor A200/250、B231/250，B task3仍19失败。本项只读检查相同样本上的神经critic，不能再以E16已知bank25/25替代新样本核验。

## 初态复建与强核验

E17保存了完整轨迹和结束integration，但官方reset初始integration未另存。按完全相同官方评价rng0复建：先np.seed0/env.reset(seed0)/np.seed0，每个任务前消费原evaluate的一次np.random.randint(0,2**32)，五任务各50次相同reset，不执行旧完整策略episode。该操作只适用于本无传送、step不采随机数的PointMaze；不是换seed复现训练。

全部250起点/goal必须与E17 A/B逐位相同。task3的50例保存完整reset integration后，再用原core.load_case实际恢复，分别执行两组已保存首个动作，要求下一完整观测与各自原轨迹第一后继逐位相同，共100次核验；不能只凭相同二维位置认定状态一致。若任一不符保留失败并停止，不静默放宽精度或换样本。

## 固定反事实

从相同50个task3完整初态，分别用原100k与E16固定critic执行原121动作真实后继贪心、完整1000步/成功终止，神经forward评分，不用潜在动力学或actor。E17两组critic未更新且digest逐位相同，因此复用原/E16模型，不重新训练。保存全部观测/动作/goal/state_values/结束integration及逐例成功，源码沿用已验证T1物理与精确输入缓存，每个所选实际后继仍逐位核验。

另从E17 B全部19个task3失败结束integration出发，用同一个改进critic继续贪心。为诊断重新给予1000步TimeLimit，不是原episode的成功，也不能把续行结果加进31/50官方指标。不按失败模式挑subset，全部19例保留。若有失败，如实区分初态泛化、末状态局部陷阱和提取差距，不自动归因BC或动力学。

总240秒含复建/100首步核验、两组100个完整episode及19个末状态续行；超时不扩预算/缩episode，保留日志/部分轨迹，不重启。无训练更新、新目标扰动seed、1M或教师/图查询，无新增参考场/覆盖审计。模型/原轨迹/源码哈希前后不变。

测试用实际官方evaluate在确定mock环境对照复建的250 reset序列，另用step耗随机数的mock作负例，确保不泛化跳过step的假设。主运行必须通过实际全部250/100强核验。

报告按50个起点配对统计原/改进critic与原/改进actor的得失，19续行单独报告，不能当官方策略成绩或连续最优证明。只限定当前训练初始化、固定目标流与候选控制；不要求seed1/1M补跑，不宣称跨seed稳定。完成源码/测试/协议/中文报告/引用图单独commit；raw official_counterfactual的bank/JSON/NPZ/日志保留本地，不提交。
