# E17：冻结critic的官方动力学/actor提取对照

2026-10-30，E16提交46aa095后冻结。E16独立无图神经贪心25/25、task3五例全部下降，通过预定critic门槛；原actor/dynamics尚冻结。本项检验原官方提取方法能否利用新距离，不将贪心诊断代替最终策略。

## 两组与固定阶段

A从原100k完整QRL出发，B从E16固定1000终点出发；两者actor/dynamics/lam参数及其Adam矩、agent RNG起始完全相同，唯一学习输入差别是value参数及对应已冻结矩。B已有1000次value更新，不能称同总训练预算；本项只匹配额外提取更新与数据。

两组先各1000次仅dynamics更新，直接调用固定官方QRLAgent.dynamics_loss，value/lam/actor参数及Adam矩冻结。再各1000次仅actor更新，直接调用QRLAgent.actor_loss(ddpgbc)，value/lam及已适配dynamics参数/矩冻结。阶段分开、critic完全冻结，属于官方损失的提取适配实验，不是原main联合训练或论文预算复现。

lr3e−4、actor alpha0.0003、512×3网络、const_std=True、原DDPG+BC归一化和原actor future-goal采样100%保持完整官方配置。batch256，np seed2001仅在正式开始设置一次，共2000个原官方GCDataset批次连续流，两组逐次共用同一批，阶段间不重置RNG。原轨迹/valid/future语义不改，不使用图/教师、评价目标、墙体/路线、模拟器动作搜索作训练标签。

原Adam矩/count保留，各阶段只将活动模块交给Adam，冻结模块不受动量漂移。A初始count100000，B101000，终点分别102000/103000；这些只是分阶段额外更新的优化器计数，不能称官方联合102k/103k。由于已有count很大，FP32 Adam两项bias correction均为1，测试核验计数相差1000不改变同梯度更新；不伪造/重置count来匹配。agent RNG沿用官方update/total_loss的两次split顺序，虽const_std推理确定，仍完整保存。逐阶段/终点全状态往返、有限性、冻结参数及矩、critic输出逐位核验；原模型与E16文件不变。

先实际smoke：每组dynamics两步、actor两步，完整保存加载/冻结检查，再回到原两组初态及seed2001正式运行。不恢复smoke，不看中间成功率改预算。每100步记录训练及固定2048 val损失/actor统计，保存两阶段固定终点，不挑最好。第一阶段(s,s_next,value_goal)流SHA须同E16前1000，另对actions/actor_goals及两阶段分别哈希。

## 评价与预算

总600秒（10分钟）硬上限含编译、两组共4000提取更新、核验与完整评价，超时保留日志/完成checkpoint，不扩预算、缩episode或重启。无额外critic训练、换seed、1M或独立初始化复现。

每组终点在全新独立Python进程调用固定官方utils.evaluation.evaluate，五个官方任务每任务50episode、temperature0、gaussian=None、video0；不修改官方评价器/actor。只加透明reset/终止记录和deadline检查，记录全部250条轨迹、起点/目标和末步官方info；无图/教师/动作枚举参与推理。每进程先同样env.reset(seed=0)，再np.random.seed(0)，用于两组同一评价样本流，不是新增训练seed/非偶然性重训。核验A/B全部250个起点及目标逐位相同，逐任务报告成功数/50、平均及配对得失，不冒充原100k那次评价的逐例重放。

随后在既有25-case bank上只调用原actor，完整1000步/成功终止，同原bank/原actor20/25比较新增/失去；不新增扰动seed或重跑原基线。保存完整观测/动作/goal/结束integration state。critic完全冻结，已验证的E16贪心/场不重复运行。

门槛：B相对A提高task3且不损害其他四任务，并检查既有bank是否保留原actor成功集、task3恢复。不能靠一个均值或挑episode宣称稳定；报告范围限定当前模型/固定提取预算/这次匹配评价，不作跨seed/完整预算泛化。若失败，不搜索BC权重/lr/阶段长度/追加预算挽救，保留已改进critic并根据实际提取误差另设机制检查。完成后代码/测试/协议/中文报告/引用图单独commit，raw extraction_1000保留本地，模型/JSON/CSV/NPZ/日志不提交。
