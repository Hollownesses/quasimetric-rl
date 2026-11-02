# 原训练问题中的可行候选比较（计算前冻结）

本项承接8177519，区分“已知较好距离场被原目标惩罚”与“原问题内尚有更好候选、当前训练未找到”。只读已有参数，在同一网络类内做预定正齐次缩放；训练更新为0。不把有限候选比较称作全局最优证书，不改变两边目标、约束、模型结构或完整重训。

1. U 候选固定为 seed606 的R0/R1 50k（原MPPI28/64、32/64）。每个候选以全部42536物理记录求 a*=min(c/d)，**不截断到1**，取 a=a*(1−1e−4)，仅缩放原 projector 最后线性层 weight/bias，重新核验全部硬约束 max h≤1e−5。记录 raw/boundary 两种函数的完整原R0目标 F0=.6 mean(Jtask)+.4 mean(JB1)，另算R1原源权重下F1作测度对照；全部原B1/goal映射保留。选择a不读取目标/参考/成败。
2. OG critic候选固定为原100k和E16 101k。动力学每个critic固定两份：原100k的共享raw dynamics、其匹配E17提取终点dynamics（原critic用A，E16用B；均已有，不追加提取）。actor不进入本项critic目标或控制。λ统一保留原参数，优化器/RNG不更新。先对每个critic的全部100万训练有效转移扫描dpos、两种dynamics逐转移损失D，求原风险R(a)=mean(relu(a*dpos−1)^2)的最大可行a*；从1起倍增至跨过eps，最多64，二分60次；若不能夹逼则标记未认证，不扩大搜索。a=a*(1−1e−4)，不截断到1。
3. OG boundary只缩放phi末层kernel/bias；对每份dynamics同时做坐标等价变换：首层kernel的latent输入行除a，action行不变，末层kernel/bias乘a。原隐藏层/LayerNorm/结构不变，预期D_new≈a D_old。实际参数重新扫描全部训练边与D，不以理论乘法代替可行性验证；要求R_actual≤eps+1e−6，报告齐次误差及参数/形状/冻结状态。该变换不是重训或重置优化器。
4. OG原距离目标P=E[100 softplus(5−J/100)]（越小越好）。critic+dynamics的原活动primal目标为P+D：actor_loss对这两模块的梯度为0；局部项作为原约束R≤eps，另用原λ的P+D+λR核验其value/dynamics梯度与官方total_loss一致。**不比较含lam_loss的total_loss前向标量**，它的stop_gradient抵消不是约束目标值。前述D按全100万物理源精确平均，P用下项固定原查询流估计。
5. 查询预算固定1024批×1024=1048576，官方train GCDataset、seed4400，单次取batch同时评全部候选/两种尺度。原三字段和完整字段流保留SHA，前32批须与c279ed0一致。逐查询原softplus在J≥0下属于[0,B]，B=100*log(1+exp(5))。对预定四个OG主比较（raw/boundary×raw/extracted dynamics，各比较E16−original）报告paired batch SE/95%区间，另报告Bonferroni四比较的Hoeffding半宽 B*sqrt(2*log(2*4/.05)/N)；P+D只给P的不确定性，因为D已全量计算。区间跨零标记未分辨，不追加查询、选尺度或选模型。raw不满足原约束者不能作为可行见证。

   统计出处补充：先识别前32批历史重放，再核验本目标native_ablation已用同流前512批；最终主要比较只用后512批、N=524288，总1024批预算/原连续流不变。范围修正不依赖后512批效应，独立采样指纹及重聚合规则详见[STATISTICAL_NOTES.md](STATISTICAL_NOTES.md)。
6. 尺度确定后才读取参考/评价输入。U raw两组64回合和R0 boundary64回合均复用；R0 boundary必须与8177519的a完全一致，否则不复用、标记不匹配。只新增R1 boundary原64回合MPPI、H10/N128/原300步；另统计同58分支/483对、165起点图成本。OG两种boundary critic各执行同50个task3 reset的121真后继贪心、1000步，raw0/50、42/50复用；dynamics不参与该控制，故每critic只跑一次。完整轨迹保留，不用理论缩放不变性代替FP32实际执行。
7. 梯度/坐标smoke在正式扫描前完成并丢弃，正式过程只读且不更新参数/优化器；保存原输入SHA与actual transformed参数摘要。全部候选、约束、目标和执行含失败均报告。较好的可行候选若原目标更优，只证明这些有限点/固定查询审计下有改进空间；若更差，也不证明坏场是全局最优。目标相近且控制不同则支持目标对关键次序分辨不足。各环境内部比较，不把F0/F1/P/P+D数值跨环境混合。

只提交源码、测试、协议、报告和引用图；全量距离/D缓存、查询表、参数见证、轨迹、审计与日志留本地。
