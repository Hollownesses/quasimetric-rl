# E18：E6 mean接同一E17提取流程

2026-11-01，按用户要求补齐普通价值续训后的策略提取控制。不重跑E6、E17-A/B、其他训练seed或官方1M，不追加critic学习。

## 固定起点与区别

唯一新增组C加载E6 mean完整终点 `tail_1000/mean/params_101000.pkl`，SHA256 `d92cf664f10345f96a77045f988a689f0985ac2120781ec596ef6b932fdaacec`。它从原seed0 100k经官方value_loss额外1000步而来，value及lam/对应Adam矩已更新；actor/dynamics及矩、agent RNG应仍同原100k，运行前核验。原Adam/count101000完整保留，禁止只移植value或重置乘子。

E17-A原critic、E17-B E16critic的训练和评价结果直接复用。C与B均有1000次价值学习及随后相同提取更新，但前期价值目标、查询分布、教师/额外前向开销和lam处理不同，不能称等总FLOP或单独某个损失项的消融。C的lam不同于B，本次完全冻结；官方dynamics_loss/actor_loss不读取lam，新增小网络损失/活动梯度测试核验这一点，不声称两个起始完整agent相同。

## 与E17完全相同的提取设置

- 先仅官方dynamics_loss更新1000次，再仅官方actor_loss(ddpgbc)更新1000次。value/lam及其Adam矩全程冻结，actor阶段动力学及其矩也冻结。直接复用extraction.py和evaluate_extraction.py，不修改历史E17源码。
- batch256、lr0.0003、alpha0.0003、512×3网络、const_std、官方100%未来actor目标等保持E17配置。数据仍官方完整GCDataset，不引入图/教师/真实后继动作搜索标签。
- 正式开始仅一次np.seed2001，动力学→actor连续2000批、阶段不重置。分别核验两个阶段(s,s_next,value_goal)及含actions/actor_goals完整流SHA与E17逐位一致；验证2048批seed9001及hash亦同E17。评价seed仅原rng0匹配样本，不换训练初始化。
- 实际smoke各阶段2步并完整保存加载，然后丢弃smoke状态，从E6 mean原状态/采样2001正式运行。每100步记录训练/同一验证损失，不做中途成功率评价；只保存dynamics102000、actor103000两个固定端点。核验冻结参数/矩、有限性、count/step、critic验证输出逐位不变、完整agent往返以及输入文件未变。
- 总上限600秒，含载入、smoke、训练和完整评价，等同E17上限而非强制耗满。超时/失败保留日志与阶段产物，不自动缩数据/episode、重启、扩预算或换算法。

## 评价与交付

独立进程原evaluate_extraction.py直接调用固定官方actor/evaluate，五任务各50完整episode，temperature0、gaussian=None、video0，另跑原25-case actor bank。没有新扰动seed，不重复critic贪心诊断。全部250起点/目标须逐位匹配E17-A/B，所有原25case的起点/goal及bank SHA亦匹配；保留275完整轨迹和结束integration。

逐任务及平均报告A/B/C三列，分别报告C相对A、B相对C的配对新增/失去（包括其他四任务）。本项在看到E17结果后补设，属于事后追加的控制，不冒充原预注册第三组或独立验证。无论C表现如何，不改终点/参数或续训。它检验原1000次价值续训接相同提取能否解释E17-B收益，并保留前期训练问题/lam/计算先验的差异；不做跨seed/论文预算结论。

全新raw目录 `results/ogbench_critic_20261029/mean_extraction_1000`，日志同级.log；协议、源码、测试、报告、引用图片提交exp/ogbench，模型/JSON/CSV/NPZ/日志保留本地不提交。
