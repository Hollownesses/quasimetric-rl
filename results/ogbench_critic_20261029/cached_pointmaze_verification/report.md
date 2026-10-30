# 固定模型的精确评价缓存：全25例逐位复现

本项解决后续诊断的计算开销，不是新的学习方法或成功率改善。没有训练、换seed或运行1M；原QRL 100k的同25例贪心仍为16/25，task3仍0/5。

## 先前两个阶段未通过完整门槛

| 实现阶段 | 结束状态 | 用时 | 已完成完整episode |
|---|---|---:|---:|
| 单线程原生物理批量 | 预定120秒超时，未启用 | 120.01秒 | 16/25 |
| 四线程原生物理批量 | 预定120秒超时，未启用 | 120.01秒 | 19/25 |
| 四线程物理批量＋精确评分缓存 | 全部核验通过 | 76.76秒 | 25/25 |

前两阶段的35组候选都逐位一致，已完成episode亦一致，但未完成全量门槛，不能当通过；failure.json、未完成进度、候选数组、原源码快照和日志全部保留。没有延长或恢复这两个进程。第三阶段显式新增固定神经评分缓存，在全新目录单独核验；三个阶段计算合计约316.78秒，低于预定合计360秒。前两阶段在120秒截止检查后约10毫秒内记录异常并退出。

## 原物理、原模型及原episode语义

物理入口只支持固定官方pointmaze-medium状态环境。按PointEnv原语义，对候选qpos加0.2*a、速度清零，再执行5个MuJoCo物理步。原生Rollout使用四个独立MjData；FULLPHYSICS、完整USER输入与qacc_warmstart全部复制，尺寸之和等于INTEGRATION的23个数。不支持的teleport/模型或观测类型拒绝，未简化接触或丢掉warmstart。

episode直接复用原core.rollout同一字节码，在独立globals中绑定候选实现，原core模块不修改。每个被选动作仍真实env.step，后继与候选逐位核对；保留成功终止和完整1000步上限，不按停滞提前结束。

固定模型的评分缓存仅适用于encoder=None、JAX x64关闭的状态输入。键包含完整形状和完整观测/目标在原jit实际接收的float32表示中的字节；原函数本来就将float64输入转换到float32，因此没有额外近似或人为舍入。cache miss仍直接调用原common.distances与原始输入；每个键第一次hit也重新调用原函数逐位比较。返回数组复制，LRU容量2048只影响命中率，不改变数值。每个不可变agent单独创建缓存，训练中不能替换agent或复用旧缓存。

## 已完成核验范围

- 25个bank起点及E6 mean/tail十个task3结束integration state，共35×121个候选，原逐个恢复实现与四线程原生后继逐位相同；候选前后完整环境积分状态和episode计数不变。
- 第三阶段核对物理实现与上一阶段源码摘要相同，复用其35组原始候选数组，对原评分与缓存逐位核验。
- 全部25个原始case重新执行完整episode，逐步observations、actions、goal、distances与已保存baseline_greedy逐位相同，成功与步数相同；不存在仅挑成功case或仅查终点的缩小验证。
- 两项缓存方法测试通过：实际float32相同但float64不同的输入命中/首次复算、有效输入变化时miss、形状/目标隔离、输出数组所有权及LRU淘汰。
- 实际模型摘要前后不变；源码快照与当前文件SHA均核对通过。bank SHA为5bb143317adf5c9198d1fd12ee3263a57236973719bbb036ff4dffe55525a95b，原模型仍为固定SHA 397e919f2c453cee02ec840c615dce3532f31532fd639fbea6757c970fedbf0f。

## 实际开销及解释范围

25个episode共10889次评分请求。原函数实际调用5125次，已包含首次命中重算；episode内缓存命中5773次。全部episode用时74.30秒，连候选/载入/核验共76.76秒。原保存基线episode计时合计207.83秒仅作历史参考，硬件负载和热状态不同，不能把二者比值当严格速度基准。收益主要适用于重复有效输入的失败/接触状态，移动中的成功episode可能没有命中。

通过只支持本固定状态任务/依赖/设备与控制语义，不能推广到图像、其他MuJoCo任务或任意有随机性的评分器。后续新critic必须创建自己的冻结模型缓存、保留首次hit复算及选中真实后继核验；该工具不能代替新模型的真实执行证据。

## 复现与后续研究

协议依次为[FAST_POINTMAZE_PROTOCOL.md](../../../experiments/ogbench_diagnostics/FAST_POINTMAZE_PROTOCOL.md)、[PARALLEL_POINTMAZE_PROTOCOL.md](../../../experiments/ogbench_diagnostics/PARALLEL_POINTMAZE_PROTOCOL.md)、[CACHED_SCORE_PROTOCOL.md](../../../experiments/ogbench_diagnostics/CACHED_SCORE_PROTOCOL.md)。入口与前置产物见[诊断说明](../../../experiments/ogbench_diagnostics/README.md)。测试为test_exact_score_cache.py；全量核验为verify_cached_pointmaze.py，必须使用全新输出目录。

原始数组/JSON/日志保留在三个verification目录，Git只提交源码、测试、协议和本报告。下一项仍需原神经critic的受控改进；可复用E6已经验证的1000步mean对照及其批次流，不必再训练相同控制组。既有精确重复审计比例很低，不能把简单精确去重当作已获支持的改进。当前稳定改善目标未完成，不恢复seed1或1M。
