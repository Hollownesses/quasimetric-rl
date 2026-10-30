# E2：同表示的一步 / 多步备份对照

根据E1固定终点失败而制定，2026-10-29。E1不再搜索权重。E2检验明确的目标相关备份，而不是继续替换单侧上界残差。

## 方法与控制

两组从相同原始100k状态IQE表示出发，actor/dynamics原参数与Adam矩冻结。新增状态动作编码器phi，状态表示psi沿用原value编码器；phi第一层的状态权重复制psi，动作权重为零，其余层完全复制，初始phi(s,a)=psi(s)。IQE及其mixing alpha沿用原定义，不改成MRN。

gamma=0.995，距离单位改为log-discount单位。将psi/phi最后一层的kernel与bias统一乘以c=-log(gamma)，理论上使初始距离为c倍原距离、排序保持，数值误差另行核验。该确定性换单位不使用参考路线或成功率标定。新critic优化器全部从零矩开始，两组一致；旧value Adam不用于新目标。实际新优化计数从0起，原actor/dynamics状态不动。

critic借鉴MQE的备份与动作不变性：

- d_Q=d_IQE(phi(s,a),psi(g))；d_next=d_IQE(psi(w),psi(g))，后者stop-gradient。
- delta=d_Q-stop(d_next)。LINEX备份取 gamma^k exp(delta)-d_Q，delta>5时用官方MQE同样的线性分支delta；不把行为轨迹长度当最优距离等式。
- 动作不变性项 mean((exp(-d_IQE(psi(s),phi(s,a)))-1)^2)，推动状态值对数据内动作保持乐观界。
- 不保留原独立距离最大化或pair/lambda约束。E2与E1有多个变量差异，不能据二者差异把收益只归于多步；因果比较仅限E2两组。

两组唯一差异：one组恒取w=s_next、k=1；multi组20%取一步，其余使用Geom(1-0.95)的未来waypoint，截到本轨迹的future goal索引。future goal由Geom(1-gamma)采样并截到官方GCDataset末端；保留compact valids与双terminal边界语义，包括末端offset=0的情况。一歩混合分支仍使用真实next_observation，不能跨轨迹。

## 固定预算与批内目标

各额外10,000更新，seed4501，batch256，lr3e-4，LINEX阈值5、diag_backup=0.5（MQE代码默认，非声称pointmaze调优值）。两组逐次共享转移、future goal、waypoint及批内目标索引。actor不训练，故不使用MQE论文的actor alpha=0.03；原actor仅作冻结对象。

完整批内B×B的IQE排序开销较高，固定每个源状态抽取16个独立均匀批内目标（可含自己），另保留自身future goal。损失为0.5*自身目标均值 + 0.5*抽样目标均值。这是固定MQE对角/全配对加权目标的无偏Monte Carlo估计，两组相同；不依据结果调整16这个数量。batch256与E1的1024不同，分别报告更新与源转移数，不声称同计算预算。

每500更新记录loss、动作不变性、waypoint长度及固定held-out随机目标诊断（seed9001，2048条）；保存初始化供数值核验，每5k保存完整新critic/优化器/RNG。只评价10k终点。两步smoke通过后才启动正式运行；不按成功率停止、延长、改权重或挑中间模型。

新格式为critic_N.pkl，N是新目标的实际优化次数。评价时只在内存中构造原agent的value推理视图，旧agent优化器不具有E2续训含义，也不另存成可混淆的官方checkpoint。续训状态只来自完整新critic bundle。下降判定的数值容差乘以c以保持单位一致；旧官方评价默认容差不变。

## 评价与解释

沿用E1固定25例真实后继贪心与参考图，另增加对应尺度转换后的pair/goal残差。图与控制只用于评价，不能进入训练。初始缩放表示的排序需核验；最终actor/dynamics冻结、实际新优化计数、保存加载和actor输出相等需核验。

multi优于one且恢复通道/执行，支持多步传播作用；两者都恢复则不能把收益归于多步长度，需进一步分离备份目标、动作不变性与表示变化；两者失败则不以调权重挽救本试验。任何正结果需独立采样/初始化、目标扰动与其他任务复核，后续再检验actor提取。

这称为“MQE-style IQE critic消融”，不是官方MQE复现。原生MQE有MRN、双critic和完整actor联合学习等差异；仍需纳入原生方法比较，见MQE_REVIEW.md。多步本身不作为新贡献。
