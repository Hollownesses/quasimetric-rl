# OGBench critic 机制实验

当前交付见[中文汇总报告与逐项审计](../../results/ogbench_critic_20261029/research_summary/report.md)。同预算官方提取对照平均80%→92.4%，task3为0/50→31/50，其余四任务各50/50；同50个task3输入的神经critic真后继贪心0/50→42/50，仅为诊断，不代替actor成绩。残余失败、额外先验及同初始化的结论边界均保留。

不依赖历史控制组产物的正向配方见[独立复现说明](STANDALONE_REPRODUCTION.md)：`run_improvement.py --mode plan|smoke|improve --output <全新目录>`。本次只完成打包入口各阶段2步smoke（18.93秒、五完整官方episode）；正式性能来自已完成E16/E17，没有再跑1000×3来证明重复性。`finalize_delivery.py`只读审计已有文件并重建汇总图，不训练或运行episode。

当前预算以[CURRENT_LIMITS.md](CURRENT_LIMITS.md)为准：2026-10-30用户已要求停止seed1与seed0完整预算，不再换seed重跑证明非偶然性，也不运行1M QRL。下文及旧协议相应内容仅为历史记录，不得自动启动。

当前研究主线与每轮决策见 `WORKLOG.md`，预注册 A/B 设计见 `PROTOCOL.md`。模型、原始数据与 checkpoint 不进入 Git；报告及其引用图片单独提交。

## M6：同官方50例的critic/actor反事实

[OFFICIAL_COUNTERFACTUAL_PROTOCOL.md](OFFICIAL_COUNTERFACTUAL_PROTOCOL.md)只读复建E17评价流，250起点/goal和100个实际首步逐位核验后，比较原/改进critic在同50个task3初态的真后继贪心，并从全部19个actor失败末状态续行，不训练或新增seed。

原阶段240秒因振荡计算缓存miss超时，仅完成40例；[显式修订](OFFICIAL_COUNTERFACTUAL_COMPLETION.md)只补未执行案例，40例不重跑。补齐242.53秒/300秒，累计482.54秒，不能称原240秒协议通过或未扩计算预算。结果原0/50→改进42/50，末状态9/19可救援；剩余8个初态陷阱、2个actor进入的新陷阱，不能把残留全归于actor。

```bash
$P "$E/test_official_replay.py"
$P "$E/run_official_counterfactual.py" "$G" --output "$G/counterfactual_new"
# 当前补齐目录报告只读重建，不重复119个episode。
$P "$E/report_official_counterfactual.py" "$G"
```

`finish_official_counterfactual.py`仅复用特定已确认超时阶段的完整前缀，不是通用无条件重启命令。报告和两图见`official_counterfactual_completion/report.md`；原actor正式指标仍31/50，额外诊断不加进它的成功率。

## E17：冻结critic的官方策略提取对照

[EXTRACTION_PROTOCOL.md](EXTRACTION_PROTOCOL.md)两组分别固定原100k/E16 critic，使用完全相同原批次，先官方dynamics_loss1000步，再原ddpgbc actor_loss1000步；原alpha/lr/网络/100%未来actor目标不改。活动模块之外的参数和Adam矩冻结，原critic输出逐位不变，非官方联合训练预算。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_extraction.py"
$P "$E/run_extraction.py" "$G" --output "$G/extraction_new"
$P "$E/report_extraction.py" "$G" # 当前报告只读重建
```

三项测试、实际每组四步smoke/完整状态往返/冻结/原批次SHA核验通过，277.15秒完成。独立进程直接调用官方评价器，每任务50例，全部250起点/目标逐位匹配；A[50,50,0,50,50]、B[50,50,31,50,50]，平均80%→92.4%，31配对新增、无失去。原bank actor20/25→B23/25、task3 3/5，旧20全保留。原actor已能利用改进critic，但19个官方task3仍失败，需只读区分残留critic/提取缺口；不调BC/预算、不换seed/1M。报告与两图见`extraction_1000/report.md`。

## E16：固定八候选池的最坏全局拟合监督

[HARD_GOAL_PROTOCOL.md](HARD_GOAL_PROTOCOL.md)沿用原IQE/Adam/1000步与固定教师，只在半批同source上从8个均匀支持goal中选更新前双向MSE最大者；候选0逐位同E15。原MSE update不改，目标相当于半批原MSE加半批有限池max MSE，新增前向不当等FLOP对照。E12/E15控制不重跑，M5教师限制保留。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_hard_goal.py"
$P "$E/run_hard_goal.py" "$G" --output "$G/hard_goal_new"
$P "$E/report_hard_goal.py" "$G" # 当前报告只读重建
```

三项选择/梯度测试、实际smoke、全状态/冻结/count与原流/候选0流核验通过，297.95秒完成，低于360秒。独立无图神经推理首次25/25、task3 5/5（68/76/64/67/65步，每步距离下降），原成功全保留；固定验证池max MSE11.09→6.92。事后只读十点/五实际目标场5.90秒，旧两失败终点教师regret降到0，五目标通道均恢复。通过critic门槛，原actor仍冻结；当前bank用于方法选择，尚未达到整个研究目标。报告两图见`hard_goal_1000/report.md`。

## E15：全局教师监督的目标支持均衡

[GOAL_SUPPORT_PROTOCOL.md](GOAL_SUPPORT_PROTOCOL.md)只改变E12半批目标中心分布为6302实际数据原型的均匀互素遍历，source/幅度/比例/双向MSE/原IQE及Adam不变。每个原型20–21次，控制E12不重跑，验证保留原/旧扩展/新均衡扩展三组。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_goal_support.py"
$P "$E/run_goal_support.py" "$G" --output "$G/goal_support_new"
$P "$E/report_goal_support.py" "$G" # 当前报告只读重建
```

三项采样测试、实际smoke/完整往返/冻结/count和原流/旧查询SHA核验通过，141.78秒完成。21/25、task3 3/5，失去E12的两个task4成功，无新增；三组验证误差略降未换取执行收益。停止本均衡参数/预算搜索，不训练actor，现有较好神经结果仍23/25。报告及两图见`goal_support_1000/report.md`。

## M5：新增教师的查询可拟合性边界

[QUERY_REGULARITY_PROTOCOL.md](QUERY_REGULARITY_PROTOCOL.md)只读原100k/E13与E10教师，固定76点的对角极限及11个非对角查询响应，不训练/跑episode或重做覆盖审计。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_query_regularity.py"
$P "$E/query_regularity.py" "$G" --output "$G/query_regularity_new"
$P "$E/report_query_regularity.py" "$G" # 当前报告只读重建
```

两项测试通过，2.51秒完成。76点全有正对角极限0.115–10.638，实际/解析一致，说明抽象教师不具备连续神经距离的全域一致可拟合性；但11个固定非对角查询的最小扰动响应仅约1e−6，不能以对角反例解释实际远距离失败。现有神经改进仍23/25、task3 3/5；报告和图见`query_regularity/report.md`。

## E14：固定128地标的只读表示检查

[LANDMARK_PROFILE_PROTOCOL.md](LANDMARK_PROFILE_PROTOCOL.md)在固定E10数据教师上，以确定最远点遍历选128个地标，逐状态教师距离坐标作L∞伪度量；选择/保存完成后才读评价bank，不使用墙体或评价目标构造表示，无神经训练。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_landmark_profile.py"
$P "$E/run_landmark_profile.py" "$G" --output "$G/landmark_new"
# 当前报告只读重建，不重跑执行。
$P "$E/report_landmark_profile.py" "$G"
```

两项测试与直接教师/保存加载/源码/全部同bank轨迹核验通过，27.08秒完成。仅16/25、task3 3/5，失去其他任务成功；六个起点均有参考图通道但真实平台仍存在。事后全部九个失败终点，profile最优动作近零位移，而同候选教师仍有下降的0.2位移动作。未通过门槛，不训练该表示或扫描地标数/范数，报告及两图见`landmark_profile/report.md`。

## E13：有符号局部距离形状监督

[LOCAL_SHAPE_PROTOCOL.md](LOCAL_SHAPE_PROTOCOL.md)同E12查询/原IQE/Adam/教师，仅新增双向有符号教师距离差MSE，按固定0.2位移单位归一。允许正负差，不要求行为最优；扩展next同source平移，只作教师查询而非虚构真实转移。E12控制不重跑。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_local_shape.py"
$P "$E/run_local_shape.py" "$G" --output "$G/local_shape_new"
# 当前报告只读重建。
$P "$E/report_local_shape.py" "$G"
```

四项测试、smoke/count/完整往返与冻结核验通过，214.35秒完成；原采样、当前查询及当前标签三个流与E12逐位相同。仍23/25、task3 3/5，成功集合未再扩大。验证形状误差有所下降，两个完整执行仍失败；停止该形状缩放/系数/预算搜索，不启动actor训练。FP32累加次序造成首次梯度容差失败的原日志和测试保留，训练公式未改；报告与两图见`local_shape_1000/report.md`。

## E12：任务无关查询分布扩展

[QUERY_AUGMENT_PROTOCOL.md](QUERY_AUGMENT_PROTOCOL.md)与E11同原IQE/Adam/双向MSE/固定教师，只把半批完整状态/目标查询增加±1的确定性Sobol扰动，幅度来自官方环境噪声公式。无新seed、评价标签、几何过滤或虚构真实转移，E11控制不重训或重评。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_query_augmentation.py"
$P "$E/run_query_augmentation.py" "$G" --output "$G/query_augmentation_new"
# 当前报告只读重建。
$P "$E/report_query_augmentation.py" "$G"
```

两项采样器测试、真实smoke、原流SHA/冻结/count/完整往返与独立神经评价通过，133.81秒完成。23/25、task3 3/5，相对E11新增3104且无失去；canonical通道恢复仍不足以解决两个真实停滞。扩展验证RMSE1.507→1.329、原验证基本不变。停止本扩展幅度/比例/预算搜索，未启动actor训练或宣称总目标完成。报告及两图见`query_augmentation_1000/report.md`。

## E11：原IQE拟合固定教师，独立神经推理

[IQE_DISTILL_PROTOCOL.md](IQE_DISTILL_PROTOCOL.md)只更新原value，双向MSE拟合固定E10近似教师，原Adam矩/count保留，actor/dynamics/lam全部冻结。1000步、batch256、相同官方采样流，控制组E6不重跑。独立Python评价进程不导入或使用教师图，只使用原forward神经评分。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_iqe_distill.py"
$P "$E/run_iqe_distill.py" "$G" --output "$G/iqe_distill_new"
# 当前报告仅只读重建，不重训或重复评价。
$P "$E/report_iqe_distill.py" "$G"
```

三项测试、两步完整smoke与冻结/计数/往返核验通过，133.42秒完成。学生22/25，task3为2/5，原16成功无失去；教师25/25。随机验证RMSE26→1.1，但三个失败终点局部误差18.9–42.3且动作排序错误；事后十点探针无新增episode。最低峰差归零仍无canonical严格通道，未通过actor训练门槛，不追加本MSE预算。报告及两图在`iqe_distillation_1000/report.md`；不将神经部分恢复当官方actor或总目标完成。

## E10：先核验数据支持距离，再考虑神经拟合

[ROADMAP_METRIC_PROTOCOL.md](ROADMAP_METRIC_PROTOCOL.md)仅用全部官方valid转移与完整二维观测，选实际样本原型、双向几何代价和8连接的一致查询构造；全局三角结构与实际可执行性分开核验。局部可逆/几何代价是新增先验，非原官方步数真值，也不是M4单变量修补。原型/图/全部距离完成后才读bank，墙体与参考路线只用于末尾评价。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_roadmap_metric.py"
$P "$E/run_roadmap_metric.py" "$G" --output "$G/roadmap_new"
# 当前roadmap_metric报告重建，不重建图或重复评价。
$P "$E/report_roadmap_metric.py" "$G"
```

三项测试与全量距离/查询/轨迹核验通过，17.98秒完成，训练更新0。全部25例成功，task3五例65–75步且每步距离严格下降，原16个成功无失去；六个目标都有参考图下降通道。这只通过已知bank上的神经拟合候选门槛，尚未训练神经critic、改进actor或达到总目标。全距离float64约318MB留本地，不提交Git。

## E9：原100k的只读方向分解与固定对称投影

[SYMMETRY_PROTOCOL.md](SYMMETRY_PROTOCOL.md)固定原100k，主要critic仅为(d(s,g)+d(g,s))/2，不训练或搜索系数。正向/反向/对称场在canonical及五个实际task3目标分别检查；反对称分量只作有符号分解，不当距离。原F执行复用，S完成全部25例，actor/BC不改；不假定真实MuJoCo严格可逆。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_symmetry.py"
$P "$E/run_symmetry.py" "$G" --output "$G/symmetry_new"
# 当前symmetry_projection报告只读重建。
$P "$E/report_symmetry.py" "$G"
```

三项测试与模型/源码/数组核验通过，训练更新0，76.63秒低于240秒上限。同库16→19/25，新增三个task4成功、无失去，task3仍0/5；固定旧0/15上坡6.631中反对称贡献2.373，对称仍4.258。不能以不对称解释全部失败；对称pair风险也高于原F。不扫描系数或挑反向方向挽救，详见`symmetry_projection/report.md`及两图。

## E8：原IQE的真实多步路径上界

[PATH_BOUND_PROTOCOL.md](PATH_BOUND_PROTOCOL.md)保留原距离最大化/原IQE，仅把局部风险改为单步与额外真实路径上界违反的逐行最大值。固定2/4/8/16/32/64跨度，保留官方first terminal/valid/零长度语义，原批次与后续RNG不变；不将行为路径长度当最优距离等式。A复用E6 mean，仅新训练B1000步，actor/dynamics冻结。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_path_bound.py"
$P "$E/run_path_bound.py" "$G" --output "$G/path_bound_new"
# 当前已完成path_bound_1000报告重建，不训练或重复评价。
$P "$E/report_path_bound.py" "$G"
```

四项测试、真实两步smoke及count101000/全状态往返/冻结核验通过。主程序181.24秒完成，低于300秒上限；批次流SHA与E6相同。18/25对16/25，新增两个task4成功、无失去，task3仍0/5。路径风险大幅下降但归一化峰差仍正、无原锚点下降通道，不搜索跨度或其他超参数挽救。FP32首次诊断断言相差一ULP的失败日志/源码保留，最终数值标准见协议。

## E7：复用控制组的源格子均衡

[SOURCE_BALANCE_PROTOCOL.md](SOURCE_BALANCE_PROTOCOL.md)只改变原局部约束与push的经验源分布。用训练集完整二维观测的固定0.2格子频次给行权重，不删数据、不离散网络输入、不使用图边或参考路线。只新训练B1000步，直接复用E6 mean及其25例终点；训练数据流SHA必须相同，原Adam及actor/dynamics冻结方式相同，不重训控制组或换seed。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_source_balance.py"
# 前置：已有E6完整控制组与T1精确评价器verification，入口会核对哈希。
$P "$E/run_source_balance.py" "$G" --output "$G/source_balance_new"
# 当前已完成source_balance_1000报告重建，不训练或重复评价。
$P "$E/report_source_balance.py" "$G"
```

含实际两步smoke、B训练及全部25例/场共138.07秒，在300秒上限内完成；批次SHA与E6完全相同。两组均16/25、task3均0/5，B新增两个task4成功、失去两个task1成功。格子均衡的人口重要性有效样本数仅原N的约5.1%，不能称温和精确去重；该方案不调网格/权重截断等参数挽救。三项方法测试、全状态往返/冻结/终点身份及报告两图核验通过。

## E6：局部约束平均／尾部风险

[TAIL_PROTOCOL.md](TAIL_PROTOCOL.md)仅将原局部惩罚聚合从均值改为最差10%（26/256），同原100k、同批次，各额外1000更新，原Adam保留且actor/dynamics冻结。沿用E1采样seed2001，不是独立seed复现；主进程连训练与全部诊断上限600秒，超时不扩预算或把不完整结果充终点。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_tail_critic.py"
$P "$E/run_tail.py" "$G" --output "$G/tail_smoke_new" --smoke
# 两步完整状态/冻结验证通过后，从原100k开始，不能加载smoke。
$P "$E/run_tail.py" "$G" --output "$G/tail_main_new"
# 当前已完成tail_1000报告只读重建。
$P "$E/report_tail.py" "$G"
```

实际训练与状态核验165.01秒，完整预定工作581.75秒。原/mean/tail同库成功16/25、16/25、15/25，task3两组均0/5；tail新增两个task4成功、失去一个task2和两个task5成功。约束风险显著降低仍未恢复下降通道，不搜索尾部比例、eps或学习率挽救该固定方案。3项测试与两步实际网络smoke通过，见`tail_1000/report.md`及两张引用图。

## 数据转移图的限时非参数对照

[DATA_GRAPH_PROTOCOL.md](DATA_GRAPH_PROTOCOL.md)固定只使用官方训练集有效转移构建0.2粗粒化有向单位边图，不读验证集/墙体/参考路线来构造；原QRL和图距离各占一半，不修改原网络。完整预定诊断限360秒，未完成则停止，不能自动续跑扩预算。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_data_graph.py"
$P "$E/run_data_graph.py" "$G" --output "$G/data_graph_new"
# 当前已完成目录的post-hoc单步探针；另限30秒，无新完整episode。
$P "$E/probe_data_graph.py" "$G" --output "$G/data_graph_critic/local_probes_new"
# 当前已完成M4及local_probes_v2报告重建，不调用模型或环境。
$P "$E/report_data_graph.py" "$G"
```

当前M4耗时194.54秒，同25例成功16→19/25，仅新增三个task4成功，task3混合/纯图均0/5。两个实际目标在有限图中不可达，其余终点存在离散平台，不能把它当原QRL根因或神经训练真值；不通过平滑/权重/网格搜索挽救本次对照。源码与报告核验、两项方法测试通过，细节见`results/ogbench_critic_20261029/data_graph_critic/report.md`。

## 环境

复用官方基线的独立 CPU JAX 环境，官方源码 commit 固定为 a977253ce5b88434f4e4f531c50047cc95993a51。先按 `../ogbench_official_qrl/README.md` 安装，或将本工作树的 `.venv` 与 `data` 链接至同一机器已有的官方独立环境和只读数据。`runtime.py` 将导入固定本工作树 vendor，故不会误用旧 editable 安装。

脚本从本仓库根目录执行。需要已有完整基线产物：`results/ogbench_official_qrl_20261026_seed0/main/OGBench/navigate_100k/sd000_20261026_115628/{flags.json,params_100000.pkl}`。checkpoint 校验严格哈希，不能静默用别的模型替代。

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
R=results/ogbench_critic_new_run
mkdir -p "$R"
export PYTHONDONTWRITEBYTECODE=1
$P "$E/test_critic.py"
$P "$E/test_evaluation.py"
$P "$E/test_field.py"
$P "$E/train.py" --output "$R/smoke" --steps 2 --log-interval 1 --save-interval 2
$P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --make-bank
$P "$E/field.py" --output "$R/baseline_field"
$P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --output "$R/baseline_actor" --controller actor
$P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --output "$R/baseline_greedy"
# 预先确认协议后启动，默认各额外10k；配对批次来自同一采样调用。
$P "$E/train.py" --output "$R/paired_10k" --steps 10000 --seed 2001 --log-interval 500 --save-interval 5000
$P "$E/verify.py" "$R/paired_10k" --steps 10000 --bank "$R/evaluation_bank.npz"
# 两组只评价110k终点；不得根据105k中间模型选最好。
for ARM in pair goal; do
  $P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --output "$R/${ARM}_greedy" --checkpoint "$R/paired_10k/$ARM" --epoch 110000
  $P "$E/field.py" --output "$R/${ARM}_field" --checkpoint "$R/paired_10k/$ARM" --epoch 110000
done
$P "$E/compare.py" "$R"
```

`compare.py` 只接受两组完整10k终点、通过冻结验证的110k checkpoint、相同评价库与动作网格。评价产物记录实际加载的epoch、checkpoint SHA256和critic参数摘要；分析会逐项核对，避免将105k中间评价误充终点。训练未完成时分析入口会拒绝生成结论。它同时比较两类约束、尺度归一化距离场及五任务配对得失，最终解释还需结合逐步轨迹。

已有训练可用 `finish_e1.py` 接上终点后处理，参数为结果根目录及 `--producer-pid` 的实际训练PID。它以启动时间和命令核实进程身份，等待固定10k完成后依次执行verify、两组greedy/field和compare；不启动任何训练。进程退出却没有完整终点、PID复用、源码变化或任何子步骤失败都会保留日志并停止。故障注入测试为 `test_finish_e1.py`。`postprocess_e1/ANALYSIS_READY` 仅表示报告已生成，尚需研究者审阅、图形检查和实验commit，不代表总目标达成。

用于核验完整预算结论边界的独立官方运行，见 `FULL_BASELINE_PROTOCOL.md`；它不改变E1配对协议。

多步相关的官方源码及论文配置差异，见 `MQE_REVIEW.md`。E1固定终点没有改善task3执行，后续E2方案见 `MULTISTEP_PROTOCOL.md`；原生MQE固定100k、同库执行及实际目标场分析均已完成，未恢复task3，不重启该运行，入口见 `../ogbench_mqe/README.md`。

## E2：同表示的一步／多步备份

这是MQE-style IQE critic消融，不是原生MQE复现。两组同样改变了备份目标、增加状态动作编码器、换用log-discount单位并重置新critic优化器；只有E2内部one/multi对照能分离备份长度。原actor/dynamics参数及旧优化器矩均冻结。

```bash
$P "$E/test_multistep.py"
$P "$E/train_multistep.py" --output "$R/multistep_smoke" --steps 2 --log-interval 1 --save-interval 2
$P "$E/verify_multistep.py" "$R/multistep_smoke" --steps 2 --bank "$R/evaluation_bank.npz"
$P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --output "$R/multistep_smoke_actor" --checkpoint "$R/multistep_smoke/multi" --epoch 2 --checkpoint-kind multistep --controller actor
$P "$E/field.py" --output "$R/multistep_initial_field" --checkpoint "$R/multistep_smoke/one" --epoch 0 --checkpoint-kind multistep
# 上述检查通过、协议冻结后才启动正式配对训练。
$P "$E/train_multistep.py" --output "$R/multistep_10k" --steps 10000 --seed 4501 --log-interval 500 --save-interval 5000
# 也可在另一终端中挂接已运行的生产进程；训练结束后同样适用。
TRAIN_PID=$($P -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_pid"])' "$R/multistep_10k/protocol.json")
$P "$E/test_finish_e2.py"
$P "$E/finish_e2.py" "$R" --producer-pid "$TRAIN_PID"
```

`finish_e2.py`不启动训练：按生产manifest的原哈希保存源码/协议副本，核实训练PID身份，等待完整10k后依次执行verify_multistep、两组evaluate/field（`--checkpoint-kind multistep --epoch 10000`）及compare_multistep。已有LAUNCHED标记会阻止重复挂接；任一步失败须检查保留日志，不能删除标记后盲目重跑。`postprocess_e2/ANALYSIS_READY`只表示报告可供审阅，图片检查、研究解释及实验commit仍须完成。

E2的`critic_N.pkl`中N指新目标的真实优化次数，不能写成官方110k checkpoint。其bundle保存完整新critic、Adam和采样RNG；推理时在内存中替换原agent的value，旧agent优化器不能用于E2续训。`compare.py`/`finish_e1.py`只用于E1，不接受这种新格式。E2报告须同时核对单位转换、终点完整状态、同评价库/动作网格、两类残差及五任务配对得失；不能根据中间模型改预算。

## 原始模型的梯度机制检查

M1协议见 `MECHANISM_PROTOCOL.md`。固定原始100k模型、seed4400、32个官方采样批次，分别测量损失分量的一阶投影及实际一次Adam反事实；不查看E1中间模型，不保存新权重，也不用于修改E1预算。

```bash
$P "$E/test_mechanism.py"
$P "$E/mechanism.py" "$R" --output "$R/mechanism_gradient"
$P "$E/report_mechanism.py" "$R/mechanism_gradient"
```

`$R/baseline_field/field.npz` 必须来自前述固定原始模型的场复核。SGD分量投影与实际Adam变化分开解释；32批次不是32个模型复现。本项只解释原始模型附近的训练压力，不代替固定预算终点的距离场/执行检验。

`training_complete.json` 记录固定更新数、冻结模块参数/动量哈希和共享批次流哈希。CSV 同时记录两类约束，不能只比较各自被优化的统计。图和greedy均是诊断工具，不属于官方 actor 评价或论文结果；所有5000/10000续训 checkpoint均保留，评价仅取10000终点。训练失败应保留目录与日志，使用新结果路径修复，不覆盖、缩水或换任务。

`field.py` 读取真实墙体 box 与 pointbody 半径，对线段到矩形的最小距离进行分段二次精确求解，不以端点无碰撞假设整条边无碰撞。参考图位置离散化（默认0.2），每条边是一次每轴不超过0.2的合法动作。最小峰值搜索涵盖全部图路线；严格下降判断独立于统一缩放与平移。网格锚点有吸附误差，报告明确列出；它不是连续环境最优性的证明。

`evaluate.py` 保存每个 episode 逐步轨迹。候选动作评价前用 mj_copyData 恢复完整 mjData，并恢复 TimeLimit 计数；最终选择的候选被真实执行，要求后继状态与枚举结果逐位一致。固定25个初始 integration state/目标，覆盖全部五任务，每任务5个扰动。报告小样本计数，不把5/5称统计稳定。actor/dynamics冻结意味着本轮不能改善原actor闭环；若critic方向有效，还必须独立检验策略提取。
