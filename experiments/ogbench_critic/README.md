# OGBench critic 机制实验

当前预算以[CURRENT_LIMITS.md](CURRENT_LIMITS.md)为准：2026-10-30用户已要求停止seed1与seed0完整预算，不再换seed重跑证明非偶然性，也不运行1M QRL。下文及旧协议相应内容仅为历史记录，不得自动启动。

当前研究主线与每轮决策见 `WORKLOG.md`，预注册 A/B 设计见 `PROTOCOL.md`。模型、原始数据与 checkpoint 不进入 Git；报告及其引用图片单独提交。

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

多步相关的官方源码及论文配置差异，见 `MQE_REVIEW.md`。E1固定终点没有改善task3执行，后续E2方案见 `MULTISTEP_PROTOCOL.md`；原生MQE已通过两步接入验证，固定100k方法对照进行中，入口见 `../ogbench_mqe/README.md`。

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
