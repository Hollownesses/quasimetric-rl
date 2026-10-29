# 原生MQE距离诊断的等价性与烟雾验证

2026-10-29。已建立原生MQE的同库V贪心及距离场入口，并验证其物理/图工具与既有QRL诊断一致。该项是评价工具核验，不是100k学习结果；当前MQE、E2和官方基线训练均继续按原预算运行。

## 避免算法模块混用

MQE与QRL使用同名agents/utils包。原生MQE评价进程只加载固定MQE实现与同一官方OGBench环境，由MQE原生psi和MRN distance给出状态V距离。模型无关的图与模拟器工具位于experiments/ogbench_diagnostics，避免导入QRL训练运行时来给MQE评分。

几何、精确线段碰撞、建图、全部路线最低峰值/下降通道、完整状态恢复和候选动作枚举共12个函数从E0/E1直接复制，原文件在E2运行中保持不变。测试逐项比较AST相等；另用小型官方QRL、固定墙角接触状态，核对两套工具的候选真实后继及完整greedy/actor episode的观测、动作和距离数组逐位相等。不是只检查末端成功率。

## 已完成的实际验证

| 检查 | 实际结果 |
|---|---|
| 模型无关核心 | 2项测试通过，含12函数AST一致与接触状态完整轨迹相等 |
| 原生MQE推理 | 3项测试通过：V等于原生action-free distance、固定max/两分量及NaN拒绝、非声明终点拒绝 |
| 真实两步checkpoint | params_2.pkl，SHA256 a065dd909fc2ac46b98ae422176e157f22bbfb1266365e6bf5694a701b73032b |
| controller烟雾 | 固定task3_seed3100，两ensemble、max、原生actor各完成1000步；所有逐步轨迹有限且完整 |
| 真实后继一致性 | 每步121个动作恢复完整mjData/TimeLimit，所选动作实际执行与枚举后继逐位相同 |
| 三套距离场 | 两分量及max逐点关系正确；与原QRL相同7091节点、53950有向边的碰撞图 |
| 参考路径重放 | 三套field均为图参考60步、真实61步成功，跟踪误差小于1e-7 |
| 后处理保护 | 4项故障注入测试通过，覆盖失败/缺失终点、PID复用及launcher完成竞态 |

共9项测试通过，另有真实两步模型的完整数值烟雾验证。各controller在这一单例两步smoke均失败，只用于I/O验证，不是训练表现或正式成功率。没有加载正式运行的中间模型来选择controller或聚合方式。

## 固定终点与主要比较方式

训练前主协议已经指定max距离为主要保守聚合；两ensemble分开列出，用于判断差异，不能择优宣称收益。正式终点仍为100k，同既有evaluation_bank.npz的25个完整integration state/目标，五任务各5例，另报告官方actor各50episode结果。V真实后继贪心与actor分开解释；不以原生Q动作搜索代替主要V通道问题。

下降容差按-log(0.995)单位转换，峰值差用IQR归一化；保留参考图的离散化与锚点吸附限制。正式评价写入实际epoch、checkpoint/参数摘要、bank/动作网格及源码hash；校验轨迹长度/有限性、各任务计数、同碰撞图、原生分量与max关系。中间模型被入口拒绝；两步模型必须显式smoke。

## 已挂接现有训练，正式分析仍待完成

finish_mqe实际绑定训练PID41218及launcher41216的启动时间/完整命令，exec session38426、后处理PID43726。当前状态为waiting_for_fixed_endpoint。等待原运行正常完成100k后，才执行四个controller的全25例与三套field，并读取原shell job负责的checkpoint全状态往返结果；不会重复训练或重复创建模型核验目录。

数值产物完成后写DIAGNOSTICS_READY，仅表示可供研究分析，正式比较报告、曲线与图、配对得失解释及实验commit仍需完成。稳定改善、独立seed与官方完整预算、改善后actor提取等总目标均尚未证明。

本次原始烟雾产物在results/ogbench_critic_20261029/mqe_diagnostics_smoke，校验在同目录verification.json，测试日志分别为tests_diagnostic_core.log、tests_mqe_diagnostics.log、tests_finish_mqe.log；实际挂接记录在postprocess_mqe。原始产物不进入Git。

方法、命令及边界见[诊断协议](../../../experiments/ogbench_mqe/DIAGNOSTICS_PROTOCOL.md)与[MQE README](../../../experiments/ogbench_mqe/README.md)。本项没有需要引用的结果图，源码、测试、说明、协议及本报告单独提交。
