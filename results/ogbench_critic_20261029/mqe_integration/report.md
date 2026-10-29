# 原生 MQE 接入、环境一致性与两步烟雾测试

2026-10-29。原生MQE已经接入同一公共基准，完成两次真实参数更新、全五任务完整episode评价和完整checkpoint保存加载。正式seed0、100k方法对照已经启动，尚无终点学习结论。本项不代表已达到稳定改善，也不代替E2的一步／多步因果对照。

## 固定实现与来源核验

MQE固定commit为 `04c2697405233e4313fab43e05bd4d95f1b3e111`，直接调用其 `impls/main.py`、`impls/agents/mqe.py` 和原生 `utils/datasets.py`。双MRN、完整B×B LINEX备份、动作不变性、waypoint/目标采样、actor联合优化与官方评价器均保留。没有使用E2的IQE表示、批内16目标估计或冻结actor。

Git clone首次发生连接重置，未获得完整checkout。实际通过GitHub连接器读取固定tree和源码，以Git blob SHA1验证90个运行相关源码文件；其中68个与现有官方OGBench逐字节相同而复用，22个来自固定MQE ref。所有90个核验通过，锁文件见 [SOURCE_BLOBS](../../../experiments/ogbench_mqe/SOURCE_BLOBS)。没有下载无关任务二进制资源，不声称已安装全仓库所有环境；标准clone同commit或bootstrap均需通过相同文件核验。

## 环境差异是本次最重要的接入发现

MQE的requirements建议外部OGBench，但其仓库自带maze在step中加入按动作幅值1%标准差的高斯噪声，并在构造时强制渲染。直接把该fork加入Python路径，会改变现有QRL的评价环境，无法把差异仅归于算法。[固定源码中的maze](https://github.com/WJ2003B/mqe-release/blob/04c2697405233e4313fab43e05bd4d95f1b3e111/ogbench/locomaze/maze.py#L433)

实际适配层只加入MQE的impls路径，环境包显式绑定原QRL官方commit `a977253ce5b88434f4e4f531c50047cc95993a51`，加载同一原始训练/验证数据。实际模块来源已经断言并记录，算法源码均未修改。这是透明的环境依赖选择，不能与MQE自带噪声fork的结果混称相同协议。

首次两步smoke在创建环境前失败：上游main硬编码Linux EGL，macOS MuJoCo导入报 `invalid value for environment variable MUJOCO_GL: egl`，退出码1、约2.18秒。失败目录 `results/ogbench_critic_20261029/mqe_smoke` 完整保留。适配层改为先使用本机CGL导入MuJoCo，再执行未修改的main，使用全新目录 `mqe_smoke_cgl` 重试成功。没有更改动作、动力学、成功检测或采样。

## 已通过的实际验证

| 核验项 | 证据 |
|---|---|
| 原生完整batch更新 | batch256，两次；真实Adam count=2，TrainState.step=3 |
| checkpoint完整性 | 参数与优化器等全部状态有限；恢复后再保存加载，PyTree逐叶完全相同 |
| actor推理往返 | 同25条输入、相同key及temperature0，动作逐位相同且范围合法 |
| 双critic推理 | V和Q距离均为2×25，全部有限 |
| 原生数据边界 | 256条固定源，其中128条末端零waypoint偏移；目标/waypoint未跨官方轨迹边界，真实next索引相等 |
| 数据规模 | 原始compact train 1,001,000状态、1,000,000有效源；validation 100,100状态 |
| 完整数值评价 | 第1/2步均完成全五任务、每任务1个完整episode，无视频 |
| 设备与耗时 | TFRT_CPU_0；成功smoke主进程约9.78秒 |

两步模型五任务均0/1，只说明数值评价流程可执行，不是算法学习表现或正式成功率。没有据此调参或挑checkpoint。

独立CPU环境依赖未安装、升级或改写：Python3.10.19、JAX/JAXlib0.4.35、Flax0.10.2、Optax0.2.4、NumPy1.26.4、SciPy1.14.1、Distrax0.1.5、MuJoCo3.17.0、Gymnasium1.4.0、ml_collections1.1.0、W&B0.18.7；全部pip freeze保留在smoke与正式目录。本机未使用CUDA。

原始核验产物在 `mqe_smoke_cgl/checkpoint_check.log`、`checkpoint_verification.json`、`runtime.json`；checkpoint为 `mqe_smoke_cgl/OGBench/mqe_navigate_smoke/sd000_20261029_220610/params_2.pkl`，SHA256 `a065dd909fc2ac46b98ae422176e157f22bbfb1266365e6bf5694a701b73032b`。这些原始产物不提交Git。

## 正式对照已冻结并启动，结果待定

正式路径 `results/ogbench_critic_20261029/mqe_100k_seed0`；exec session79677，launcher PID41216、训练PID41218（仅本次观察，恢复时重验身份）。已确认进入官方第1步全五任务各50episode评价，进程存活；未从两步smoke恢复。

固定seed0、从头100k更新、batch256；每1000步日志，每10k步全五任务各50episode评价与保存。完整配置、依赖版本及运行源码快照均已保存并核对，只有100k终点用于正式结论。相比原QRL每步batch1024，MQE源转移为25.6M而非102.4M，需同时报告更新数、源转移数和实际耗时，不能称同计算预算。

alpha=0.03采用论文pointmaze表3；其余网络与采样配置与表2一致。diag_backup=0.5明确使用代码默认，论文未列该任务取值，代码注释的locomotion建议区间也没有逐任务命令，故不把本运行称为完全确定的论文超参数复现。[论文v1附录B.3/B.4](https://arxiv.org/html/2511.07730v1#A2.SS3)

终点后仍需同固定25例真实后继V贪心及距离场比较，双ensemble分别检查，并按预先声明的max距离聚合。actor与距离场分开解释；不以actor成功替代critic下降通道证据。独立采样/初始化、官方QRL完整预算及改善后actor提取仍未完成，总研究目标保持进行中。

安装/复现和完整1M命令见 [MQE README](../../../experiments/ogbench_mqe/README.md)，选择与解释边界见 [固定协议](../../../experiments/ogbench_mqe/PROTOCOL.md)。仅接入源码、锁文件、说明、协议和本报告进入Git；本轮没有需要引用的图。
