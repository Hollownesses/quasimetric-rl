# 原生 MQE 对照入口

先阅读[固定协议](PROTOCOL.md)。调用固定官方MQE的main/agent/sampler，环境与原QRL均来自同一官方OGBench。MQE仓库自带环境有额外动作噪声和强制渲染，不纳入本对照。保留全部算法实现，只适配环境依赖选择、本地路径、离线日志及macOS渲染绑定。

前置条件：按 `../ogbench_official_qrl/README.md` 准备独立CPU JAX环境、固定官方OGBench与原始数据。MQE复用该依赖环境，不执行pip升级或修改包；不使用原PyTorch实验环境。

仓库根目录执行，结果根目录必须全新：

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_mqe
R=results/mqe_new_run
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
# 官方MQE的90个运行相关源码文件，每个都核对固定Git blob。
# 首选复用原OGBench中逐字节相同的文件，其余从固定官方ref下载。
$P "$E/bootstrap.py"
$P "$E/bootstrap.py" --verify-only
$P "$E/protocol.py" smoke "$R/smoke"
$P "$E/launch.py" "$R/smoke"
$P "$E/check_checkpoint.py" "$R/smoke" 2
# smoke通过后独立从头启动固定100k；不会恢复两步模型。
$P "$E/protocol.py" 100k "$R/main"
$P "$E/launch.py" "$R/main"
$P "$E/check_checkpoint.py" "$R/main" 100000
```

若在受限沙箱内启动，离线W&B仍需本机服务通信；不能因该错误更改学习目标。macOS的适配层会先用CGL导入MuJoCo，处理上游main无条件设置Linux EGL的兼容问题；数值评价不创建renderer。

每次冻结输出完整配置、argv、源码摘要、依赖版本和实际设备；launch使用独占LAUNCHED标记，不覆盖已有目录。模型、日志与CSV/JSON/轨迹留在结果目录，Git只提交入口、源码锁、说明、报告及引用图。

`alpha=0.03`来自论文pointmaze表3；`diag_backup=0.5`明确采用代码默认，论文未给出该任务取值，不能宣称完全复现其所有超参数。其他方法配置保持默认，不按成功率选终点或调参。100k使用每10k全五任务各50episode评价、保存；正式结果只取100k。

完整1M预算命令（单独目录，不能与100k目录混用）：

```bash
$P "$E/protocol.py" full "$R/full"
$P "$E/launch.py" "$R/full"
$P "$E/check_checkpoint.py" "$R/full" 1000000
```

该命令不等于完整论文复现：原文还使用末三次评价和多seed汇总。当前研究主要关注100k模型的距离恢复；原生MQE是已有方法参照，E2才是相同IQE表示的一步／多步因果对照。MQE终点仍需追加同库V贪心和距离场比较；没有这些结果时不能仅靠actor成功率断言critic机制得到解决。

## 同库V贪心与距离场

详见[终点诊断协议](DIAGNOSTICS_PROTOCOL.md)。通用物理/图模块与原E0/E1工具作等价验证，MQE进程只加载原生算法；正式报告以max距离聚合为主要结果，两ensemble另列，不能择优。

```bash
G=results/ogbench_critic_20261029
$P experiments/ogbench_diagnostics/test_core.py
$P "$E/test_diagnostics.py"
$P "$E/test_finish_mqe.py"
# 原两步smoke checkpoint，仅作I/O验证，路径取实际保存位置。
$P "$E/diagnostics.py" --checkpoint "$R/smoke/OGBench/mqe_navigate_smoke/<实际运行目录>" --epoch 2 --bank "$G/evaluation_bank.npz" --output "$R/diagnostics_smoke" --smoke
$P "$E/verify_diagnostics.py" "$R/diagnostics_smoke" --bank "$G/evaluation_bank.npz" --baseline-field "$G/baseline_field" --smoke
# 正式100k终点：将checkpoint路径换成实际保存目录；全部25例、两ensemble/max与actor都运行。
$P "$E/diagnostics.py" --checkpoint "$R/main/OGBench/mqe_navigate_100k/<实际运行目录>" --epoch 100000 --bank "$G/evaluation_bank.npz" --output "$R/endpoint_diagnostics"
$P "$E/verify_diagnostics.py" "$R/endpoint_diagnostics" --bank "$G/evaluation_bank.npz" --baseline-field "$G/baseline_field"
```

当前研究目录亦可在训练中挂接`finish_mqe.py "$G" --producer-pid <实际训练PID> --launcher-pid <实际launcherPID>`；它只消费`$G/mqe_100k_seed0`的固定终点，不重复训练或全状态保存加载。运行身份与源码已锁定，已有LAUNCHED标记禁止重复挂接。数值产物在`mqe_100k_diagnostics`，后处理日志在`postprocess_mqe`；DIAGNOSTICS_READY出现后仍须完成研究分析与报告。

## 既有100k终点的只读综合分析

协议见[ENDPOINT_ANALYSIS_PROTOCOL.md](ENDPOINT_ANALYSIS_PROTOCOL.md)。只读原QRL与MQE终点、已有轨迹及五个固定task3目标，不启动训练、换seed或新增episode：

```bash
$P "$E/analyze_endpoint.py" "$G" --output "$G/mqe_endpoint_analysis_new"
```

输出包括正式终点表、学习曲线、同库配对得失及实际目标的距离场；主要max聚合保持不变。若仅报告阶段中断，`--resume-analysis`可复用该输出目录已有场数组，核对目标/网格并以原模型固定节点前向检验缓存；不是恢复训练。当前研究预算遵守 `../ogbench_critic/CURRENT_LIMITS.md`，不运行QRL 1M或换seed复跑。
