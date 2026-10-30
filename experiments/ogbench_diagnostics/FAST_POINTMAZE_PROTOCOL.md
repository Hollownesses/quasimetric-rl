# 仅加速候选物理模拟的等价核验

2026-10-30，E6实际训练/核验165秒、训练加完整诊断582秒。为降低后续小预算实验成本，核验MuJoCo原生rollout批量执行121候选，不重训任何模型、不改动作网格、终止语义或1000步评价上限。

- 仅支持本固定官方pointmaze-medium状态环境、PointEnv的qpos+=0.2*a、qvel=0、mj_step(frame_skip=5)语义；拒绝teleport、其他观测/动力学类型或未覆盖的积分状态。
- 复制当前FULLPHYSICS，分别对121个候选预加位置动作、清零速度；完整USER输入与qacc_warmstart传给原生mujoco.rollout，每个候选精确执行5个物理步。FULLPHYSICS+USER+WARMSTART尺寸必须与INTEGRATION一致，插件/激活等不支持状态拒绝。原环境保持不变，实际选择的动作仍通过原env.step执行并核对真实后继。
- 复用原core.rollout的同一字节码，只在独立函数globals中绑定candidate函数，不修改原core模块或复制控制算法；所有episode步数、动作评分和argmin规则保持不变。
- 预定核验25个bank起点和E6 mean/tail的十个task3结束integration state，共35组候选。原逐个恢复模拟与批量模拟的121×2后继要求逐位相同，同时核验环境积分状态及episode计数未变。不接受浮点容差放宽。
- 通过后，用固定原100k QRL在全部25例重放快速版本，逐步观测、动作、距离与已保存baseline_greedy完全相同；成功/步数相同。旧baseline不重跑。核验负例拒绝teleport，不另换seed或训练。
- 本项总计算上限120秒，任一候选/轨迹不相等立即拒绝启用，保留差异/日志；超时不扩预算。若通过仅说明这个固定环境/库/模型范围的评价等价，后续新模型每个选中后继仍逐位核验，不能推广到其他MuJoCo任务或设备。

保存候选/轨迹的核验摘要、耗时与源码哈希。源码、协议、核验报告和必要测试提交；原始数组/JSON/日志留fast_pointmaze_verification全新目录。性能工具不代替稳定改善的研究目标。
