# OGBench critic 机制实验

当前研究主线与每轮决策见 `WORKLOG.md`，预注册 A/B 设计见 `PROTOCOL.md`。模型、原始数据与 checkpoint 不进入 Git；报告及其引用图片单独提交。

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
# 两组只评价110k终点；不得根据105k中间模型选最好。
for ARM in pair goal; do
  $P "$E/evaluate.py" --bank "$R/evaluation_bank.npz" --output "$R/${ARM}_greedy" --checkpoint "$R/paired_10k/$ARM" --epoch 110000
  $P "$E/field.py" --output "$R/${ARM}_field" --checkpoint "$R/paired_10k/$ARM" --epoch 110000
done
```

`training_complete.json` 记录固定更新数、冻结模块参数/动量哈希和共享批次流哈希。CSV 同时记录两类约束，不能只比较各自被优化的统计。图和greedy均是诊断工具，不属于官方 actor 评价或论文结果；所有5000/10000续训 checkpoint均保留，评价仅取10000终点。训练失败应保留目录与日志，使用新结果路径修复，不覆盖、缩水或换任务。

`field.py` 读取真实墙体 box 与 pointbody 半径，对线段到矩形的最小距离进行分段二次精确求解，不以端点无碰撞假设整条边无碰撞。参考图位置离散化（默认0.2），每条边是一次每轴不超过0.2的合法动作。最小峰值搜索涵盖全部图路线；严格下降判断独立于统一缩放与平移。网格锚点有吸附误差，报告明确列出；它不是连续环境最优性的证明。

`evaluate.py` 保存每个 episode 逐步轨迹。候选动作评价前用 mj_copyData 恢复完整 mjData，并恢复 TimeLimit 计数；最终选择的候选被真实执行，要求后继状态与枚举结果逐位一致。固定25个初始 integration state/目标，覆盖全部五任务，每任务5个扰动。报告小样本计数，不把5/5称统计稳定。actor/dynamics冻结意味着本轮不能改善原actor闭环；若critic方向有效，还必须独立检验策略提取。
