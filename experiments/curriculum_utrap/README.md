# 无专家动作的 U-trap 起点课程

已完成数据采集、真实动力学重放与训练入口验证，未启动正式 QRL 训练。主产物为 `results/curriculum_utrap_20261020/main`，详见其中的 `audit/report.md`。

本轮得到 27,399 条物理转移、505 个 episode，其中 151 成功、354 失败；额外有 151 条合法成功 anchor。所有失败与各阶段数据均保留。课程通过阶段 0–2，在阶段 3 上下东角连续六批未晋级后停止。固定诊断采样覆盖上下外走廊、西出口、U 内部，但这三个阶段均未获得成功轨迹，不能称为课程已经掌握。

## 采集规则与边界

复用 `minimal_qrl.dataset.collect_qrl_explore_episode_pair` 的持续随机动作：ω/ωmax 从 `[-1, -.5, 0, .5, 1]` 采样，直行概率 .5，其余各 .125，保持 3–10 步，局部安全前视关闭。动作不依赖起点、目标方向、专家动作、规划器、Oracle 或图参考。既有采集器生成的可选 suffix return 清零并关闭 mask；失败后独立采样的 anchor 丢弃。它们不参与动作、课程选择或训练。

起点使用目标条件、障碍几何与转弯半径 v/ωmax=1/3；不是目标盲探索。阶段按合法接近边界、接近区、东侧转弯空间、上下东角、上下外走廊、西出口、U 内部安排，逐步扩大 heading 至完整圆周。精确区域与 heading 范围见主目录 `config.json`，起点均为有效非终态。

正常批次 24 个前沿新起点＋8 个回访新起点。连续两批前沿成功率≥12.5%、至少2次成功、至少8个位置-heading格及2个heading格才晋级；回访不计入门槛。三批未晋级后改为16前沿＋16回访，六批未晋级停止；另有24批和30k真实交互上限。位置格大小 .25、heading 24格，只用于覆盖统计，绝不作为状态合并标准。

预先另分配阶段1–6各1000步诊断采样，不参与晋级、不会用它强行扩大课程。总上限36k（旧200k的18%），实际课程21,399步＋诊断6,000步；停滞后剩余8,601步不转移、不追加。全部协议及有效采集源码在首次交互前快照，只有一份正式采集；小预算单元测试另存临时目录，不用于挑选正式结果。

起点 RNG、回访 RNG、每 episode 动作 RNG 分离，seed=2026102007；505个起点与episode种子唯一。冻结测试pose只读取其位置用于排除，reset位置离所有清单pose超过.25，不论heading。真实沿途轨迹不会按测试邻域删边；报告另列沿途邻域命中，因此不把此数据当作完全隔离的测试泛化证据。

## 结果及训练语义

严格同 context 的成功记录路径支持 **967/27,399（3.53%）** 个 task 源；同任务完整观测精确连接的图也为967，未增加跨轨迹成功支持。成功episode比例29.90%并不等于受支持转移比例，成功轨迹只有1–69步，仍以近目标短链为主。`all` 与 `witnessed` **不等价**。

| U-trap 数据 | 物理转移 | 成功/episode | 严格见证源 | 位置-heading格 |
|---|---:|---:|---:|---:|
| 旧随机 replay 的 U 子集 | 50,000 | 2/188 | 370（0.74%） | 25,505 |
| 四路径 | 335 | 4/4 | 335（100%） | 191 |
| 专家网络 | 11,949 | 64/64 | 11,949（100%） | 801 |
| 本轮课程＋诊断 | 27,399 | 151/505 | 967（3.53%） | 12,840 |

这是描述性对照：数据规模、起点分布、控制策略与重复采样都不同。新数据比四路径/专家网络覆盖更广，但覆盖格不等于分支连接；它没有解决 U 内到目标的长程记录链支持问题。不能据此推断 QRL 优化已改善或课程在连续环境中最优。

`data.py` 直接读取完整 `verified_replay.pt`，校验文件、replay 与环境指纹，复用 production terminal quotient 与固定 seed42 B1。只提供正常转移、一步成本及合法终态；训练不读取 `audit/recorded_support.npz`。`witnessed` 只限制 task push，未见证的失败源仍参与 PHR 与同轨迹 B1。

全部27,399条转移已独立正向重放，原始状态、42维观测、动作和float32 reward逐位一致；同时重建起点、回访和动作 RNG，复算晋级与停止规则。合法anchor、失败无anchor、B1边界均通过。初次启动审计发现极近目标处 MPPI 向量成本近似的9.32e-5误差；核验改用环境标量成本（重建观测后的最大差9.42e-7），环境/MPPI/存储成本未改。旧加载器快照及失败审计日志保留，新旧校验器hash和原因见审计summary。

数据内容 SHA256：`bb55f1558d145b0530226926317705bac170376670aac632c4c2c02a8def8f28`。
PT 文件 SHA256：`5b63372c34034eeea2caa0bd615ccbd3d4f9e9aa01be361f0edc9a6a99bd940d`。

## 复现和重新审计

从仓库根目录执行；新输出目录必须不存在，防止覆盖历史产物：

```sh
export MPLCONFIGDIR=/tmp/qrl-curriculum-mpl
PY=../quasimetric-rl/.venv/bin/python
$PY -m experiments.curriculum_utrap.collect \
  --config results/curriculum_utrap_20261020/main/config.json \
  --output results/curriculum_utrap_reproduction
$PY -m experiments.curriculum_utrap.audit \
  --root results/curriculum_utrap_reproduction
```

重查已有数据，不重新采集：

```sh
MPLCONFIGDIR=/tmp/qrl-curriculum-mpl ../quasimetric-rl/.venv/bin/python \
  -m experiments.curriculum_utrap.audit \
  --root results/curriculum_utrap_20261020/main \
  --output results/curriculum_utrap_reaudit
```

固定 Python3.9.25、NumPy2.0.2、Torch2.8.0 的记录环境；跨平台浮点变化应复核数据指纹。文件封装hash可能受源码快照与序列化环境影响，replay内容hash用于内容复现。当前数据目录的原始manifest不因验证修正而重写。

## 后续 QRL 命令（本轮未执行）

以下保留 `full_replay_four_path_20261017` 的 `task_mode=all`，便于直接对照；仅切换数据来源及输出目录：

```sh
MPLCONFIGDIR=/tmp/qrl-curriculum-mpl ../quasimetric-rl/.venv/bin/python \
  -m experiments.full_replay_four_path.train \
  --curriculum-data results/curriculum_utrap_20261020/main \
  --output results/curriculum_utrap_qrl_20261020/all_seed606 \
  --seed 606 --threads 4 --batch 256 \
  --phr-steps 20000 --decay-steps 5000 \
  --audit-interval 1000 --task-mode all
```

如目标是限制 task push 到有记录成功链的967个源，将最后一项改为 `--task-mode witnessed`，并使用新输出目录 `results/curriculum_utrap_qrl_20261020/witnessed_seed606`。这是明确改变优化目标的对照，不能与 `all` 混称。训练器默认仍为 `all`。

两种模式均保持共享 production encoder/projector/IQE、AdamW无weight decay、20k×2e-5＋5k指数衰减2e-6→2e-7、rho100、每次访问dual eta1、task/B1系数.6/.4、固定B1 seed42、每1000步全量约束审计，不改MPPI。必要差异：本数据为单来源权重1、每epoch无放回采样全部成功/失败物理记录；旧实验为随机/四路径各.5，并且每步使用全部四路径。它们的边访问频次与分层目标不同，不是纯数据替换的因果实验。

恢复时沿用全部配置与输出目录并加 `--resume OUTPUT/latest.pt`。指纹、dataset kind、预算及task mask防止混用checkpoint。只有少量步数的临时单元测试用于验证梯度、失败掩码及逐位续训一致性，未产生正式课程模型。

```sh
MPLCONFIGDIR=/tmp/qrl-curriculum-mpl ../quasimetric-rl/.venv/bin/python -m pytest -q \
  experiments/curriculum_utrap/test_curriculum.py \
  experiments/curriculum_utrap/test_training.py \
  experiments/expert_utrap_network/test_training_adapter.py \
  experiments/full_replay_four_path/test_full_replay.py
```
