# OGBench 官方 QRL 独立入口

本入口仅接入官方状态版 PointMaze QRL。与原仓库的 R1/R2、PHR、Dubins、MPPI、MultiEpisodeData 和终态 quotient 无关。固定官方版本见 `OFFICIAL_COMMIT`。官方源码放在被忽略的 `vendor/ogbench`，通过 setup.sh 固定下载；没有改写网络、损失、采样器、actor 或评价器。

## 安装

从本仓库根目录执行（路径有空格时请保留引号）：

```bash
export OGBENCH_PYTHON=/opt/homebrew/bin/python3.10
bash experiments/ogbench_official_qrl/setup.sh
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_official_qrl
```

独立虚拟环境 `$E/.venv`；官方数据 `$E/data`；不写入原实验环境。`requirements-lock.txt` 记录本次最终解析的全部版本。macOS arm64 用 CPU JAX 代替官方 requirements 的 CUDA12 extra。未安装 JAX Metal，不声称使用 Apple GPU。首次运行需要网络下载官方原始 train/val 文件。

## 烟雾、审计与 100k 试跑

每次使用全新结果路径，入口拒绝重复启动或覆盖冻结配置。

```bash
R=results/ogbench_qrl_new_run
$P "$E/protocol.py" navigate smoke "$R/smoke"
$P "$E/launch.py" "$R/smoke"
$P "$E/check_checkpoint.py" "$R/smoke" 1
$P "$E/audit.py" "$R/data_audit.json"
# 仅烟雾验证成功后：从头初始化，不加载烟雾权重。
$P "$E/protocol.py" navigate 100k "$R/main"
$P "$E/launch.py" "$R/main"
$P "$E/check_checkpoint.py" "$R/main" 100000
$P "$E/report.py" "$R"
```

`protocol.py` 导入官方 main 的全部 flag 默认值及 QRL config，冻结为 JSON 并记录 SHA256。`launch.py` 校验冻结文件，调用 `run_official.py`，后者直接执行官方 `main.main`。100k 模式：seed0、100,000 次更新、batch 1024、每 1,000 步训练/验证日志、每 10,000 步评价并保存 checkpoint。所有官方任务各 50 个完整 episode。保留官方第 1 步评价；不按结果停止、调参、延长预算或选择最好 checkpoint。正式结果仅为 100k 终点。烟雾是单次更新、全部任务各 1 episode，并额外检查保存/加载后一个完整 episode。

## 官方完整预算（本轮不启动）

```bash
$P "$E/protocol.py" navigate full results/ogbench_qrl_navigate_full_new
$P "$E/launch.py" results/ogbench_qrl_navigate_full_new
```

full 模式保留官方 main 的 1,000,000 更新、5,000 日志间隔、100,000 评价间隔、1,000,000 保存间隔；hyperparameters.sh 要求每任务 50 episode。仍关闭视频并离线记录 W&B。单 seed 的完整预算运行也不等同于论文多 seed 全表复现。

对应的直接官方入口命令（由薄包装处理离线日志和数据路径）：

```bash
$P "$E/run_official.py" --agent="$E/vendor/ogbench/impls/agents/qrl.py" \
  --env_name=pointmaze-medium-navigate-v0 --seed=0 --agent.alpha=0.0003 \
  --train_steps=1000000 --log_interval=5000 --eval_interval=100000 \
  --save_interval=1000000 --eval_episodes=50 --video_episodes=0 \
  --save_dir=results/ogbench_navigate_full_direct_new
```

## Stitch（本轮不启动）

```bash
$P "$E/protocol.py" stitch 100k results/ogbench_qrl_stitch_100k_new
$P "$E/launch.py" results/ogbench_qrl_stitch_100k_new
# 官方完整预算改为 stitch full，并使用另一个全新结果目录。
```

二者都为 `actor_loss=ddpgbc`、`const_std=True`、`alpha=0.0003`。navigate 的 actor 目标概率为 current/trajectory/random = 0/1/0；stitch = 0/0.5/0.5；二者 `actor_geom_sample=False`，使用官方均匀未来目标分支（包含官方 round 与终点处理），并非几何采样。value 的目标概率均为 0/0/1；`value_geom_sample=True` 在零轨迹目标概率下不改变最终目标来源。

stitch 对应的完整预算直接命令为上面的 navigate 命令将环境改成 `pointmaze-medium-stitch-v0`，另加 `--agent.actor_p_randomgoal=0.5 --agent.actor_p_trajgoal=0.5`，并更换全新 `save_dir`。本轮没有执行这些命令。

## 适配及复现边界

- `runtime.py` 显式加入固定 commit 的官方源码根目录，检查导入位置，并通过 partial 给官方 `ogbench.make_env_and_datasets` 指定本地 `dataset_dir`。仍由官方加载原始数据，保留 `compact_dataset=True`、valids、双 terminal 标记和 GCDataset 实际边界行为。显式源码路径可避免 macOS 隐藏标记导致 editable 安装的 `.pth` 被 Python 跳过。
- 官方 `setup_wandb` 显式使用 online，因此包装它指定 offline；W&B 原临时目录收纳到每次运行的 tmp 下。CSV 和 checkpoint 仍由官方实现写出。
- `--video_episodes=0` 是官方 flag，数值评价不调用 render；不必配置 EGL、显示服务或视频编码器。视频可在另一次明确指定的运行中开启。
- 依赖采用兼容的 CPU JAX 版本，源码包从固定本地 commit editable 安装，避免 PyPI ogbench 与源码版本不一致。
- W&B 固定 0.18.7，以兼容官方 `Settings(start_method='thread')`。离线模式也需要本地回环端口；在禁止绑定端口的沙箱中须使用允许本地服务的执行环境。本次两次初始化失败均保留日志，未发生训练更新。
- 没有修改官方 seed 行为；官方对 Python/NumPy/JAX 设 seed，但没有完整显式种子化 Gymnasium 的各 RNG。seed0 是官方协议 seed，不保证跨设备/依赖版本逐位复现。
- 官方 checkpoint 包含 agent、优化器和 JAX RNG，不包含全部环境/NumPy RNG，官方 restore 也不会恢复训练循环计数。本入口不把它宣称为无缝断点续训。
- 官方 TrainState.step 初值为 1，故训练 N 次后的字段为 N+1；检查脚本同时验证 Adam count=N，以免把计数偏移误判为预算错误。

## 数据审计定义

审计单独进程运行，不改变训练权重或训练 RNG。分别统计 train/validation 的原始 episode、有效转移数、episode 转移长度、全部观测和有效源观测的精确整行重复，以及 `(完整 s, 完整 a, 完整 s_next, terminal)` 的有效转移重复。使用浮点原值的数值精确相等，不舍入、不按位置投影；报告 `(N-unique)/N` 和处于重复组的行占比。episode 数根据原始 npz terminals 统计，不能将 compact 数据的双 terminal 标记误计成两个 episode。

结果位置：运行目录下 `OGBench/<run_group>/<timestamp>/train.csv`、`eval.csv`、`flags.json`、`params_<step>.pkl`；console.log、runtime.json、frozen_config.json 和 pip-freeze.txt 位于该运行目录顶层。

读取当前进度（不会导入 JAX 或干扰训练）：

```bash
$P "$E/status.py" results/ogbench_official_qrl_20261026_seed0
```

本次正式训练已经完成。训练后的原始执行链因隐藏 `.pth` 导入路径问题失败；现已修复，并补做 `check_checkpoint.py ... 100000` 和 `report.py`。结果目录扫描仅接收带 `flags.json` 的目录，避免 `.DS_Store` 干扰。

仅当终点评价、十个 checkpoint、100 行训练日志、完整加载验证、官方源码/配置/依赖版本及数据哈希检查通过后，报告脚本才写出 `COMPLETE`、`summary.json`、`analysis.json`、最终 `report.md` 和曲线。后处理校验及新修复源码保存在结果目录的 `postprocessing/`，历史源码快照、原失败日志及训练时冻结配置均保留。没有 `COMPLETE` 时不要把全部交付视为完成。
