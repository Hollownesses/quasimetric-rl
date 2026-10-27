"""Summarize the fixed endpoint; never select a best checkpoint."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys

from runtime import COMMIT, ROOT, UPSTREAM

root = Path(sys.argv[1]).resolve()
run = root/'main'
directories = [p for p in run.glob('OGBench/*/*') if p.is_dir() and (p/'flags.json').is_file()]
assert len(directories) == 1
out = directories[0]


def read_csv(name):
    with (out/name).open() as f:
        return [{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]


evaluation = read_csv('eval.csv')
training = read_csv('train.csv')
assert all(math.isfinite(v) for row in evaluation + training for v in row.values())
assert [int(row['step']) for row in evaluation] == [1, *range(10000,100001,10000)]
assert [int(row['step']) for row in training] == list(range(1000,100001,1000))
assert all((out/f'params_{step}.pkl').is_file() for step in range(10000,100001,10000))
endpoint = evaluation[-1]
assert endpoint['step'] == 100000
runtime = json.loads((run/'runtime.json').read_text())
assert runtime['exit_code'] == 0
audit = json.loads((root/'data_audit.json').read_text())
smoke = json.loads((root/'smoke/checkpoint_verification.json').read_text())
check = json.loads((run/'checkpoint_verification.json').read_text())
assert smoke['passed'] and check['passed']
assert check['all_state_leaves_finite'] and check['optimizer_step'] == 100000
assert check['official_trainstate_step'] == 100001
manifest = json.loads((run/'frozen_config.json').read_text())
assert manifest['official_commit'] == COMMIT
assert hashlib.sha256((run/'frozen_config.json').read_bytes()).hexdigest() == (run/'frozen_config.sha256').read_text().strip()
actual_flags = json.loads((out/'flags.json').read_text())
assert actual_flags['agent'] == manifest['flags']['agent']
for key in ['train_steps','env_name','seed','eval_interval','eval_episodes','eval_tasks','save_interval','log_interval',
            'eval_temperature','eval_gaussian','video_episodes','restore_path']:
    assert actual_flags[key] == manifest['flags'][key], key
assert actual_flags['restore_path'] is None and actual_flags['eval_tasks'] is None
assert actual_flags['eval_episodes'] == 50
for name, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() == expected, name
for split in ['train','validation']:
    assert hashlib.sha256(Path(audit[split]['path']).read_bytes()).hexdigest() == audit[split]['sha256']
checkpoint_hashes = {f'params_{step}.pkl':hashlib.sha256((out/f'params_{step}.pkl').read_bytes()).hexdigest()
                     for step in range(10000,100001,10000)}
assert checkpoint_hashes['params_100000.pkl'] == check['checkpoint_sha256']
tasks = [task['task_name'] for task in audit['tasks']]
assert len(tasks) == 5
for row in evaluation:
    rates = [row[f'evaluation/{task}_success'] for task in tasks]
    assert all(0 <= v <= 1 and math.isclose(v*50,round(v*50),abs_tol=1e-6) for v in rates)
    assert math.isclose(statistics.mean(rates),row['evaluation/overall_success'],abs_tol=1e-6)
recovery = root/'postprocessing'
recovery.mkdir(exist_ok=True)
versions = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata['Name'].lower().replace('_','-').replace('.','-')
    versions[name] = distribution.version
version_comparison = {}
for line in (ROOT/'requirements-lock.txt').read_text().splitlines():
    if '==' in line:
        name, expected = line.split('==',1)
        actual = versions.get(name.lower().replace('_','-').replace('.','-'))
        assert actual == expected, (name, actual, expected)
        version_comparison[name] = dict(expected=expected, actual=actual)
(recovery/'dependency_version_comparison.json').write_text(json.dumps(version_comparison,indent=2)+'\n')
metric_statistics = {}
for key in training[0]:
    if key.startswith(('training/','validation/')):
        values = [r[key] for r in training]
        metric_statistics[key] = dict(first_logged_1k=values[0],endpoint_100k=values[-1],
            minimum_logged=min(values),maximum_logged=max(values),last_10_logged_mean=statistics.mean(values[-10:]),
            last_10_logged_std=statistics.pstdev(values[-10:]))
analysis = dict(fixed_endpoint=100000,learning_curve=evaluation,logged_metric_statistics=metric_statistics,
    finite_train_and_validation_logs=True,train_log_rows=100,evaluation_rounds=11,
    formal_episodes_per_round=250,formal_evaluation_episodes_total=2750,
    task_success_counts={task:round(endpoint[f'evaluation/{task}_success']*50) for task in tasks},
    checkpoint_sha256=checkpoint_hashes,
    evidence_limits=['Single seed and 100k budget.','CSV does not identify the mechanism of task3 failures.',
                     'Normalized actor q_loss near 1 does not imply zero gradients.',
                     'Exact duplicates do not measure approximate density imbalance.'])
(root/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
summary = dict(status='complete', fixed_endpoint=endpoint, runtime=runtime,
               final_training_metrics=training[-1], checkpoint_directory=str(out),
               official_commit=manifest['official_commit'], smoke=smoke,
               checkpoint_sha256=checkpoint_hashes,report='report.md',analysis='analysis.json',
               final_checkpoint_verification=check)

os.environ.setdefault('MPLCONFIGDIR',str(recovery/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for key in endpoint:
    if key != 'step':
        axes[0,0].plot([r['step'] for r in evaluation],[r[key] for r in evaluation],
                       label=key.replace('evaluation/',''))
axes[0,0].set(ylabel='Success rate', ylim=(-.02,1.02))
axes[0,0].legend(fontsize=7)
for key in ['training/value/d_pos_loss', 'validation/value/d_pos_loss']:
    axes[0,1].plot([r['step'] for r in training],[r[key] for r in training],label=key)
axes[0,1].axhline(.05,color='black',linestyle='--',label='eps=0.05')
axes[0,1].set(ylabel='Mean relu(d_pos - 1)^2',yscale='log')
axes[0,1].legend(fontsize=7)
axes[1,0].plot([r['step'] for r in training],[r['training/value/lam'] for r in training])
axes[1,0].set(ylabel='Official dual multiplier lambda')
for key in ['training/value/value_loss', 'training/actor/actor_loss', 'training/dynamics/dynamics_loss']:
    axes[1,1].plot([r['step'] for r in training],[r[key] for r in training],label=key)
axes[1,1].set(yscale='symlog', ylabel='Loss (symlog)')
axes[1,1].legend(fontsize=7)
for axis in axes.flat:
    axis.set_xlabel('Updates')
    axis.grid(alpha=.2)
fig.tight_layout()
fig.savefig(root/'learning_curves.png',dpi=160)
fig.savefig(root/'learning_curves.pdf')
plt.close(fig)

rows = ['| 官方任务 | 100k 成功数 / 50 | 成功率 |', '|---|---:|---:|']
for key, value in endpoint.items():
    if key not in ('step','evaluation/overall_success'):
        rows.append(f'| {key.removeprefix("evaluation/").removesuffix("_success")} | {round(value*50)} / 50 | {value:.1%} |')
rows.append(f'| 平均 | — | {endpoint["evaluation/overall_success"]:.1%} |')
curve_rows = ['| 更新步数 | task1 | task2 | task3 | task4 | task5 | 平均 |', '|---:|---:|---:|---:|---:|---:|---:|']
for row in evaluation:
    rates = [f'{row[f"evaluation/{task}_success"]:.0%}' for task in tasks]
    curve_rows.append(f'| {int(row["step"]):,} | '+' | '.join(rates)+f' | {row["evaluation/overall_success"]:.0%} |')
diagnostic_rows = ['| 日志指标 | 1k 首次记录 | 100k 终点 | 91k–100k 十个日志批次均值 |', '|---|---:|---:|---:|']
for key in ['training/value/d_pos_loss','validation/value/d_pos_loss','training/value/lam',
            'training/dynamics/dynamics_loss','validation/dynamics/dynamics_loss',
            'training/value/d_neg_mean','training/actor/actor_loss','training/actor/mse']:
    m = metric_statistics[key]
    diagnostic_rows.append(f'| `{key}` | {m["first_logged_1k"]:.6g} | {m["endpoint_100k"]:.6g} | {m["last_10_logged_mean"]:.6g} |')
audit_rows = ['| 数据 | 原始观测行 | 有效转移 | episode | 长度 min / mean / max | 全观测冗余率 | 全转移冗余率 |', '|---|---:|---:|---:|---|---:|---:|']
for split in ['train','validation']:
    a = audit[split]
    lengths = a['episode_transition_lengths']
    audit_rows.append(f'| {split} | {a["shapes"]["observations"][0]} | {a["valid_transitions"]} | {a["episodes"]} | {lengths["min"]} / {lengths["mean"]:.1f} / {lengths["max"]} | {a["full_observations_all_rows"]["redundant_fraction"]:.6%} | {a["full_valid_transitions"]["redundant_fraction"]:.6%} |')
tz = timezone(timedelta(hours=8))
started = datetime.fromtimestamp(runtime['start_unix'],tz).strftime('%Y-%m-%d %H:%M:%S')
ended = datetime.fromtimestamp(runtime['end_unix'],tz).strftime('%Y-%m-%d %H:%M:%S')
generated = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
text = f'''# OGBench 官方 QRL 100k 接入试跑报告

报告生成时间：{generated}（Asia/Shanghai）。接入、烟雾、固定 100,000 次更新、正式终点评价和后续 checkpoint 加载校验均已完成。**100k 终点五任务平均成功率为 80%：四个任务各 50/50，task3 为 0/50。** 本结果是 pointmaze-medium-navigate-v0、seed0 的 100k 接入试跑，完整论文预算复现仍未进行。没有启动 stitch、多 seed 或其他算法，没有按成功率筛选 checkpoint。

## 固定版本、协议与设备

官方仓库：[seohongpark/ogbench](https://github.com/seohongpark/ogbench)，固定 commit：`{manifest['official_commit']}`，包版本 1.2.1。官方 main.py、agents/qrl.py 及冻结配置内全部官方 Python 源码哈希核对一致，归档见 [official_source.tar.gz](official_source.tar.gz)。

独立 Python 3.10 环境，macOS 26.5.2 arm64、32 GiB 内存、10 核 CPU。训练设备：`{manifest['jax_devices']}`；恢复验证设备：`{check['devices']}`，均为 CPU。主要依赖为 JAX/jaxlib 0.4.35、Flax 0.10.2、Optax 0.2.4、Distrax 0.1.5、NumPy 1.26.4、MuJoCo 3.17.0、Gymnasium 1.4.0、W&B 0.18.7。修复后各依赖版本与安装锁逐项一致，见 [依赖版本核对](postprocessing/dependency_version_comparison.json)。

固定配置为 IQE QRL、actor/value 隐层 (512,512,512)、latent_dim=512、layer_norm=True、lr=0.0003、eps=0.05；actor_loss=ddpgbc、const_std=True、alpha=0.0003。batch_size=1024，seed=0，从头初始化、restore_path=None；100k 更新，每 1k 记录训练/验证日志，每 10k 评价及保存 checkpoint。实际运行 flags 与 [冻结配置](main/frozen_config.json) 核对一致，[配置 SHA256](main/frozen_config.sha256) 校验通过。

正式主进程：{started} → {ended}（Asia/Shanghai），耗时 **{runtime['elapsed_seconds']/3600:.3f} 小时**，平均约 {100000/runtime['elapsed_seconds']:.3f} 次更新/秒，包含初始化、编译、训练、逐轮评价、保存和日志退出；主进程退出码为 0。

## 固定终点的官方评价

{chr(10).join(rows)}

全部官方任务各 50 episode，官方 actor、temperature=0、无额外 Gaussian 噪声，成功率来自 step=100000 的官方 eval.csv。学习曲线仅用于观察，见 learning_curves.png/pdf；每轮 250 episode，保留官方 step1 评价，总计 11 轮。单 seed、每任务 50 个 episode 的结果存在采样波动，不能据此宣称达到论文成绩。

{chr(10).join(curve_rows)}

四个成功任务在首次 10k 评价已经达到 100%，此后每个评价点保持相同结果；task3 在所有评价点均为 0%。本轮证明接入可训练且能完成四个官方任务，但没有观察到 10k 以后成功率继续提高。task3 的官方起点为 (8,16)、目标为 (4,12)，其失败机制尚未定位：现有 CSV 没有保存逐步轨迹，不能据此断定是距离估计、策略提取、数据覆盖或绕行中的哪一环节造成失败。

共完成 2,750 个正式评价 episode，正式结论仅取固定 100k 终点。额外加载验证 episode 不计入上表，也没有替换任何正式评价结果。

![学习曲线](learning_curves.png)

曲线文件：[PNG](learning_curves.png)、[PDF](learning_curves.pdf)。

## 烟雾及 checkpoint

烟雾通过：一次更新，全部五任务各一个完整 episode；随后确认参数发生改变、Adam count=1、checkpoint 完整 PyTree 保存/加载相等，并完成额外评价 episode（{smoke['complete_episode_steps']} 步），见 [烟雾验证](smoke/checkpoint_verification.json)。

100k checkpoint 亦通过相同 round-trip 检查：Adam count=100000，官方 TrainState.step=100001（其初值为 1），所有状态叶子为有限值，参数与初始化不同，包含优化器和 JAX RNG 的完整状态逐叶一致。加载后的 task1 完整 episode 为 {check['complete_episode_steps']} 步，success={check['success']:.0f}。该 episode 只验证加载后的模型能够运行，未计入正式成功率，见 [终点验证](main/checkpoint_verification.json)。

## 数据只读审计

{chr(10).join(audit_rows)}

“冗余率”定义为 `(N-unique)/N`。完整观测和 `(s,a,s_next,terminal)` 按全部维度精确比较，无舍入和位置投影。原始 episode 由原始 terminals 计算，compact valids 保持官方有效转移语义。更详细的重复组比例、最大重数、数据 SHA256、维数及 dtype 见 data_audit.json。没有根据审计结果修改采样权重。精确重复统计只能判断完全重复样本的规模，不能单凭它推断近邻密度不均衡、拓扑瓶颈或后续加权方法收益。

本任务官方完整观测本身是二维 PointMaze 状态，动作同为二维，没有主动截取位置或丢弃状态维度。训练集完整观测冗余 345 行、完整有效转移冗余 19 行；验证集分别为 30 行、2 行。仅考察有效源状态时，训练集冗余 342 行、占 0.0342%。重复组最大重数均为 2。报告生成前再次核对原始 train/val 数据 SHA256 均未改变。

对后续源/边均衡而言，完全相同源/边的重复规模很小，单纯按精确重复次数均衡直接涉及的样本比例有限；近邻密度、瓶颈覆盖等问题不由此统计刻画。因此这不能排除其他均衡方法，也不能作为其有效性的证明。本轮没有进行重加权或去重。

## 训练损失、局部约束与乘子分析

train.csv 共 100 行，覆盖 1k–100k，训练及验证的全部记录均为有限值，没有 NaN/Inf。下表 1k 是第一次日志采样，不是模型初始化时的损失。

{chr(10).join(diagnostic_rows)}

100k 时训练批次的 d_pos_loss={training[-1]['training/value/d_pos_loss']:.6g}，官方约束容差 eps=0.05，lambda={training[-1]['training/value/lam']:.6g}。这只是最后一个日志批次的局部约束统计，不是全数据约束保证。train.csv 同时保留训练/验证的 d_pos min/mean/max、局部罚项、lambda、value/actor/dynamics 损失及耗时。

官方局部统计为 `mean(relu(d(s,s_next)-1)^2)`；终点训练/验证约 0.04853/0.04792，最后十个日志批次均值约 0.04813/0.04792。乘子从首次记录约 1.8 增长并在后期约 183 附近稳定。这支持“被记录批次的平均局部罚项已压至容差附近”，不等于每条边都满足 d<=1，也不保证 task3 的目标策略已学会。

动力学训练/验证损失从首次记录约 4.54/4.46 降至终点约 0.0624/0.0576，两者后期量级接近。随机目标距离均值由早期约 278 变为终点约 69.3，要结合局部约束和乘子一起解读，不能用某个损失是否单调下降判定整个约束优化是否正确。

actor 总损失约 1.0009，并不能仅凭数值接近常数判断 actor 没有学习。官方 DDPG+BC 使用 `q_loss=-mean(q)/stop_gradient(mean(abs(q))+1e-6)`；q 为负时前向标量接近 1，但分母停止梯度，分子梯度不会因此相消。BC 项约 0.0009，与官方 alpha=0.0003 有关。actor 训练 MSE 没有整体下降，说明模仿误差和任务成功率并非同一指标。日志未单独保存 actor 梯度范数，因此不把该解释延伸为 task3 失败的因果诊断。

完整指标首值、终值、范围和最后十个批次均值/标准差见 [analysis.json](analysis.json)。

## 后处理故障修复与复现边界

训练结束后最初的报告链失败，是导入路径和目录发现问题：

1. `ogbench.pth` 带 macOS `UF_HIDDEN` 标记，Python 的 site.addpackage 会跳过隐藏 .pth。包元信息和固定源码仍在，但源码目录未进入 sys.path，导致 `ModuleNotFoundError: ogbench`。
2. runtime.py 现显式加入已固定的官方源码根目录，并验证实际导入位置；没有修改依赖版本或官方算法。
3. 结果目录扫描原先会把 Finder 的 `.DS_Store` 算作实验目录，现仅接收带 flags.json 的目录。
4. 原失败日志保留为 main/checkpoint_check_failed_import.log、main/checkpoint_check_failed_discovery.log；原临时报告备份在 postprocessing/provisional_report_before_recovery.md。正式文件统一命名为 **report.md**。

本次只补做后处理及加载验证，没有重训、续训、增加更新预算或改写原始训练 checkpoint、train.csv、eval.csv。十个 checkpoint 的 SHA256 见 analysis.json、summary.json；新修复源码及哈希位于 postprocessing，训练时冻结配置与历史源码快照均保留。

安装、复现命令、1M 官方预算命令及 stitch 配置差异见仓库 experiments/ogbench_official_qrl/README.md。适配仅包括 CPU 依赖、本地 dataset_dir、显式离线 W&B、关闭视频、预算和日志频率。官方网络、损失、目标采样和策略提取未修改；没有迁移现有实验权重、PHR、Dubins 或 MPPI。

官方 seed0 设置未覆盖 Gymnasium 的所有随机源，checkpoint 也不保存完整 NumPy/环境 RNG，官方 restore 不恢复主循环计数，因此不承诺跨机器逐位一致或无缝断点续训。

## 安装、复现与未启动的完整预算

完整命令见 [实验入口 README](../../experiments/ogbench_official_qrl/README.md)。从仓库根目录仅重做当前结果的加载检查与报告：

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_official_qrl
R=results/ogbench_official_qrl_20261026_seed0
$P "$E/check_checkpoint.py" "$R/main" 100000
$P "$E/report.py" "$R"
```

stitch 对应官方配置已准备：actor 的 current/trajectory/random 概率由 navigate 的 0/1/0 改为 0/0.5/0.5；两者 actor_geom_sample=False，均使用官方均匀未来目标分支，alpha=0.0003，actor_loss=ddpgbc。value 目标概率均为 0/0/1。本轮没有启动 stitch。

官方完整预算为 1M 更新、5k 日志、100k 评价、1M 保存、每任务 50 episode，后续手动命令如下，本轮未执行：

```bash
$P "$E/protocol.py" navigate full results/ogbench_qrl_navigate_full_new
$P "$E/launch.py" results/ogbench_qrl_navigate_full_new
$P "$E/protocol.py" stitch full results/ogbench_qrl_stitch_full_new
$P "$E/launch.py" results/ogbench_qrl_stitch_full_new
```

## 产物索引与结论

checkpoint/CSV 目录：`{out}`。各 10k checkpoint 全部保留，正式终点为 params_100000.pkl。

- 正式结果：[summary.json](summary.json)、本文件 report.md；统计分析：[analysis.json](analysis.json)。
- [训练日志](main/OGBench/{out.parent.name}/{out.name}/train.csv)、[官方评价日志](main/OGBench/{out.parent.name}/{out.name}/eval.csv)、[主进程日志](main/console.log)、[耗时](main/runtime.json)。
- [冻结配置](main/frozen_config.json)、[配置哈希](main/frozen_config.sha256)、[训练时依赖](main/pip-freeze.txt)，离线 W&B 在 main/tmp。
- [数据审计](data_audit.json)、[加载验证](main/checkpoint_verification.json)、[学习曲线 PNG](learning_curves.png) / [PDF](learning_curves.pdf)。
- [后处理校验清单及哈希](postprocessing/integrity.json)，全部生成及校验成功后写出 [COMPLETE](COMPLETE)。

接入已成功，固定 100k 的学习表现为平均 80%、task3 持续失败；官方 1M、多 seed 与完整论文结果复现仍待后续独立实验。
'''
(root/'report.md').write_text(text)
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
source_dir = recovery/'source_snapshot'
source_dir.mkdir(exist_ok=True)
for name in ['runtime.py','check_checkpoint.py','report.py','README.md']:
    (source_dir/name).write_bytes((ROOT/name).read_bytes())
integrity = dict(passed=True,generated_at=generated+' Asia/Shanghai',
    all_official_python_sources_match_frozen_hashes=True,frozen_configuration_hash_valid=True,
    actual_flags_match_frozen=True,dataset_hashes_valid=True,dependency_versions_match_lock=True,
    all_csv_values_finite=True,train_rows=100,evaluation_rows=11,checkpoint_sha256=checkpoint_hashes,
    postprocessing_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir() if p.is_file()},
    deliverable_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ['report.md','summary.json','analysis.json','learning_curves.png','learning_curves.pdf']})
(recovery/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
(root/'COMPLETE').write_text('100k fixed endpoint, checkpoint recovery, integrity checks and report.md completed.\n')
print(text)
