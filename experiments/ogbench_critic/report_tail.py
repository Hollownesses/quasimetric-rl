"""Verify and report the fixed tail-risk experiment without rerunning training/evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(a, b):
    assert list(a) == list(b)
    return dict(gains=[k for k in a if b[k]['success'] and not a[k]['success']],
                losses=[k for k in a if a[k]['success'] and not b[k]['success']])


def run(root):
    root = Path(root).resolve(); out = root / 'tail_1000'
    data = read(out / 'analysis.json'); protocol = data['protocol']; training = data['training']
    assert data['complete'] and data['elapsed_seconds'] <= 600 and not (out / 'failure.json').exists()
    assert protocol['steps'] == training['steps'] == 1000 and protocol['batch_size'] == 256
    assert protocol['tail_count'] == 26 and protocol['tail_fraction'] == .1
    assert data['bank_sha256'] == sha(root / 'evaluation_bank.npz')
    for name, expected in protocol['source_sha256'].items():
        assert sha(out / 'source_snapshot' / name) == expected
        path = Path(__file__).with_name(name) if name != 'diagnostic_core.py' else Path(__file__).parents[1] / 'ogbench_diagnostics/core.py'
        assert sha(path) == expected
    assert sha(protocol['base_checkpoint']) == protocol['base_sha256']
    smoke = read(root / 'tail_smoke/training_complete.json')
    assert smoke['steps'] == 2 and all(v['passed'] for v in smoke['verification'].values())
    for arm, proof in training['verification'].items():
        assert proof['passed'] and proof['actual_updates'] == 1000 and proof['adam_count'] == 101000
        assert proof['frozen_parameters_and_moments'] and proof['frozen_bank_actor_outputs'] and proof['full_state_roundtrip']
        assert sha(out / arm / 'params_101000.pkl') == proof['checkpoint_sha256']
    base = read(root / 'baseline_greedy/summary.json')['cases']
    rollouts = data['rollouts']
    assert list(rollouts) == ['mean', 'tail']
    for arm, cases in rollouts.items():
        assert list(cases) == list(base) and len(cases) == 25
        for name, info in cases.items():
            with np.load(out / f'{arm}_greedy' / f'{name}.npz') as trace:
                assert trace['observations'].shape == (info['steps'] + 1, 2)
                assert trace['actions'].shape == (info['steps'], 2)
                assert trace['distances'].shape == (info['steps'],)
                assert trace['final_integration'].ndim == 1
                assert all(np.isfinite(trace[k]).all() for k in trace.files)
                np.testing.assert_array_equal(trace['observations'][-1], info['final_xy'])
                assert 0 < info['steps'] <= 1000
                assert not info['success'] or np.linalg.norm(trace['observations'][-1] - trace['goal']) <= 1
                with np.load(root / 'baseline_greedy' / f'{name}.npz') as original:
                    np.testing.assert_array_equal(trace['observations'][0], original['observations'][0])
                    np.testing.assert_array_equal(trace['goal'], original['goal'])
    initial = read(out / 'initial_statistics.json')
    with (out / 'metrics.csv').open() as f:
        rows = [{k: (v if k == 'arm' else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    log = {arm: [r for r in rows if r['arm'] == arm] for arm in rollouts}
    for arm in log:
        assert [r['step'] for r in log[arm]] == [1] + list(range(100, 1001, 100))
    counts = lambda cases: [sum(v['success'] for n, v in cases.items() if n.startswith(f'task{t}_')) for t in range(1, 6)]
    comparison = dict(tail_vs_mean=paired(rollouts['mean'], rollouts['tail']),
                      mean_vs_original=paired(base, rollouts['mean']), tail_vs_original=paired(base, rollouts['tail']))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, metric, label, logarithmic in [
        (axes[0, 0], 'pair_mean_squared_positive', 'Held-out mean local penalty', True),
        (axes[0, 1], 'pair_tail_squared_positive', 'Held-out worst 10% local penalty', True),
        (axes[0, 2], 'lam', 'Dual multiplier', False),
        (axes[1, 0], 'goal_mean', 'Held-out random-goal distance scale', False)]:
        for arm in log:
            ax.plot([0] + [r['step'] for r in log[arm]], [initial[metric]] + [r[f'heldout/{metric}'] for r in log[arm]], label=arm)
        if logarithmic: ax.set_yscale('log')
        ax.set(title=label, xlabel='Additional critic updates'); ax.legend(fontsize=8)
    for i, (label, cases) in enumerate([('Original', base), *rollouts.items()]):
        axes[1, 1].bar(np.arange(5) + (i - 1) * .25, counts(cases), width=.25, label=label)
    axes[1, 1].set(title='Fixed-bank true-successor greedy', xticks=range(5), xticklabels=[f'T{t}' for t in range(1, 6)], ylim=(0, 5.5), ylabel='Successes / 5')
    axes[1, 1].legend(fontsize=8)
    baseline_stats = read(root / 'baseline_field/summary.json')
    from scipy.spatial import cKDTree
    with np.load(root / 'baseline_field/field.npz') as reference:
        route = reference['path'].copy(); nodes = reference['nodes'].copy(); original_values = reference['values'].copy()
    indices = cKDTree(nodes).query(route)[1]
    axes[1, 2].plot((original_values[indices] - original_values[indices][0]) / baseline_stats['distance_iqr'], label='Original')
    for arm in rollouts:
        with np.load(out / f'{arm}_field.npz') as field:
            np.testing.assert_array_equal(nodes, field['nodes']); values = field['values'][indices]
        axes[1, 2].plot((values - values[0]) / data['fields'][arm]['distance_iqr'], label=arm)
    axes[1, 2].set(title='Canonical goal: SAME old reference route', xlabel='Graph steps', ylabel='(d - d_start) / field IQR'); axes[1, 2].legend(fontsize=8)
    fig.savefig(out / 'tail_comparison.png', dpi=160); plt.close(fig)

    # Illustrate all five task3 cases, not one favorable trajectory.
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), constrained_layout=True)
    with np.load(out / 'mean_field.npz') as field: walls = field['walls']
    for i, (name, ax) in enumerate(zip([f'task3_seed{s}' for s in range(3100, 3105)], axes)):
        for wall in walls: ax.add_patch(Rectangle(wall[:2] - wall[2:], *(2 * wall[2:]), color='.85'))
        for label, path in [('Original', root / 'baseline_greedy'), ('mean', out / 'mean_greedy'), ('tail', out / 'tail_greedy')]:
            with np.load(path / f'{name}.npz') as trace:
                xy = trace['observations']; goal = trace['goal']
                ax.plot(*xy.T, label=label, lw=1.2)
        ax.scatter(*goal, color='green', marker='*', s=65)
        ax.set(title=f'Existing task3 case {3100+i}', xlim=(-2, 14), ylim=(8, 22), aspect='equal')
    axes[0].legend(fontsize=7)
    fig.savefig(out / 'task3_trajectories.png', dpi=160); plt.close(fig)

    table = '\n'.join(f'| task{t} | {counts(base)[t-1]}/5 | {counts(rollouts["mean"])[t-1]}/5 | {counts(rollouts["tail"])[t-1]}/5 |' for t in range(1, 6))
    metrics = ['pair_mean_squared_positive', 'pair_tail_squared_positive', 'pair_q99', 'pair_max', 'goal_squared_positive', 'goal_mean', 'lam']
    metrics_table = '\n'.join(f'| {k} | {initial[k]:.6g} | {log["mean"][-1]["heldout/"+k]:.6g} | {log["tail"][-1]["heldout/"+k]:.6g} |' for k in metrics)
    field_table = '\n'.join(f"| {arm} | {v['strict_descent_from_anchor']} | {v['minimum_peak_gap_over_iqr']:.6f} | {v['all_graph_spearman']:.6f} | {v['corridor_spearman']:.6f} |" for arm, v in data['fields'].items())
    pair_text = '\n'.join(f'- {label}：新增{p["gains"]}，失去{p["losses"]}。' for label, p in comparison.items())
    tail_task3 = counts(rollouts['tail'])[2]
    decision = ('tail未解决task3，不搜索尾部比例、eps、学习率或lambda权重挽救本方案。' if tail_task3 == 0 else
                'tail出现task3成功，但尚需结合全部既有case、其他任务得失及策略提取核验，不能仅凭小样本成功称总目标完成。')
    report = f'''# E6：局部约束平均／尾部风险的固定短预算对照

两组从相同原始QRL 100k IQE出发，各完成额外1000次critic更新，实际Adam count101000，actor/dynamics参数与Adam矩冻结。仅将局部正部平方从均值改为最差26/256项均值，目标采样、学习率、IQE、负距离最大化和eps不变。两组同批次、沿用采样seed2001，不是换初始化/独立seed验证；与原官方1024 batch不同，只解释本配对内部聚合变量。

正式训练及完整诊断用时{data['elapsed_seconds']:.2f}秒，处于预定600秒上限；训练、保存加载与冻结核验阶段{training['train_and_verify_seconds']:.2f}秒。没有选中间模型、按成功率早停、追加更新或启动1M。3项方法测试与实际全网络两步smoke通过，正式从原100k开始而非smoke模型。

## 全部既有25例真实后继贪心

| 任务 | 原100k | mean额外1000 | tail额外1000 |
|---|---:|---:|---:|
{table}

原/mean/tail总成功数为{sum(counts(base))}/25、{sum(counts(rollouts['mean']))}/25、{sum(counts(rollouts['tail']))}/25。这是同121动作网格、完整MuJoCo状态恢复、1000步上限的诊断控制，不是官方actor 50episode指标。actor同库动作逐位未变，不声称已改善原actor。

{pair_text}

## 固定2048验证转移的约束与尺度

| 指标 | 原始 | mean终点 | tail终点 |
|---|---:|---:|---:|
{metrics_table}

尾部约束仍是上界风险定义，不能自动保证目标距离沿可达路径下降。两组的乘子面对不同风险统计，lambda数值不能直接当作同一原约束已被更好满足。训练取ceil(0.1*256)=26，即10.15625%；验证尾部取205/2048。两种风险均保存，不只报告有利指标；尺度变化需要和归一化场/实际执行一起解释。

tail的验证平均/尾部局部惩罚分别约为mean的10.6%/10.8%，平均目标距离也下降约21.1%。风险收紧确实生效，却伴随三个原成功例丢失，说明“把较大局部违反压低”在当前预算下没有自动转化为更好的闭环距离。该目标也比原均值约束更严格，不能把差异单独归为稀有状态重加权或某一墙角得到关注。

![约束曲线、乘子、距离尺度、全任务执行与同一路线场](tail_comparison.png)

## canonical目标场与全部task3轨迹

| critic | 原锚点严格下降通道 | 最低峰差/IQR | 全图Spearman | 原走廊Spearman |
|---|---|---:|---:|---:|
{field_table}

场采用既有0.2参考图/0.7半径、原锚点及canonical目标，不参与训练；它与bank五个实际目标区分。图吸附不是连续环境最优性证明，峰差下降/归零也不等价严格下降通道。

tail相对mean的原锚点峰差/IQR从0.1357降为0.0736，仍为正且没有通道；原走廊排序相关反从0.2455降至0.1264。它改变了局部排序和距离尺度，但没有消除阻断。不能由此声称局部约束与原失败无关，或声称1000更新已充分收敛；这里只否定这次固定短预算改法达成稳定改善。

![全部五个既有task3 case的真实轨迹](task3_trajectories.png)

## 决策与复现

{decision} 本项只能解释当前初始化、固定数据流和短预算的结果，不作跨seed或完整预算概括。当前稳定改善总目标仍未完成；不恢复seed1/1M，也不把它们作为完成门槛。

完整模型/优化器/RNG保存加载逐位相等，数值有限、count/step正确，冻结模块的参数及非零Adam矩不变，同25例actor输出相等；两个终点SHA、源码快照、同bank起点/目标、逐步评分及结束integration state已核对。批次流SHA为{training['paired_stream_sha256']}。mean checkpoint SHA为{training['verification']['mean']['checkpoint_sha256']}；tail为{training['verification']['tail']['checkpoint_sha256']}。

协议见[TAIL_PROTOCOL.md](../../../experiments/ogbench_critic/TAIL_PROTOCOL.md)。实际路径`results/ogbench_critic_20261029/tail_1000`，日志同级`tail_1000.log`；入口`run_tail.py <结果根目录> --output <全新目录>`，两步smoke增加`--smoke`。本报告通过`report_tail.py <结果根目录>`只读重建，不重新训练或评价。只提交源码/测试/协议/报告与引用图，完整模型、CSV/JSON/NPZ、日志保留本地。
'''
    (out / 'report.md').write_text(report); print(out / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); run(parser.parse_args().root)
