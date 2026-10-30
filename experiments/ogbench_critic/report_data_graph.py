"""Read existing M4 artifacts and render the report; no models or episodes."""
import argparse
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


def run(root):
    root = Path(root).resolve(); out = root / 'data_graph_critic'
    data = read(out / 'analysis.json'); probes = read(out / 'local_probes_v2/analysis.json')
    protocol = data['protocol']; assert data['complete'] and data['elapsed_seconds'] < 360
    assert protocol['training_updates'] == 0 and protocol['weight'] == .5
    assert protocol['bank_sha256'] == sha(root / 'evaluation_bank.npz')
    assert protocol['graph_sha256'] == sha(out / 'graph.npz')
    for name, digest in protocol['source_sha256'].items():
        assert sha(out / 'source_snapshot' / name) == digest
        assert sha(Path(__file__).with_name(name)) == digest
    baseline = read(root / 'baseline_greedy/summary.json')['cases']
    hybrid = data['results']['hybrid']; pure = data['results']['graph']
    assert list(baseline) == list(hybrid) and len(hybrid) == 25 and len(pure) == 5
    for mode, cases in data['results'].items():
        for name, info in cases.items():
            with np.load(out / mode / f'{name}.npz') as trace:
                assert trace['observations'].shape == (info['steps'] + 1, 2)
                assert trace['actions'].shape == (info['steps'], 2)
                assert trace['final_integration'].ndim == 1 and np.isfinite(trace['final_integration']).all()
                np.testing.assert_array_equal(trace['observations'][-1], info['final_xy'])
                assert np.isfinite(trace['observations']).all() and info['steps'] <= 1000
                assert not info['success'] or np.linalg.norm(trace['observations'][-1] - trace['goal']) <= 1
    assert probes['new_episodes'] == 0 and probes['seconds'] < 30 and len(probes['probes']) == 10
    for name, info in probes['probes'].items():
        with np.load(out / 'local_probes_v2' / f'{name}.npz') as trace:
            assert trace['observations'].shape == trace['actions'].shape == (121, 2)
            np.testing.assert_allclose(trace['hybrid_distance'], .5 * trace['graph_distance'] + .5 * trace['qrl_distance'])
            assert info['minimum_successor_graph_distance'] == trace['graph_distance'].min()
            assert info['successors_with_lower_graph_distance'] == int((trace['graph_distance'] < info['current_graph_distance']).sum())
    gained = [n for n in hybrid if hybrid[n]['success'] and not baseline[n]['success']]
    lost = [n for n in hybrid if baseline[n]['success'] and not hybrid[n]['success']]
    def count(cases, task): return sum(v['success'] for n, v in cases.items() if n.startswith(f'task{task}_'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(5)
    for i, (label, cases) in enumerate([('QRL', baseline), ('Hybrid', hybrid)]):
        axes[0, 0].bar(x + (i - .5) * .34, [count(cases, t) for t in range(1, 6)], width=.34, label=label)
    axes[0, 0].set(title='Fixed-bank true-successor greedy; no actor update',
                   xticks=x, xticklabels=[f'T{t}' for t in range(1, 6)], ylabel='Successes / 5', ylim=(0, 5.5))
    axes[0, 0].legend()
    fields = {}
    for label, path in [('QRL', root / 'baseline_field/field.npz'), ('Graph', out / 'graph_field.npz'),
                        ('Hybrid', out / 'hybrid_field.npz')]:
        with np.load(path) as f: fields[label] = {k: f[k].copy() for k in ['path_values', 'walls']}
        values = fields[label]['path_values']
        axes[0, 1].plot(np.arange(len(values)), values, label=label)
    axes[0, 1].set(title='Canonical goal: values along each shortest reference route',
                   xlabel='Graph steps', ylabel='Distance units (quantization affects graph hops)')
    axes[0, 1].legend(fontsize=8)
    for wall in fields['QRL']['walls']:
        axes[1, 0].add_patch(Rectangle(wall[:2] - wall[2:], *(2 * wall[2:]), color='.83'))
    name = 'task3_seed3102'  # First case whose goal is reachable in the data graph.
    for label, path in [('Hybrid', out / 'hybrid'), ('Graph', out / 'graph')]:
        with np.load(path / f'{name}.npz') as trace:
            axes[1, 0].plot(*trace['observations'].T, label=label); goal = trace['goal']
    axes[1, 0].scatter(*goal, marker='*', color='green', s=100, label='Goal')
    axes[1, 0].set(title='Task3 case 3102 (post-hoc reachable-goal illustration)', xlim=(2, 14), ylim=(10, 20), aspect='equal')
    axes[1, 0].legend(fontsize=8)
    selected = [f'hybrid_task3_seed{s}' for s in range(3100, 3105)]
    for i, key in enumerate(selected):
        row = probes['probes'][key]
        axes[1, 1].bar(i, row['successors_with_lower_graph_distance'], color='tab:blue')
        axes[1, 1].text(i, .07, 'unreachable' if row['graph_unreachable'] else 'plateau', rotation=30, ha='center', fontsize=8)
    axes[1, 1].set(title='Saved hybrid endpoints: no one-step graph descent',
                   xticks=range(5), xticklabels=[str(s) for s in range(3100, 3105)],
                   ylabel='Lower-distance candidates / 121', ylim=(0, 1))
    fig.savefig(out / 'data_graph.png', dpi=160); plt.close(fig)
    table = '\n'.join(f'| task{t} | {count(baseline,t)}/5 | {count(hybrid,t)}/5 |' for t in range(1, 6))
    fields_table = '\n'.join(f"| {k} | {v['strict_descent_from_anchor']} | {v['minimum_peak_gap_over_iqr']:.6f} | {v['all_graph_spearman']:.6f} | {v['corridor_spearman']:.6f} |" for k, v in data['fields'].items())
    probe_table = '\n'.join(f"| {key} | {v['graph_unreachable']} | {v['current_graph_distance']:.0f} | {v['minimum_successor_graph_distance']:.0f} | {v['successors_with_lower_graph_distance']} |" for key, v in probes['probes'].items())
    report = f'''# 数据转移图对照：部分任务改善，task3仍失败

本项只使用官方训练集100万条有效真实转移构建非参数critic，不训练网络、不换seed、不运行1M。固定0.2粗粒化与0.5/0.5距离混合后，五任务真实后继贪心由16/25增至19/25，但task3仍0/5；纯数据图在task3也0/5。因此尚未达到稳定改善，不能把图内最短路或局部通道当成真实执行成功。

## 构造与预算

原官方QRL 100k文件SHA {protocol['base_checkpoint_sha256']} 保持不变。沿官方compact Dataset.valid_idxs/get_subset得到全部训练(s,s')，完整观测恰为二维；penultimate终止转移保留、无效末状态不作源、无跨episode边。未使用验证集构图、MultiEpisodeData、终态quotient、R系列或墙体/参考路线；也未改变原QRL的数据权重。

图有{protocol['nodes']}个节点、{protocol['edges']}条有向单位成本边，重复边是同一约束，未将频次当成本。完整观测rint(s/0.2)生成节点，查询使用最近已占用节点；等距时优先已占用rint格子。不可达距离固定截为节点数6302，仍是有向伪准度量。主critic为原QRL与图距离的等权凸和；图的量化/样本支持可能产生非真实捷径或遗漏连通，不能当环境真值或官方网络改善。

构图及初始化{protocol['graph_build_seconds']:.2f}秒，全部预定评价/场检查{data['elapsed_seconds']:.2f}秒，上限360秒内正常完成；参数摘要与checkpoint文件前后相同。2项测试通过，覆盖官方compact边界、重复边单位成本、方向性、不可达三角结构及保存加载。原actor/dynamics未更新；下表是诊断贪心，不是官方actor 50episode指标。

## 固定25例结果

| 任务 | 原QRL距离 | 混合距离 |
|---|---:|---:|
{table}

新增{gained}，失去{lost}。纯图仅预定task3五例，全部0/5，其他任务未运行。不能以task4新增3例替代本轮task3目标；也不调整分辨率、混合权重或插值去挽救本实验。

## canonical目标场与实际目标不同

| critic | 原锚点严格下降通道 | 峰差/IQR | 全图Spearman | 原走廊Spearman |
|---|---|---:|---:|---:|
{fields_table}

纯图canonical场排序接近参考最短路，原锚点有下降通道；混合距离仍无该严格通道。两个最低峰差均为零。它们与固定bank五个实际目标不同，不能把canonical的好图形冒充全部目标表现。参考碰撞图只在数据图保存和全部episode之后用于评价，未进入构造。

## 失败终点的只读单步核查

对已保存的两种控制器各五个task3结束integration state，另做post-hoc的121候选单步评分，共{probes['seconds']:.2f}秒（独立30秒上限），没有新增完整episode。此核查不改变主实验的194.54秒预算或结果。

| 保存终点 | 数据图不可达 | 当前图距离 | 最低真实后继图距离 | 严格降低候选数/121 |
|---|---|---:|---:|---:|
{probe_table}

前两例的目标映射节点在该有限数据图中，除自身外6301节点都不能到达它，导致截断常数平台。这是图支持/查询映射的缺陷，不是现实环境不可达证据，更不能归为原QRL根因。其余三例目标在图中可达，但保存的两种控制器终点均没有真实单步候选降低图距离；对应较低距离图邻居距实际位置的最大轴差约0.232–0.374，大于名义0.2动作位移。相邻抽象节点的单位边不等于当前完整状态下一步可达，因此有图路径仍会因量化平台停滞。这也不能反推原神经场失败仅由量化造成。

![同库执行、canonical路径值、既有失败轨迹与局部平台](data_graph.png)

## 决策与复现

不将这个带不可达目标映射和平台的图直接当神经critic监督真值，也不以重新平滑/换网格/调混合系数挽救M4。该项没有证明“只需显式最短路传播即可解决当前神经QRL”，只界定了一种便宜非参数对照的局限；仍需原神经critic上的受控改进与真实执行验证。当前总目标继续，禁止恢复seed1或1M程序。

本地路径：`results/ogbench_critic_20261029/data_graph_critic`，主日志同级`data_graph_critic.log`。命令：`run_data_graph.py <结果根目录> --output <全新目录>`；只读单步核查为`probe_data_graph.py <结果根目录> --output <全新probe目录>`；本报告使用`report_data_graph.py <结果根目录>`重建，不调用模型/环境。协议见[DATA_GRAPH_PROTOCOL.md](../../../experiments/ogbench_critic/DATA_GRAPH_PROTOCOL.md)。Git只提交源码、测试、协议、报告及引用图，原始图/轨迹/JSON/日志保留本地。

第一次post-hoc探针错误地断言所有非零图距离都必有较低邻居，在合法的不可达截断平台处退出；失败源码和日志保留在local_probes及probe_data_graph.log。修复该诊断断言后在全新local_probes_v2记录两类情况，未改主方法、重跑episode或延长主预算。
'''
    (out / 'report.md').write_text(report); print(out / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); run(parser.parse_args().root)
