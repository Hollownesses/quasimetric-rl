"""Report the completed temporal diagnostic without loading models or rerunning episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root = Path(root).resolve()
    out = root / 'mqe_temporal_diagnostic'
    data = json.loads((out / 'analysis.json').read_text())
    assert data['complete'] and data['elapsed_seconds'] < 360
    protocol = data['protocol']
    assert protocol['epochs'] == [50000, 100000] and protocol['training_updates'] == 0
    assert protocol['source_sha256'] == sha(Path(__file__).with_name('temporal_diagnostic.py'))
    assert protocol['spec_sha256'] == sha(Path(__file__).with_name('TEMPORAL_PROTOCOL.md'))
    assert protocol['bank_sha256'] == sha(root / 'evaluation_bank.npz')
    for epoch, model in protocol['models'].items():
        assert sha(Path(model['checkpoint']) / f'params_{epoch}.pkl') == model['checkpoint_sha256']

    groups = data['results']
    names = list(groups['50000_V-greedy'])
    assert len(names) == 5 and all(n.startswith('task3_') for n in names)
    for key, cases in groups.items():
        assert len(cases) == (25 if key == '50000_actor' else 5)
        for name, info in cases.items():
            with np.load(out / key / f'{name}.npz') as trace:
                assert trace['observations'].shape == (info['steps'] + 1, 2)
                assert trace['actions'].shape == (info['steps'], 2)
                assert 0 < info['steps'] <= 1000
                assert np.isfinite(trace['observations']).all() and np.isfinite(trace['actions']).all()
                np.testing.assert_array_equal(trace['observations'][-1], info['final_xy'])
                if 'Q-greedy' in key:
                    assert trace['q_selected'].shape == (info['steps'],)
                    assert np.isfinite(trace['q_selected']).all()
                with np.load(root / 'mqe_100k_diagnostics' / 'actor_rollout' / f'{name}.npz') as old:
                    np.testing.assert_array_equal(trace['observations'][0], old['observations'][0])
                    np.testing.assert_array_equal(trace['goal'], old['goal'])

    def count(cases, task):
        return sum(v['success'] for n, v in cases.items() if n.startswith(f'task{task}_'))

    actor50, actor100 = groups['50000_actor'], data['existing_100k_actor']
    assert set(actor50) == set(actor100) and len(actor50) == 25
    lost = [n for n in actor50 if actor50[n]['success'] and not actor100[n]['success']]
    gained = [n for n in actor50 if not actor50[n]['success'] and actor100[n]['success']]
    score_rows = []
    for critic in [50000, 100000]:
        for actor in [50000, 100000]:
            scores = data['cross_scores'][f'critic{critic}_actor{actor}']
            assert len(scores) == 10
            gap = np.array([r['actor_minus_grid'] for r in scores.values()])
            looseness = np.array([r['grid_min_q_minus_v'] for r in scores.values()])
            assert np.isfinite(gap).all() and np.isfinite(looseness).all()
            score_rows.append((critic, actor, gap.mean(), gap.min(), gap.max(), looseness.mean()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(5)
    for i, (epoch, cases) in enumerate([(50000, actor50), (100000, actor100)]):
        axes[0, 0].bar(x + (i - .5) * .34, [count(cases, t) for t in range(1, 6)],
                       width=.34, label=f'{epoch // 1000}k')
    axes[0, 0].set(title='Same 25-case bank: native actor', xticks=x,
                   xticklabels=[f'T{t}' for t in range(1, 6)], ylabel='Successes / 5', ylim=(0, 5.5))
    axes[0, 0].legend()
    for i, critic in enumerate([50000, 100000]):
        values = [r[2] for r in score_rows if r[0] == critic]
        axes[0, 1].bar(np.arange(2) + (i - .5) * .34, values, width=.34,
                       label=f'critic {critic // 1000}k')
    axes[0, 1].set(title='10 fixed probes: learned Q(actor) - min grid Q',
                   xticks=[0, 1], xticklabels=['actor 50k', 'actor 100k'], ylabel='Log-discount distance')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(range(5), [data['start_fields'][n]['50000']['minimum_peak_gap_over_iqr'] for n in names],
                    marker='o', label='50k')
    axes[1, 0].plot(range(5), [data['start_fields'][n]['100000']['minimum_peak_gap_over_iqr'] for n in names],
                    marker='o', label='100k')
    axes[1, 0].set(title='All five starts lack a strict descent route at both epochs',
                   xticks=range(5), xticklabels=[str(3100 + i) for i in range(5)],
                   xlabel='Existing task3 case label', ylabel='Minimum peak gap / IQR')
    axes[1, 0].legend()
    name = 'task3_seed3101'  # The only lost actor success; explicitly post-hoc illustration.
    with np.load(root / 'mqe_endpoint_analysis' / f'{name}_field.npz') as field:
        walls = field['walls'].copy()
    for wall in walls:
        axes[1, 1].add_patch(Rectangle(wall[:2] - wall[2:], *(2 * wall[2:]), color='.82'))
    for epoch, path in [(50000, out / '50000_actor'), (100000, root / 'mqe_100k_diagnostics/actor_rollout')]:
        with np.load(path / f'{name}.npz') as trace:
            xy = trace['observations']; goal = trace['goal']
            axes[1, 1].plot(*xy.T, label=f'actor {epoch // 1000}k')
    axes[1, 1].scatter(*goal, marker='*', s=100, color='green', label='Goal')
    axes[1, 1].set(title='Case 3101: post-hoc view of the lost success', xlim=(-2, 12), ylim=(10, 22), aspect='equal')
    axes[1, 1].legend(fontsize=8)
    fig.savefig(out / 'temporal.png', dpi=160)
    plt.close(fig)

    table = '\n'.join(f'| task{t} | {count(actor50,t)}/5 | {count(actor100,t)}/5 |' for t in range(1, 6))
    scores = '\n'.join(f'| {c//1000}k | {a//1000}k | {m:.6f} | [{lo:.6f}, {hi:.6f}] | {v:.6f} |'
                       for c, a, m, lo, hi, v in score_rows)
    fields = '\n'.join(f"| {n} | {data['start_fields'][n]['50000']['minimum_peak_gap_over_iqr']:.6f} | "
                       f"{data['start_fields'][n]['100000']['minimum_peak_gap_over_iqr']:.6f} | False / False |" for n in names)
    report = f'''# MQE已有50k/100k模型：不能只归因于actor提取

本项只读同一次已完成MQE训练的预定50k中点和100k终点，耗时{data['elapsed_seconds']:.2f}秒（上限360秒），训练更新为零。50k仅作机制诊断，正式结果仍是100k；不挑60k峰值、不重训、不换seed、不启动1M。

## 同例真实执行

| 任务 | 50k原生actor | 100k原生actor（复用） |
|---|---:|---:|
{table}

同25例从21/25到20/25，新增成功{gained}，失去{lost}。原生actor的task3为1/5到0/5；这组固定样本仅支持一个已有成功例失去，不能用它声称重现了官方随机50episode全部幅度。官方已有50k/100k task3记录为22/50、2/50，评价样本流不同。

两个checkpoint的max-Q网格贪心在task3均0/5；50k真实后继max-V贪心也0/5，与复用的100k结果相同。全部失败episode都执行到1000步。Q控制直接最大化原生learned Q的价值（等价最小化距离），不是模拟器后继评分，也不是官方actor；离散网格、数据外动作和learned Q误差限制其含义。其失败不证明连续动作全局最优策略必然失败，但不支持“只修actor提取即可恢复”的诊断。

## 同观测/目标的2×2交叉评分

10个probe固定为五个task3起点与已有100k actor轨迹第100步观测。距离用原生双critic的max聚合，以下均为log-discount单位。actor减网格最小距离允许负数，不称全局最优性差距；min Q减V也不是独立Bellman真值误差。

| critic | actor | actor−网格最小Q均值 | 范围 | 网格最小Q−V均值 |
|---|---|---:|---:|---:|
{scores}

在两个固定critic下，100k actor的平均距离差反而小于50k actor，每个critic各有7/10 probe如此。它与失去一个真实成功例并不矛盾：learned Q的局部评分并不等价于真实闭环质量。这些post-hoc probe不构成actor损失的因果消融，也不能单凭Q−V差命名为后期遗忘或目标不一致的根因。

## 五个实际目标的起点距离场

| 既有task3 case | 50k峰差/IQR | 100k峰差/IQR | 50k / 100k严格下降通道 |
|---|---:|---:|---|
{fields}

两模型五个起点均无严格下降通道。100k部分最低峰差降为零，仍不表示存在全程下降路线；另两例峰差增大。使用原0.2网格/0.7碰撞半径，起点吸附误差最大0.1305；仅为离散图诊断。下图3101轨迹在看完结果后选作失去成功例的说明，不作预定独立验证。

![固定bank、交叉评分、起点场与失去成功例](temporal.png)

## 核验、偏离及复现

两完整checkpoint的实际更新计数50k/100k由原生loader验证，文件SHA与参数摘要前后相同；同bank SHA、保存轨迹起点/目标、完整观测/动作维数、终点与步数均已核对。2项测试通过：原生Q/max评分语义与360秒预算拒绝。50k SHA为{protocol['models']['50000']['checkpoint_sha256']}；100k SHA为{protocol['models']['100000']['checkpoint_sha256']}。

透明记录两处文字偏离：协议写“保存完整结束状态”，实际core.rollout保存了完整2D观测和动作序列，未另存MuJoCo内部结束integration state；候选评分仍使用原核验过的完整状态恢复。metadata中loader遗留的“No Q-based action search used”描述来自旧V诊断，不能套用本项Q控制；以controller目录名和上述定义为准。没有为补齐产物重复episode或修改已冻结脚本/协议。

入口：`temporal_diagnostic.py <结果根目录> --output <全新目录>`；本报告复现使用`report_temporal.py <结果根目录>`，只读取已有数组，不调用模型或环境。协议见[TEMPORAL_PROTOCOL.md](../../../experiments/ogbench_mqe/TEMPORAL_PROTOCOL.md)。原始JSON/NPZ/日志保留本目录，Git仅提交源码、测试、协议、报告及引用图。

目前仍无稳定执行改善。已明确不能用canonical角点通道、局部Q评分或最好中点替代正式100k闭环结果。后续继续针对critic目标/长程排序做小预算受控改进，不换seed、不运行1M，也不将这些取消的运行设为完成门槛。
'''
    (out / 'report.md').write_text(report)
    print(out / 'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    run(parser.parse_args().root)
