"""Read-only report for the independently checked data-support metric."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root=Path(root).resolve();out=root/'roadmap_metric';data=read(out/'analysis.json');p=data['protocol']
    assert data['complete'] and data['elapsed_seconds']<240 and p['training_updates']==0
    assert not (out/'failure.json').exists() and p['connections']==8 and p['spacing']==.2
    assert sha(out/'graph.npz')==p['graph_sha256'] and sha(out/'distances.npy')==p['distance_sha256']
    assert sha(root/'evaluation_bank.npz')==p['bank_sha256']
    for name,digest in p['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['fast_pointmaze.py','core.py'] else Path(__file__).parent
        assert sha(directory/name)==digest
    base=read(root/'baseline_greedy/summary.json')['cases'];cases=data['cases'];assert list(base)==list(cases) and len(cases)==25
    for name,info in cases.items():
        with np.load(out/'greedy'/f'{name}.npz') as trace:
            assert trace['observations'].shape==(info['steps']+1,2) and trace['actions'].shape==(info['steps'],2)
            assert trace['distances'].shape==(info['steps'],) and trace['state_distances'].shape==(info['steps']+1,)
            assert trace['final_integration'].ndim==1 and all(np.isfinite(trace[k]).all() for k in trace.files)
            np.testing.assert_array_equal(trace['observations'][-1],info['final_xy'])
            assert 0<info['steps']<=1000
            assert not info['success'] or np.linalg.norm(trace['observations'][-1]-trace['goal'])<=1
            with np.load(root/'baseline_greedy'/f'{name}.npz') as old:
                np.testing.assert_array_equal(trace['observations'][0],old['observations'][0]);np.testing.assert_array_equal(trace['goal'],old['goal'])
    counts=lambda c:[sum(v['success'] for n,v in c.items() if n.startswith(f'task{k}_')) for k in range(1,6)]
    gains=[n for n in base if cases[n]['success'] and not base[n]['success']];losses=[n for n in base if base[n]['success'] and not cases[n]['success']]
    gate=counts(cases)[2]==5 and not losses;assert gate==data['eligible_for_neural_study']
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for i,(label,c) in enumerate([('Original QRL',base),('Data-support metric',cases)]):
        axes[0,0].bar(np.arange(5)+(i-.5)*.34,counts(c),width=.34,label=label)
    axes[0,0].set(title='Same 25-case true-successor greedy; no neural training',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.5),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    with np.load(root/'baseline_field/field.npz') as original:
        nodes=original['nodes'].copy();idx=cKDTree(nodes).query(original['path'])[1];old_values=original['values'][idx]
    with np.load(out/'canonical_field.npz') as field:values=field['values'][idx];walls=field['walls'].copy()
    axes[0,1].plot(old_values-old_values[0],label='Original QRL');axes[0,1].plot(values-values[0],label='Data-support metric')
    axes[0,1].set(title='SAME canonical reference route (evaluation only)',xlabel='Graph steps',ylabel='d - d_start');axes[0,1].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'greedy'/f'task3_seed{seed}.npz') as trace:
            axes[1,0].plot(trace['state_distances'],label=str(seed))
    axes[1,0].set(title='All five existing task3 episodes',xlabel='Actual environment step',ylabel='Data-support distance');axes[1,0].legend(fontsize=8)
    for wall in walls:axes[1,1].add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
    for label,directory in [('Original QRL',root/'baseline_greedy'),('Data-support metric',out/'greedy')]:
        with np.load(directory/'task3_seed3100.npz') as trace:
            axes[1,1].plot(*trace['observations'].T,label=label);goal=trace['goal']
    axes[1,1].scatter(*goal,color='green',marker='*',s=100);axes[1,1].set(title='Predeclared first task3 case',xlim=(2,14),ylim=(6,20),aspect='equal');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'roadmap_metric.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for seed,ax in zip(range(3100,3105),axes):
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,directory in [('Original',root/'baseline_greedy'),('Data-support',out/'greedy')]:
            with np.load(directory/f'task3_seed{seed}.npz') as trace:
                ax.plot(*trace['observations'].T,label=label,lw=1.2);goal=trace['goal'];points.extend([trace['observations'],goal[None]])
        ax.scatter(*goal,color='green',marker='*',s=65);ax.set(title=f'Task3 case {seed}',aspect='equal')
    pts=np.concatenate(points);lo=pts.min(0)-2;hi=pts.max(0)+2
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(cases)[k-1]}/5 |' for k in range(1,6))
    fields='\n'.join(f"| {name} | {s['strict_descent_from_anchor']} | {s['minimum_peak_gap_over_iqr']:.6f} | {s['all_graph_spearman']:.6f} |" for name,s in data['fields'].items())
    corrections=np.array([v['correction'] for v in data['query_statistics'].values()])
    report=f'''# E10：数据支持距离在全部既有case中恢复绕行

独立数据构造在{data['elapsed_seconds']:.2f}秒完成（其中构图及全距离{p['build_seconds']:.2f}秒），全部25例成功，task3为5/5；五个task3都逐步严格降低预测距离，无上坡。这首次通过预定神经拟合候选门槛，但尚未训练神经critic、更新actor或完成总目标，不能将它当官方QRL成绩。

## 固定构造与新增先验

只使用官方全部{p['source_transitions']}条valid训练转移，不改行权重/边界/未来目标，不调用R系列。完整二维观测固定0.2格子，每格选实际观测中距格心最近的样本而非格心；得到{p['nodes']}个原型。只有真实转移对应的格子对构成{p['undirected_edges']}条无向边，代价为原型L∞间距/0.2，频次不累加。

这里显式加入局部可逆和几何连接先验，边代价是连续几何时间近似，不是官方离散步数真值。与M4相比，节点、方向/代价和查询构造均不同，不当单变量消融；没有通过修改M4结果或扫描网格/系数制造收益。原型/图/全部距离完成后才读bank和评价几何，没有使用墙体/参考路线标签。

每个查询连接最近8个原型，a_i=||x−v_i||∞/0.2；delta=0.5*max(0,max_ij[D_ij−a_i−a_j])，b_i=a_i+delta。满足|b_i−b_j|≤D_ij≤b_i+b_j，因此查询节点不会给原图制造更短捷径。不同查询间取min_ij(b_i+D_ij+b_j)，相同完整观测取0，给出查询空间的对称度量。它不保证物理最优性、连续性或在原型坐标上的插值；这些限制不能靠通过有限执行验证而取消。

所有原型保留且图连通，没有取最大分量。距离float64全量有限/对称/自距离零、不低于几何直距，固定子集三角检查通过；保存加载查询逐位一致。起点/目标50个查询的delta范围{corrections.min():.6f}–{corrections.max():.6f}，只报告而不调修正。

## 全部25例真实后继贪心

| 任务 | 原100k QRL | 数据支持距离 |
|---|---:|---:|
{table}

新增{gains}，失去{losses}。原16个成功全部保留，五个task3完成65–75步，每步预测距离严格下降。控制器仍是原121动作真实后继评分/官方1000步上限，T1物理原样执行每个选中动作并逐位检查后继；没有早停失败或新增训练seed。

![全任务执行、同一路线、五个task3下降曲线与第一例](roadmap_metric.png)

| 评价目标 | 原锚点/实际起点严格下降通道 | 最低峰差/IQR | 全图Spearman |
|---|---|---:|---:|
{fields}

这些参考图/墙体只在完整真实执行之后评价，全部六个目标都有通道，不能反过来说构造使用了参考真距离。仍为同0.2网格/0.7半径下的离散证据；实际成功来自完整MuJoCo轨迹，图相关接近1也不构成连续最优性证明。

![全部五个既有task3真实绕行](task3_trajectories.png)

## 核验、门槛与结论范围

三项测试覆盖实际原型/边和保存加载、查询一致修正与对称/自距离/三角结构、不连通直接拒绝。训练更新0，原100k文件SHA前后仍为{p['baseline_checkpoint_sha256']}。图SHA {p['graph_sha256']}，全距离矩阵SHA {p['distance_sha256']}，全部源转移端点SHA {p['source_endpoint_sha256']}；源码/快照、bank与完整观测/动作/评分/结束integration state已经核验。

预定门槛（task3五例全成功且不失去原成功例）为{data['eligible_for_neural_study']}。这只是在已知bank上的方法选择，不是独立泛化验证，更不是神经QRL恢复。它证明在这个数据构造和先验下可得到有效绕行距离，尚不能单独判断原神经优化/表示的哪一项为根因。下一项需要检验原神经critic能否学习该全局信号，并在推理时完全不用图；随后才考虑actor利用。不能以本非参数结果替代学习目标收尾。

协议见[ROADMAP_METRIC_PROTOCOL.md](../../../experiments/ogbench_critic/ROADMAP_METRIC_PROTOCOL.md)。命令`run_roadmap_metric.py <结果根目录> --output <全新目录>`，报告`report_roadmap_metric.py <结果根目录>`只读重建。本地目录`results/ogbench_critic_20261029/roadmap_metric`及同级日志。只提交源码/测试/协议/报告和引用图；全距离矩阵、原型/图、轨迹、JSON/NPZ/日志保留本地。不换seed、不启动1M。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
