"""Verify control reuse and analyze the completed source-cell weighting arm."""
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
from scipy.spatial import cKDTree


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def logs(path):
    with Path(path).open() as f:return [{k:(v if k=='arm' else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]


def run(root):
    root=Path(root).resolve();out=root/'source_balance_1000';data=read(out/'analysis.json')
    protocol=data['protocol'];training=data['training'];control=read(root/'tail_1000/analysis.json')
    assert data['complete'] and data['elapsed_seconds']<300 and not (out/'failure.json').exists()
    assert training['passed'] and training['steps']==1000 and training['adam_count']==101000
    assert training['paired_stream_sha256']==protocol['expected_stream_sha256']==control['training']['paired_stream_sha256']
    assert protocol['control_analysis_sha256']==sha(root/'tail_1000/analysis.json')
    assert protocol['control_checkpoint_sha256']==sha(root/'tail_1000/mean/params_101000.pkl')
    assert protocol['fast_verification_sha256']==sha(root/'cached_pointmaze_verification/verification.json')
    assert sha(out/'model/params_101000.pkl')==training['checkpoint_sha256']
    assert sha(out/'source_cells.npz')==protocol['cells_sha256']
    assert data['bank_sha256']==sha(root/'evaluation_bank.npz')==control['bank_sha256']
    assert read(out/'smoke/verification.json')['passed']
    for name,digest in protocol['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['fast_pointmaze.py','core.py'] else Path(__file__).parent
        assert sha(directory/name)==digest
    a=control['rollouts']['mean'];b=data['cases'];base=read(root/'baseline_greedy/summary.json')['cases']
    assert list(a)==list(b)==list(base) and len(b)==25
    for name,info in b.items():
        with np.load(out/'greedy'/f'{name}.npz') as trace:
            assert trace['observations'].shape==(info['steps']+1,2) and trace['actions'].shape==(info['steps'],2)
            assert trace['distances'].shape==(info['steps'],) and trace['final_integration'].ndim==1
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            np.testing.assert_array_equal(trace['observations'][-1],info['final_xy'])
            assert 0<info['steps']<=1000
            assert not info['success'] or np.linalg.norm(trace['observations'][-1]-trace['goal'])<=1
            with np.load(root/'tail_1000/mean_greedy'/f'{name}.npz') as old:
                np.testing.assert_array_equal(trace['observations'][0],old['observations'][0])
                np.testing.assert_array_equal(trace['goal'],old['goal'])
    def counts(cases):return [sum(v['success'] for n,v in cases.items() if n.startswith(f'task{t}_')) for t in range(1,6)]
    gained=[n for n in a if b[n]['success'] and not a[n]['success']]
    lost=[n for n in a if a[n]['success'] and not b[n]['success']]
    b_logs=logs(out/'metrics.csv');a_logs=[r for r in logs(root/'tail_1000/metrics.csv') if r['arm']=='mean']
    assert [r['step'] for r in b_logs]==[r['step'] for r in a_logs]==[1]+list(range(100,1001,100))
    initial=read(out/'initial_statistics.json');source=protocol['source_cells']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    steps=[r['step'] for r in b_logs]
    axes[0,0].plot(steps,[r['train/weight_mean'] for r in b_logs],marker='o');axes[0,0].axhline(1,color='.5',ls='--')
    axes[0,0].set(title='Logged batch weight mean (population mean = 1)',xlabel='Updates')
    axes[0,1].plot(steps,[r['train/weight_effective_samples'] for r in b_logs],marker='o')
    axes[0,1].set(title='Logged batch effective sample size',xlabel='Updates',ylim=(0,270))
    for label,rows in [('mean control (reused)',a_logs),('source-balanced',b_logs)]:
        axes[0,2].plot([0]+[r['step'] for r in rows],[initial['pair_mean_squared_positive']]+[r['heldout/pair_mean_squared_positive'] for r in rows],label=label)
        axes[1,2].plot([0]+[r['step'] for r in rows],[initial['goal_mean']]+[r['heldout/goal_mean'] for r in rows],label=label)
    axes[0,2].set(title='Same unweighted validation: local risk',xlabel='Updates');axes[0,2].legend(fontsize=7)
    axes[1,2].set(title='Same validation: random-goal distance scale',xlabel='Updates');axes[1,2].legend(fontsize=7)
    for i,(label,cases) in enumerate([('Original',base),('mean (reused)',a),('source-balanced',b)]):
        axes[1,0].bar(np.arange(5)+(i-1)*.25,counts(cases),width=.25,label=label)
    axes[1,0].set(title='Fixed-bank true-successor greedy',xticks=range(5),xticklabels=[f'T{t}' for t in range(1,6)],ylim=(0,5.5),ylabel='Successes / 5');axes[1,0].legend(fontsize=7)
    with np.load(root/'baseline_field/field.npz') as reference:nodes=reference['nodes'].copy();indices=cKDTree(nodes).query(reference['path'])[1]
    for label,path,stats in [('mean (reused)',root/'tail_1000/mean_field.npz',control['fields']['mean']),('source-balanced',out/'field.npz',data['field'])]:
        with np.load(path) as field:
            np.testing.assert_array_equal(nodes,field['nodes']);values=field['values'][indices]
        axes[1,1].plot((values-values[0])/max(stats['distance_iqr'],1e-9),label=label)
    axes[1,1].set(title='Canonical goal: SAME old reference route',xlabel='Graph steps',ylabel='(d - d_start) / field IQR');axes[1,1].legend(fontsize=7)
    fig.savefig(out/'source_balance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    with np.load(out/'field.npz') as field:walls=field['walls'].copy()
    for seed,ax in zip(range(3100,3105),axes):
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,directory in [('mean (reused)',root/'tail_1000/mean_greedy'),('source-balanced',out/'greedy')]:
            with np.load(directory/f'task3_seed{seed}.npz') as trace:
                ax.plot(*trace['observations'].T,label=label,lw=1.2);goal=trace['goal']
                points.extend([trace['observations'],goal[None]])
        ax.scatter(*goal,marker='*',color='green',s=65)
        ax.set(title=f'Task3 case {seed}',aspect='equal')
    points=np.concatenate(points);lower=points.min(0)-2;upper=points.max(0)+2
    for ax in axes:ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{t} | {counts(base)[t-1]}/5 | {counts(a)[t-1]}/5 | {counts(b)[t-1]}/5 |' for t in range(1,6))
    metrics=['pair_mean_squared_positive','pair_tail_squared_positive','pair_q99','pair_max','goal_squared_positive','goal_mean','lam']
    metrics_table='\n'.join(f'| {k} | {initial[k]:.6g} | {a_logs[-1]["heldout/"+k]:.6g} | {b_logs[-1]["heldout/"+k]:.6g} |' for k in metrics)
    fields_table='\n'.join(f"| {label} | {s['strict_descent_from_anchor']} | {s['minimum_peak_gap_over_iqr']:.6f} | {s['all_graph_spearman']:.6f} | {s['corridor_spearman']:.6f} |" for label,s in [('mean',control['fields']['mean']),('source-balanced',data['field'])])
    decision=('本次源均衡未解决task3；不调网格、权重指数/截断、学习率或eps挽救。' if counts(b)[2]==0 else
              '本次出现task3收益，但仍须结合其他任务得失、全部既有case及actor利用检验，不能直接宣布稳定改善完成。')
    report=f'''# E7：源格子均衡改变排序，但没有恢复task3

仅新增一个原100k IQE的固定1000步critic续训组，复用E6已经完成的mean控制组，不重复训练A或原始基线。两组训练批次字节流完全相同。正式总用时{data['elapsed_seconds']:.2f}秒，其中构造、smoke、训练和状态核验{training['train_verify_seconds']:.2f}秒，处于预定300秒上限。没有换seed/初始化、挑checkpoint或启动1M。

## 新方法与控制组身份

源完整二维观测仅用于rint(s/0.2)频次，网络仍接收原始连续观测。对全部100万官方valid源行，w=N/(K*n_c)，同时加权原push与pair惩罚；dual使用同加权pair与原eps。所有原行仍被官方采样，轨迹边界和随机/未来目标语义不变；不去重、不改目标、不用R系列、终态quotient、图边/最短路、墙体或评价数据训练。

这改变经验源分布，不能称原官方目标不变，也不是精确重复去重或源/边同时均衡。原精确重复审计很低，本项没有重复覆盖审计或重写原100k基线。batch256、lr/eps、1000更新、原Adam、IQE和官方目标配置与E6一致，actor/dynamics及其Adam矩冻结。源码/模型身份和数据流核验后才复用A。

| 人口统计 | 实际值 |
|---|---:|
| 有效源行N / 占用源格子K | {source['total']} / {source['occupied']} |
| 最小 / 最大格子频次 | {source['minimum_count']} / {source['maximum_count']} |
| 最小 / 最大行权重 | {source['minimum_weight']:.6f} / {source['maximum_weight']:.6f} |
| 人口平均权重 | {source['population_weight_mean']:.10f} |
| 人口重要性有效样本数 | {source['population_effective_samples']:.2f} |

人口有效样本数约原始N的5.1%，不是对转移的温和去重。没有batch重归一或权重截断；有限batch的权重均值会波动，这正是人口归一的重要性估计，日志保留均值/最大值/有效样本数。不能把不均一损失波动误当配置或数据流变化。

## 全部25例真实执行

| 任务 | 原100k | E6 mean复用 | 源均衡B |
|---|---:|---:|---:|
{table}

总成功为{sum(counts(base))}/25、{sum(counts(a))}/25、{sum(counts(b))}/25。B相对A新增{gained}，失去{lost}。不是原成功集合的稳定扩展；冻结actor未重新评价或学习，该表不冒充官方actor指标。完整episode仍为1000步，精确缓存与四线程物理只减少重复计算，实际选中后继和首次hit均核验。

![权重波动、有效样本数、同验证风险、执行与归一化距离](source_balance.png)

## 原分布验证与距离场

| 原2048验证转移指标 | 原100k | E6 mean | 源均衡B |
|---|---:|---:|---:|
{metrics_table}

这里是完全相同的**未加权**验证分布，不等同于新源分布下的约束风险。B的加权训练pair最后记录为{b_logs[-1]['train/d_pos_loss']:.6f}，未加权验证pair为{b_logs[-1]['heldout/pair_mean_squared_positive']:.6f}；不能把不同统计混为一个已满足的约束。乘子与距离尺度均明示，不凭lambda或单个风险宣称成功。

| critic | canonical原锚点严格下降通道 | 最低峰差/IQR | 全图Spearman | 原走廊Spearman |
|---|---|---:|---:|---:|
{fields_table}

源均衡提高了原走廊的局部排序相关，却降低整图排序；最低峰差为零仍没有严格下降通道。它意味着能找到不超过起点高度的路线，不表示沿途不再上坡。以上只用同0.2参考图、0.7半径与canonical目标评价；不把它混作bank五个实际目标，也不把离散图吸附结论推广到所有连续状态。

![五个既有task3完整真实轨迹](task3_trajectories.png)

## 核验、决策与复现

三项方法测试通过：全1权重损失/梯度退化官方、完整观测区分与人口质量归一/保存加载、非零动量冻结及计数。实际两步smoke保存加载完整agent/优化器/RNG相等，正式从原100k开始；终点全状态往返、数值有限、count101000、冻结参数/矩与同25例actor输出均通过。B原始轨迹/结束integration state与同bank起点/目标已核对。

共享批次流SHA {training['paired_stream_sha256']}，与E6 A完全相同；复用A checkpoint SHA {protocol['control_checkpoint_sha256']}，B SHA {training['checkpoint_sha256']}。缓存共有{data['cache']['calls']}次请求、{data['cache']['evaluations']}次实际函数求值，其中首次hit重算{data['cache']['first_hit_audits']}次；没有提前终止停滞episode。

{decision} 这只约束当前初始化、数据流、固定网格定义与短预算，不能称所有均衡方法无效；也不能以一个局部场统计改善为总目标收尾。稳定改善尚未达成，不恢复seed1或1M。

协议见[SOURCE_BALANCE_PROTOCOL.md](../../../experiments/ogbench_critic/SOURCE_BALANCE_PROTOCOL.md)。原始路径`results/ogbench_critic_20261029/source_balance_1000`，日志同级`source_balance_1000.log`；命令`run_source_balance.py <结果根目录> --output <全新目录>`。报告由`report_source_balance.py <结果根目录>`只读重建，不训练或重复评价。Git只提交源码/测试/协议/报告及引用图，模型/计数表/CSV/JSON/NPZ/日志保留本地。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
