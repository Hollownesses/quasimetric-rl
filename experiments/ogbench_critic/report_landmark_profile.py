"""File-only fixed landmark representation and all-failed-endpoint report."""
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
    root=Path(root).resolve();out=root/'landmark_profile';data=read(out/'analysis.json');p=data['protocol']
    assert data['complete'] and data['elapsed_seconds']<120 and p['training_updates']==0
    assert not (out/'failure.json').exists() and p['landmarks']==128
    assert sha(out/'profile.npz')==p['profile_sha256']
    assert sha(root/'roadmap_metric/analysis.json')==p['teacher_analysis_sha256']
    assert sha(root/'roadmap_metric/graph.npz')==p['teacher_graph_sha256'] and sha(root/'roadmap_metric/distances.npy')==p['teacher_distance_sha256']
    assert sha(root/'evaluation_bank.npz')==p['bank_sha256']
    for name,digest in p['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['fast_pointmaze.py','core.py'] else Path(__file__).parent
        assert sha(directory/name)==digest
    teacher=read(root/'roadmap_metric/analysis.json')['cases']
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
    for i,(label,c) in enumerate([('Full teacher',teacher),('Landmark profile',cases)]):
        axes[0,0].bar(np.arange(5)+(i-.5)*.34,counts(c),width=.34,label=label)
    axes[0,0].set(title='Same 25-case true-successor greedy; no neural training',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.5),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    with np.load(root/'baseline_field/field.npz') as original:
        nodes=original['nodes'].copy();idx=cKDTree(nodes).query(original['path'])[1];old_values=original['values'][idx]
    with np.load(out/'canonical_field.npz') as field:values=field['values'][idx];walls=field['walls'].copy()
    axes[0,1].plot(old_values-old_values[0],label='Original QRL');axes[0,1].plot(values-values[0],label='Landmark profile')
    axes[0,1].set(title='SAME canonical reference route (evaluation only)',xlabel='Graph steps',ylabel='d - d_start');axes[0,1].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'greedy'/f'task3_seed{seed}.npz') as trace:
            axes[1,0].plot(trace['state_distances'],label=str(seed))
    axes[1,0].set(title='All five existing task3 episodes',xlabel='Actual environment step',ylabel='Landmark-profile distance');axes[1,0].legend(fontsize=8)
    for wall in walls:axes[1,1].add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
    for label,directory in [('Original QRL',root/'baseline_greedy'),('Landmark profile',out/'greedy')]:
        with np.load(directory/'task3_seed3100.npz') as trace:
            axes[1,1].plot(*trace['observations'].T,label=label);goal=trace['goal']
    axes[1,1].scatter(*goal,color='green',marker='*',s=100);axes[1,1].set(title='Predeclared first task3 case',xlim=(2,14),ylim=(6,20),aspect='equal');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'landmark_profile.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for seed,ax in zip(range(3100,3105),axes):
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,directory in [('Teacher',root/'roadmap_metric/greedy'),('Profile',out/'greedy')]:
            with np.load(directory/f'task3_seed{seed}.npz') as trace:
                ax.plot(*trace['observations'].T,label=label,lw=1.2);goal=trace['goal'];points.extend([trace['observations'],goal[None]])
        ax.scatter(*goal,color='green',marker='*',s=65);ax.set(title=f'Task3 case {seed}',aspect='equal')
    pts=np.concatenate(points);lo=pts.min(0)-2;hi=pts.max(0)+2
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(cases)[k-1]}/5 |' for k in range(1,6))
    fields='\n'.join(f"| {name} | {s['strict_descent_from_anchor']} | {s['minimum_peak_gap_over_iqr']:.6f} | {s['all_graph_spearman']:.6f} |" for name,s in data['fields'].items())
    probe=read(out/'endpoint_probe/analysis.json');assert probe['post_hoc'] and probe['new_episodes']==0 and probe['seconds']<30
    assert probe['source_sha256']==sha(Path(__file__).with_name('probe_landmark_profile.py'))
    assert set(probe['probes'])=={name for name,v in cases.items() if not v['success']}
    endpoint_table='\n'.join(f"| {name} | {v['teacher_minus_profile']:.3f} | {v['teacher_improvement_best']:.3f} | {v['profile_improvement_best']:.3f} | {v['teacher_best_displacement']:.3f} / {v['profile_best_displacement']:.3f} |" for name,v in probe['probes'].items())
    report=f'''# E14：128地标距离坐标未保留教师的真实执行能力

固定表示在{data['elapsed_seconds']:.2f}秒内完成全部执行及距离场，训练更新0；成功{sum(counts(cases))}/25、task3 {counts(cases)[2]}/5，完整教师25/25。本表示未通过预定门槛，不启动其神经训练，也不扫描地标数量/范数/排列挽救。有限表示满足三角性且大尺度相关性高，仍会产生实际停滞，不能把这些条件当执行充分保证。

## 固定表示和核验

只读E10既有6302原型/全距离，最远点遍历固定选128地标，从第0个开始、argmax并列取最小索引；完成并保存后才读bank。原始图覆盖半径{p['raw_graph_covering_radius']:.5f}，不是连续物理覆盖保证。h_l(x)=完整教师t(x,landmark_l)，d_profile(x,g)=max_l|h_l(x)−h_l(g)|。这是拉回L∞伪度量，有限地标不保证单射/等距，三角性给出d_profile≤t但不给出正确局部下降。

原E10的局部可逆/几何连接先验保留，非真实MDP最优步数；没有评价墙体/路线标签，没有新神经网络/原QRL参数更新。到地标的连接代价预计算仍来自同教师8连接查询，浮点float64。两项测试覆盖直接教师查询一致、确定选择、自距离/对称/三角/教师下界、保存加载及非单射反例；实际完整图上32个确定偏移点×128地标与直接教师查询一致，保存加载逐位相同。表示构造/核验{p['build_seconds']:.2f}秒，总120秒上限未超。

## 全任务成功集合

| 任务 | 原100k QRL | 128地标表示 |
|---|---:|---:|
{table}

原QRL新增{gains}，失去{losses}；完整教师的25成功失去9例。task3虽为3/5，但失去3101/3103，并丢失task1/2/5的原成功，不是稳定改善。仍使用全部固定25例、121真实后继、1000完整步/成功终止及T1所选后继逐位检查；不对停滞早停，没有重复原控制。

![全任务、同参考路线、五个task3距离和首例轨迹](landmark_profile.png)

| 评价目标 | 起点严格下降通道 | 峰差/IQR | 全图Spearman |
|---|---|---:|---:|
{fields}

全部六个起点均有离散参考通道，实际仍9个失败。两个task3失败终点附近反而没有严格通道，峰差/IQR约0.01720/0.02625，吸附误差0.17065/0.01750。图是0.2离散化、半径0.7，不能把起点图证据推广成所有实际状态均有下降动作；真实执行结果优先。

![全部五个task3教师与地标表示轨迹](task3_trajectories.png)

## 事后全部失败终点候选检查

完整执行后另用{probe['seconds']:.2f}秒（30秒独立上限），只读九个失败终点、每点121个实际后继，无新episode。没有挑一个角点代替全部失败。

| 失败case | 教师−profile当前差 | 教师最佳下降 | profile最佳下降 | 教师/profile选择的位移L∞ |
|---|---:|---:|---:|---:|
{endpoint_table}

九处profile最佳下降均为0，选择位移近零；同候选下完整教师仍有0.2位移且下降1.01–3.46。教师的最佳动作在profile下反而比当前更差，故是距离排序/平台问题，非动作枚举或神经拟合误差。活跃地标仅2–6个，完整教师与投影当前数值差1.54–6.75；差值虽不大，也能改变局部动作方向。不能据这些数据证明增加地标就能解决，协议禁止数量搜索。

## 决策与复现

预定门槛{data['eligible_for_neural_study']}，停止该表示的神经训练。也不能据此断言所有距离坐标表示无效或IQE容量不足；本项没有训练任何网络。当前可用学习改进仍是E12/E13的23/25、task3 3/5，actor未变、总目标未完成。下一项需要先核验教师查询扩展与连续神经表示之间的可拟合性，避免将抽象查询代价伪影误认作真实MDP最优监督；不重做数据覆盖审计或扩大训练。

协议[LANDMARK_PROFILE_PROTOCOL.md](../../../experiments/ogbench_critic/LANDMARK_PROFILE_PROTOCOL.md)，入口run_landmark_profile.py <根目录> --output <全新目录>；事后probe_landmark_profile.py，报告report_landmark_profile.py只读重建。raw results/ogbench_critic_20261029/landmark_profile，表示profile.npz SHA {p['profile_sha256']}，教师图/距离SHA {p['teacher_graph_sha256']} / {p['teacher_distance_sha256']}。原100k文件SHA前后{p['baseline_checkpoint_sha256']}未变，源码快照/同bank/全部轨迹与结束integration state已核验。只提交源码、测试、协议、本报告和引用图，表示/JSON/NPZ/日志不入库。不换seed、不运行1M。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
