"""Read-only reporting of the predeclared frozen symmetry intervention."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    root=Path(root).resolve();out=root/'symmetry_projection';data=read(out/'analysis.json');p=data['protocol']
    assert data['complete'] and data['elapsed_seconds']<240 and not (out/'failure.json').exists()
    assert p['training_updates']==0 and p['epoch']==100000 and p['coefficient']==.5 and p['primary']=='symmetric'
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    assert sha(root/'evaluation_bank.npz')==p['bank_sha256']
    assert sha(root/'baseline_greedy/summary.json')==p['baseline_summary_sha256']
    assert sha(root/'cached_pointmaze_verification/verification.json')==p['fast_verification_sha256']
    for name,digest in p['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['fast_pointmaze.py','core.py'] else Path(__file__).parent
        assert sha(directory/name)==digest
    base=read(root/'baseline_greedy/summary.json')['cases'];cases=data['cases']
    assert list(base)==list(cases) and len(cases)==25
    for name,info in cases.items():
        with np.load(out/'greedy'/f'{name}.npz') as trace:
            assert trace['observations'].shape==(info['steps']+1,2) and trace['actions'].shape==(info['steps'],2)
            assert trace['distances'].shape==(info['steps'],) and trace['final_integration'].ndim==1
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            np.testing.assert_array_equal(trace['observations'][-1],info['final_xy'])
            assert 0<info['steps']<=1000
            assert not info['success'] or np.linalg.norm(trace['observations'][-1]-trace['goal'])<=1
            with np.load(root/'baseline_greedy'/f'{name}.npz') as old:
                np.testing.assert_array_equal(trace['observations'][0],old['observations'][0])
                np.testing.assert_array_equal(trace['goal'],old['goal'])
    for name in data['fields']:
        with np.load(out/f'{name}_field.npz') as field:
            np.testing.assert_allclose(field['forward'],field['symmetric']+field['antisymmetric'],rtol=1e-6,atol=1e-5)
            np.testing.assert_array_equal(field['symmetric'],.5*(field['forward']+field['reverse']))
    delta=data['decomposition']['delta']
    np.testing.assert_allclose(delta['forward'],delta['symmetric']+delta['antisymmetric'],rtol=1e-6,atol=1e-5)
    counts=lambda c:[sum(v['success'] for n,v in c.items() if n.startswith(f'task{k}_')) for k in range(1,6)]
    gains=[n for n in base if cases[n]['success'] and not base[n]['success']]
    losses=[n for n in base if base[n]['success'] and not cases[n]['success']]
    validation=data['validation'];assert validation['symmetric_push']<=validation['paired_push']+1e-10
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for i,(label,c) in enumerate([('Original forward',base),('Symmetric',cases)]):
        axes[0,0].bar(np.arange(5)+(i-.5)*.34,counts(c),width=.34,label=label)
    axes[0,0].set(title='Fixed-bank true-successor greedy; no learning',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.5),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    with np.load(out/'canonical_route.npz') as route:
        scale=data['fields']['canonical']['forward']['distance_iqr']
        for kind in ['forward','reverse','symmetric','antisymmetric']:
            values=route[kind];axes[0,1].plot((values-values[0])/scale,label=kind)
    axes[0,1].set(title='SAME old route; common forward IQR denominator',xlabel='Graph steps',ylabel='Relative value / forward IQR');axes[0,1].legend(fontsize=8)
    axes[1,0].bar(['Forward','Reverse','Symmetric'],[validation['forward_pair_risk'],validation['reverse_pair_risk'],validation['symmetric_pair_risk']])
    axes[1,0].axhline(.05,color='.5',ls='--',label='Original eps');axes[1,0].set(title='Same validation: directed local transition risk');axes[1,0].legend(fontsize=8)
    with np.load(out/'canonical_route.npz') as route:
        for kind in ['forward','symmetric','antisymmetric']:
            axes[1,1].plot(route[kind][:16]-route[kind][0],label=kind)
    axes[1,1].set(title='Fixed probe 0 -> 15: forward = symmetric + antisymmetric',xlabel='Graph steps',ylabel='Distance change');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'symmetry.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    with np.load(out/'canonical_field.npz') as field:walls=field['walls'].copy()
    for seed,ax in zip(range(3100,3105),axes):
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,directory in [('Forward',root/'baseline_greedy'),('Symmetric',out/'greedy')]:
            with np.load(directory/f'task3_seed{seed}.npz') as trace:
                ax.plot(*trace['observations'].T,label=label,lw=1.2);goal=trace['goal'];points.extend([trace['observations'],goal[None]])
        ax.scatter(*goal,marker='*',color='green',s=65);ax.set(title=f'Task3 case {seed}',aspect='equal')
    pts=np.concatenate(points);lo=pts.min(0)-2;hi=pts.max(0)+2
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(cases)[k-1]}/5 |' for k in range(1,6))
    field_table='\n'.join(f"| {kind} | {v['strict_descent_from_anchor']} | {v['minimum_peak_gap_over_iqr']:.6f} | {v['all_graph_spearman']:.6f} | {v['corridor_spearman']:.6f} |" for kind,v in data['fields']['canonical'].items())
    actual_table='\n'.join(f"| {name} | {v['forward']['strict_descent_from_anchor']} | {v['reverse']['strict_descent_from_anchor']} | {v['symmetric']['strict_descent_from_anchor']} | {data['failure_endpoints'][name]['minimum_peak_gap_over_iqr']:.6f} |" for name,v in data['fields'].items() if name!='canonical')
    report=f'''# E9：方向不对称解释部分上坡，等权对称仍未解决task3

固定原100k QRL，只读构造S(s,g)=(d(s,g)+d(g,s))/2，训练更新为0。全部25例真实后继贪心由16/25增至19/25，新增三个task4成功且无失去，但task3仍0/5。不能把原失败单独归因于方向不对称，也不能把S包装为官方基线或原actor改进。

## 固定投影与数学范围

F=d(s,g)，R=d(g,s)，S=(F+R)/2，A=(F−R)/2。S是本轮唯一主要critic，R仅作场/分解，没有试跑反向控制或扫描混合系数。准度量及其反向的等权和仍满足准度量结构并对称，A为有符号分量，不当距离评价。点maze的反向运动结构提供研究动机，但未断言接触/初始化/完整MuJoCo成本严格对称，不能推广到Dubins等方向性任务。

原push为凸的100 softplus(5−d/100)。源与随机value goal同分布独立时，交换源/目标不改变总体分布；配对有限样本也满足push(S)≤(push(F)+push(R))/2。原2048验证上配对push为{validation['paired_push']:.9f}，S为{validation['symmetric_push']:.9f}，仅改善约{validation['paired_push']-validation['symmetric_push']:.9f}。未配对F/R分别为{validation['unpaired_forward_push']:.9f}/{validation['unpaired_reverse_push']:.9f}，不能把配对Jensen不等式错误地当成任一有限未配对样本上的保证。

随机目标平均绝对F−R差{validation['mean_absolute_asymmetry']:.6f}，平均相对差约{100*validation['mean_relative_asymmetry']:.3f}%。原转移pair风险F/R/S为{validation['forward_pair_risk']:.6f}/{validation['reverse_pair_risk']:.6f}/{validation['symmetric_pair_risk']:.6f}。S虽低于配对风险均值{validation['paired_pair_risk']:.6f}，仍高于F且超过原eps0.05；没有声称该投影满足原已训练单向风险约束，反向转移也不直接当真单步标签。

## 同25例真实执行

| 任务 | 原F | 等权S |
|---|---:|---:|
{table}

新增{gains}，失去{losses}。全部既有case完成原121动作网格、官方1000步上限，失败episode没有按停滞早停。原F直接复用旧数组；S明确使用两个方向的原函数评分，权重/参数未改变，actor没有更新或重新学习。

## canonical场的固定0/15分解

| critic | 原锚点严格下降通道 | 最低峰差/IQR | 全图Spearman | 原走廊Spearman |
|---|---|---:|---:|---:|
{field_table}

固定旧路径节点{data['decomposition']['probe_nodes']}间，ΔF={delta['forward']:.6f}，ΔR={delta['reverse']:.6f}，ΔS={delta['symmetric']:.6f}，ΔA={delta['antisymmetric']:.6f}。ΔF=ΔS+ΔA在FP32误差内成立；反对称部分约占原上坡35.8%，但剩余对称上坡仍约4.26。这个分解说明不对称参与了该模型的上坡，并不证明训练历史的唯一因果；F、R、S均没有原锚点下降通道。

![同库执行、同路线分解、局部风险与固定上坡probe](symmetry.png)

## 五个实际task3目标与失败终点

| 既有case | 起点F通道 | 起点R通道 | 起点S通道 | S终点附近峰差/IQR |
|---|---|---|---|---:|
{actual_table}

三个方向场在全部五个起点均无严格下降通道；S五个真实失败终点附近也无通道，终点附近峰差/IQR约0.019–0.148。终点吸附误差最大约0.206，保留同0.2网格/0.7碰撞半径的离散限制，不能将附近检查当连续状态最优性证明。canonical目标与实际五个目标分开计算，未用一个好看的目标代表全部执行。

![全部五个既有task3真实轨迹](task3_trajectories.png)

## 核验与决策

三项测试通过：等权定义/分解、对称/自距离/三角结构、配对凸性及缓存评分等于原函数双方向。完整agent摘要和checkpoint SHA前后不变，source/模型/bank摘要、字段代数与轨迹起点/目标/评分/结束integration state核验通过。整个诊断{data['elapsed_seconds']:.2f}秒，低于240秒上限；精确缓存{data['cache']['calls']}次原方向评分请求，实际求值{data['cache']['evaluations']}次（首次hit复算{data['cache']['first_hit_audits']}次），没有训练或新checkpoint。

S在现有bank的task4有收益，但未解决核心task3；不扫描系数、挑R方向或换较好终点挽救本项。当前证据要求保留“对称场本身仍有错误上坡”，不能把方向不对称命名为唯一根因。稳定恢复目标继续，不恢复seed1/1M，不以跨seed或完整预算作为必要门槛。

协议见[SYMMETRY_PROTOCOL.md](../../../experiments/ogbench_critic/SYMMETRY_PROTOCOL.md)。原始路径`results/ogbench_critic_20261029/symmetry_projection`，日志同级`symmetry_projection.log`；入口`run_symmetry.py <结果根目录> --output <全新目录>`。报告由`report_symmetry.py <结果根目录>`只读重建。源码/测试/协议/报告和引用图提交，数组/JSON/日志保留本地。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
