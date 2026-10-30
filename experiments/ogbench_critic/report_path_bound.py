"""Read-only analysis of E8, including the previously completed E6 control."""
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
def rows(path):
    with Path(path).open() as handle:return [{k:(v if k=='arm' else float(v)) for k,v in r.items()} for r in csv.DictReader(handle)]


def run(root):
    root=Path(root).resolve();out=root/'path_bound_1000';data=read(out/'analysis.json');p=data['protocol'];t=data['training']
    control=read(root/'tail_1000/analysis.json');a=control['rollouts']['mean'];b=data['cases']
    assert data['complete'] and data['elapsed_seconds']<300 and not (out/'failure.json').exists()
    assert t['passed'] and t['steps']==1000 and t['adam_count']==101000
    assert t['paired_stream_sha256']==p['expected_stream_sha256']==control['training']['paired_stream_sha256']
    assert p['control_analysis_sha256']==sha(root/'tail_1000/analysis.json')
    assert p['control_checkpoint_sha256']==sha(root/'tail_1000/mean/params_101000.pkl')
    assert p['fast_verification_sha256']==sha(root/'cached_pointmaze_verification/verification.json')
    assert sha(out/'model/params_101000.pkl')==t['checkpoint_sha256']
    assert sha(out/'validation_paths.npz')==p['validation_paths_sha256']
    assert data['bank_sha256']==sha(root/'evaluation_bank.npz')==control['bank_sha256']
    assert read(out/'smoke/verification.json')['passed']
    for name,digest in p['source_sha256'].items():
        assert sha(out/'source_snapshot'/name)==digest
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['fast_pointmaze.py','core.py'] else Path(__file__).parent
        assert sha(directory/name)==digest
    hist=np.asarray(t['path_cost_counts']);assert hist.shape==(65,) and hist.sum()==256000
    with np.load(out/'validation_paths.npz') as targets:
        np.testing.assert_array_equal(targets['end_indices']-targets['indices'],targets['costs'])
        requested=np.array([2,4,8,16,32,64])[np.arange(2048)%6]
        assert (targets['costs']>=0).all() and (targets['costs']<=requested).all()
    assert list(a)==list(b) and len(b)==25
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
    counts=lambda c:[sum(v['success'] for n,v in c.items() if n.startswith(f'task{k}_')) for k in range(1,6)]
    gains=[n for n in a if b[n]['success'] and not a[n]['success']];losses=[n for n in a if a[n]['success'] and not b[n]['success']]
    al=[r for r in rows(root/'tail_1000/metrics.csv') if r['arm']=='mean'];bl=rows(out/'metrics.csv')
    assert [r['step'] for r in al]==[r['step'] for r in bl]==[1]+list(range(100,1001,100))
    initial=read(out/'initial_statistics.json');ip=read(out/'initial_path_statistics.json');ap=read(out/'control_path_statistics.json')
    np.testing.assert_allclose(ap['pair_loss'],al[-1]['heldout/pair_mean_squared_positive'],rtol=1e-6,atol=2e-6)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for label,log in [('mean (reused)',al),('path-bound',bl)]:
        x=[0]+[r['step'] for r in log]
        axes[0,0].plot(x,[initial['pair_mean_squared_positive']]+[r['heldout/pair_mean_squared_positive'] for r in log],label=label)
        axes[0,2].plot(x,[initial['lam']]+[r['heldout/lam'] for r in log],label=label)
    axes[0,0].set(title='Same validation: local pair risk',xlabel='Additional updates');axes[0,0].legend(fontsize=8)
    axes[0,2].set(title='Dual multiplier (different risk definitions)',xlabel='Additional updates');axes[0,2].legend(fontsize=8)
    axes[0,1].plot([0]+[r['step'] for r in bl],[ip['path_loss']]+[r['path_heldout/path_loss'] for r in bl],label='path-bound')
    axes[0,1].scatter([1000],[ap['path_loss']],marker='D',color='tab:blue',label='mean endpoint (only)')
    axes[0,1].set(title='Same validation: extra path upper-bound risk',xlabel='Additional updates',yscale='log');axes[0,1].legend(fontsize=8)
    for i,(label,cases) in enumerate([('mean (reused)',a),('path-bound',b)]):
        axes[1,0].bar(np.arange(5)+(i-.5)*.34,counts(cases),width=.34,label=label)
    axes[1,0].set(title='Fixed-bank true-successor greedy',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.5),ylabel='Successes / 5');axes[1,0].legend(fontsize=8)
    axes[1,1].bar(np.arange(65),hist);axes[1,1].set(title='All 256k auxiliary path lengths (after clamping)',xlabel='Actual valid path length',ylabel='Count')
    with np.load(root/'baseline_field/field.npz') as original:nodes=original['nodes'].copy();idx=cKDTree(nodes).query(original['path'])[1]
    for label,path,stats in [('mean (reused)',root/'tail_1000/mean_field.npz',control['fields']['mean']),('path-bound',out/'field.npz',data['field'])]:
        with np.load(path) as field:
            np.testing.assert_array_equal(nodes,field['nodes']);v=field['values'][idx]
        axes[1,2].plot((v-v[0])/max(stats['distance_iqr'],1e-9),label=label)
    axes[1,2].set(title='Canonical goal: SAME old reference route',xlabel='Graph steps',ylabel='(d - d_start) / field IQR');axes[1,2].legend(fontsize=8)
    fig.savefig(out/'path_bound.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    with np.load(out/'field.npz') as field:walls=field['walls'].copy()
    for seed,ax in zip(range(3100,3105),axes):
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,directory in [('mean (reused)',root/'tail_1000/mean_greedy'),('path-bound',out/'greedy')]:
            with np.load(directory/f'task3_seed{seed}.npz') as trace:
                ax.plot(*trace['observations'].T,label=label,lw=1.2);goal=trace['goal'];points.extend([trace['observations'],goal[None]])
        ax.scatter(*goal,marker='*',color='green',s=65);ax.set(title=f'Task3 case {seed}',aspect='equal')
    pts=np.concatenate(points);lo=pts.min(0)-2;hi=pts.max(0)+2
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=7);fig.savefig(out/'task3_trajectories.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(a)[k-1]}/5 | {counts(b)[k-1]}/5 |' for k in range(1,6))
    keys=['pair_mean_squared_positive','pair_tail_squared_positive','goal_squared_positive','goal_mean','lam']
    risk_table='\n'.join(f'| {k} | {al[-1]["heldout/"+k]:.6g} | {bl[-1]["heldout/"+k]:.6g} |' for k in keys)
    path_table='\n'.join(f'| {k} | {ap[k]:.6g} | {bl[-1]["path_heldout/"+k]:.6g} |' for k in ['path_loss','d_pos_loss','path_active_fraction','path_distance_mean','path_cost_mean','path_zero_fraction'])
    field_table='\n'.join(f"| {label} | {s['strict_descent_from_anchor']} | {s['minimum_peak_gap_over_iqr']:.6f} | {s['all_graph_spearman']:.6f} | {s['corridor_spearman']:.6f} |" for label,s in [('mean',control['fields']['mean']),('path-bound',data['field'])])
    report=f'''# E8：直接多步上界改善task4，task3仍未恢复

新增一个固定1000步原IQE critic续训组，复用E6 mean控制组而不重训/重跑。全部原训练批次流逐字节相同，actor/dynamics参数及Adam矩冻结。25例成功数由{sum(counts(a))}增至{sum(counts(b))}，新增两个task4成功且没有丢失原成功；task3仍0/5，canonical原锚点仍无严格下降通道，因此稳定恢复目标尚未达成。

总用时{data['elapsed_seconds']:.2f}秒，其中载入/smoke/训练/完整状态核验{t['train_verify_seconds']:.2f}秒，低于预定300秒上限。没有延长预算、换seed/初始化、选中间checkpoint或启动1M。

## 方法与可解释范围

B保留原距离最大化/随机目标、原IQE和单位、原Adam/lr/eps。原逐行h1=relu(d(s,s_next)−1)^2，新真实路径hk=relu(d(s,s_k)−k)^2，以mean(max(h1,hk))作为同一dual约束风险。max不削弱原pair项，不引入混合权重、动作编码器、折扣/LINEX、目标网络或图监督。它与E2多步备份是不同实验，单纯多步不当新贡献。

硬满足所有单步上界时，三角不等式本已蕴含路径上界；这里检验软平均局部预算下的累积误差与学习效率。k只是真实行为路径成本上界，未把它当最优距离等式。跨度固定2/4/8/16/32/64轮转，原value/actor目标字段及RNG完全保留。

源索引通过保存/恢复RNG状态恢复原官方采样索引；新路径截到官方first terminal，valid前缀保证跨度内每个源转移有效，保留双terminal和k=0语义。所有256000个辅助路径中，k=0有{hist[0]}个，平均实际长度{float(np.dot(np.arange(65),hist)/hist.sum()):.6f}；原pair仍包括penultimate的真实next，没有删掉这些转移。

## 全部25例真实后继贪心

| 任务 | E6 mean复用 | path-bound |
|---|---:|---:|
{table}

新增{gains}，失去{losses}。这是当前固定bank中的成功集扩展，不能以task4的两例收益替代task3目标或推广为跨seed稳定性。冻结actor输出不变，不将诊断贪心冒充官方actor成绩。全部失败episode仍完成1000步，T1精确缓存只减少原函数求值，没有早停。

## 约束、尺度与目标相关排序

| 同原2048验证转移 | mean | path-bound |
|---|---:|---:|
{risk_table}

| 同2048额外真实路径 | mean终点（只读前向） | path-bound终点 |
|---|---:|---:|
{path_table}

多步上界风险从约1.287降至0.00836，最大风险从1.313降至0.01806，确实约束了原模型在这些真实路径上的过估。平均随机目标距离也从69.89降到57.14，不能单凭风险大幅下降或绝对峰差变小宣称距离恢复。路径成本均值及零长度比例在两个终点完全相同，控制组没有重新训练来适配新路径。

![风险、乘子、路径长度、全部任务执行及同路线归一化值](path_bound.png)

| critic | 原锚点严格下降通道 | 最低峰差/IQR | 全图Spearman | 原走廊Spearman |
|---|---|---:|---:|---:|
{field_table}

归一化最低峰差仅从0.1357降至0.1263，仍为正；原始100k对应0.1195。绝对峰差的缩小伴随距离尺度变化，不能当全程下降路线。参考图0.2网格/0.7半径只用于评价，canonical目标与五个bank实际目标分开，吸附结论不等于连续环境最优性证明。

![全部五个既有task3真实轨迹](task3_trajectories.png)

## 核验、决策及复现

四项测试覆盖原batch和后续RNG完全不变、双terminal/valid边界与零长度、一步退化损失/梯度、非零动量冻结/实际count。首次仅风险均值的单独编译归约相差一个FP32 ULP，失败日志与原测试源码保留；最终允许该诊断比较2e−6误差，一步损失精确、梯度rtol1e−6/atol2e−6，未改损失公式。最终四项通过。

实际两步smoke与正式终点完整agent/优化器/RNG往返相同，count101000、step101001、冻结参数/矩、同25例actor输出和所有数值有限均通过。源/目标及结束integration state、模型和源码摘要已核验。原批次流SHA {t['paired_stream_sha256']} 与E6一致，额外路径流SHA {t['path_stream_sha256']}；终点模型SHA {t['checkpoint_sha256']}。

该固定短预算方法对task4有真实收益，但未修复task3；不搜索跨度、归一化、混合权重、lr或eps挽救本次方案，也不据1000步否定所有长程约束或宣称充分收敛。后续可检验当前场中的方向不对称等结构偏差，不能由本次实验断言其为唯一根因；不把约束风险再压低当作目标完成。不恢复seed1或1M。

协议见[PATH_BOUND_PROTOCOL.md](../../../experiments/ogbench_critic/PATH_BOUND_PROTOCOL.md)。本地原始目录`results/ogbench_critic_20261029/path_bound_1000`，日志同级`path_bound_1000.log`。入口`run_path_bound.py <结果根目录> --output <全新目录>`，报告`report_path_bound.py <结果根目录>`只读重建，不训练或重跑评价。只提交源码/测试/协议/报告及引用图，原始模型/CSV/JSON/NPZ/日志留本地。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
