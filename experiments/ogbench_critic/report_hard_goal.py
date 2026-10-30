"""File-only finite worst-goal-pool comparison and fixed endpoint verification."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree
from report_iqe_distill import read,sha,counts


def run(root):
    root=Path(root).resolve();out=root/'hard_goal_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];n=a['neural']
    before=read(root/'goal_support_1000/analysis.json');old=before['neural']['cases'];cases=n['cases']
    base=read(root/'baseline_greedy/summary.json')['cases'];teacher=read(root/'roadmap_metric/analysis.json')['cases']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==360 and t['steps']==1000 and t['adam_count']==101000
    assert t['passed'] and t['full_state_roundtrip'] and t['frozen_parameters_and_moments'] and t['frozen_bank_actor_outputs']
    assert t['paired_stream_sha256']==p['expected_stream_sha256']==before['training']['paired_stream_sha256']
    assert t['old_queries_stream_sha256']==before['training']['old_queries_stream_sha256']
    assert t['uniform_queries_stream_sha256']==before['training']['queries_stream_sha256']
    assert p['candidates']==8 and sha(out/'validation_pool.npz')==p['validation_pool_sha256']
    assert sha(out/'selected_support_counts.npy')==t['selected_counts_sha256']
    selected=np.load(out/'selected_support_counts.npy');assert selected.shape==(6302,) and selected.sum()==128000
    assert (selected>0).sum()==t['selected_unique_prototypes'] and selected.max()==t['selected_max_count']
    e12=read(root/'query_augmentation_1000/analysis.json');assert sha(root/'query_augmentation_1000/analysis.json')==p['e12_analysis_sha256']
    assert sha(out/'support_counts.npy')==t['support_counts_sha256']
    support=np.load(out/'support_counts.npy');assert support.shape==(6302,) and support.sum()==128000 and support.min()==20 and support.max()==21
    assert sha(out/'model/params_101000.pkl')==t['checkpoint_sha256']==n['checkpoint_sha256']
    assert sha(root/'goal_support_1000/analysis.json')==p['previous_analysis_sha256']
    assert sha(root/'goal_support_1000/model/params_101000.pkl')==p['previous_checkpoint_sha256']
    assert sha(root/'roadmap_metric/analysis.json')==p['teacher_analysis_sha256']
    assert sha(root/'roadmap_metric/graph.npz')==p['teacher_graph_sha256']
    assert sha(root/'roadmap_metric/distances.npy')==p['teacher_distance_sha256']
    for name,digest in p['source_sha256'].items():
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['core.py','fast_pointmaze.py'] else Path(__file__).parent
        assert sha(out/'source_snapshot'/name)==digest==sha(directory/name)
    assert sha(out/'validation_targets.npy')==p['validation_targets_sha256']
    assert sha(out/'augmented_validation_targets.npy')==p['augmented_validation_targets_sha256']
    assert sha(root/'evaluation_bank.npz')==n['bank_sha256']==before['neural']['bank_sha256']
    assert n['complete'] and n['only_neural_scores'] and not n['roadmap_module_imported'] and not n['data_graph_module_imported']
    assert n['candidate_evaluator_source_verified'] and read(out/'smoke/verification.json')['passed']
    assert not (out/'failure.json').exists() and list(old)==list(cases)==list(base) and len(cases)==25
    for name,info in cases.items():
        with np.load(out/'neural_evaluation'/f'{name}.npz') as f:
            assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2)
            assert f['state_values'].shape==(info['steps']+1,) and f['final_integration'].ndim==1
            assert all(np.isfinite(f[k]).all() for k in f.files)
            np.testing.assert_array_equal(f['observations'][-1],info['final_xy'])
            with np.load(root/'goal_support_1000/neural_evaluation'/f'{name}.npz') as previous:
                np.testing.assert_array_equal(f['goal'],previous['goal']);np.testing.assert_array_equal(f['observations'][0],previous['observations'][0])
    rows=list(csv.DictReader((out/'metrics.csv').open()));assert int(rows[-1]['step'])==1000;last=rows[-1]
    previous_stats=read(out/'previous_statistics.json');initial=read(out/'initial_statistics.json');aug_initial=read(out/'initial_augmented_statistics.json')
    gains=[x for x in old if cases[x]['success'] and not old[x]['success']];losses=[x for x in old if old[x]['success'] and not cases[x]['success']]
    base_losses=[x for x in base if base[x]['success'] and not cases[x]['success']];gate=counts(cases)[2]==5 and not base_losses
    review=read(out/'gate_review/analysis.json');assert review['complete'] and review['post_hoc'] and review['training_updates']==review['new_episodes']==0 and review['elapsed_seconds']<60
    assert review['hard_analysis_sha256']==sha(out/'analysis.json') and review['mean_analysis_sha256']==p['previous_analysis_sha256']
    assert review['source_sha256']==sha(Path(__file__).with_name('probe_hard_goal.py'))
    assert len(review['probes'])==10 and len(review['fields'])==5
    for name in review['probes']:
        with np.load(out/'gate_review'/f'{name}.npz') as f:
            assert f['observations'].shape==(121,2) and f['mean'].shape==f['hard'].shape==f['teacher'].shape==(121,)
            assert all(np.isfinite(f[k]).all() for k in f.files)
    for name in review['fields']:
        with np.load(out/'gate_review'/f'{name}_field.npz') as f,np.load(out/'neural_evaluation'/f'{name}.npz') as episode:
            assert f['nodes'].shape==(7091,2) and f['values'].shape==(7091,)
            np.testing.assert_array_equal(f['goal'],episode['goal'])
            assert all(np.isfinite(f[k]).all() for k in ['nodes','values','walls','goal'])
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,(label,c) in enumerate([('Original QRL',base),('E12',e12['neural']['cases']),('E15 uniform',old),('E16 max-pool',cases)]):
        axes[0,0].bar(np.arange(5)+(i-1.5)*.2,counts(c),width=.2,label=label)
    axes[0,0].set(title='Same fixed 25 cases',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.7),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    for key,label,stat in [('heldout','Original queries',initial),('augmented_heldout','Uniform-goal expanded queries',aug_initial)]:
        line,=axes[0,1].plot([0]+[float(r['step']) for r in rows],[stat['forward_rmse']]+[float(r[f'{key}/forward_rmse']) for r in rows],label=label)
        axes[0,1].axhline(previous_stats['original' if key=='heldout' else 'augmented']['forward_rmse'],color=line.get_color(),ls=':',label=f'E15 endpoint: {label}')
    axes[0,1].set(title='Fixed heldout forward fit',xlabel='Extra value updates',ylabel='RMSE',yscale='log');axes[0,1].legend(fontsize=7)
    pool_controls=read(out/'pool_controls.json')
    axes[0,2].plot([0]+[float(r['step']) for r in rows],[pool_controls['original']['pool_max_rmse']]+[float(r['pool_heldout/pool_max_rmse']) for r in rows],label='E16 worst-of-8 RMSE')
    for color,(label,key) in zip(['C1','C2'],[('E12','e12'),('E15','e15')]):axes[0,2].axhline(pool_controls[key]['pool_max_rmse'],color=color,ls=':',label=f'{label} fixed endpoint')
    axes[0,2].set(title='SAME fixed 8-goal validation pool',xlabel='Extra updates',ylabel='sqrt(mean max paired MSE)',yscale='log');axes[0,2].legend(fontsize=7)
    axes[1,2].plot([float(r['step']) for r in rows],[float(r['selection/selected_separation_below_one_fraction']) for r in rows],label='Selected source-goal L_inf < 1')
    axes[1,2].plot([float(r['step']) for r in rows],[float(r['selection/selected_column_zero_fraction']) for r in rows],label='Chosen candidate zero')
    axes[1,2].set(title='Logged minibatch selection diagnostics',xlabel='Extra updates',ylabel='Fraction',ylim=(-.03,1.03));axes[1,2].legend(fontsize=7)
    with np.load(root/'baseline_field/field.npz') as f:nodes=f['nodes'].copy();idx=cKDTree(nodes).query(f['path'])[1]
    for label,path in [('E15',root/'goal_support_1000/neural_evaluation/field.npz'),('E16',out/'neural_evaluation/field.npz'),('Teacher',root/'roadmap_metric/canonical_field.npz')]:
        with np.load(path) as f:np.testing.assert_array_equal(f['nodes'],nodes);v=f['values'][idx];walls=f['walls'].copy()
        axes[1,0].plot(v-v[0],label=label)
    axes[1,0].set(title='SAME canonical reference route',xlabel='Reference step',ylabel='d - d_start');axes[1,0].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'neural_evaluation'/f'task3_seed{seed}.npz') as f:axes[1,1].plot(f['state_values'],label=str(seed))
    axes[1,1].set(title='E16 all five task3 episodes',xlabel='Actual step',ylabel='Neural forward distance');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'hard_goal.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for ax,seed in zip(axes,range(3100,3105)):
        name=f'task3_seed{seed}'
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path in [('E15',root/'goal_support_1000/neural_evaluation'/f'{name}.npz'),('E16',out/'neural_evaluation'/f'{name}.npz')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=f'{seed}: {"success" if cases[name]["success"] else "timeout"}',aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=8);fig.savefig(out/'task3_hard_goal.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(old)[k-1]}/5 | {counts(cases)[k-1]}/5 | {counts(teacher)[k-1]}/5 |' for k in range(1,6))
    task3='\n'.join(f"| {name} | {v['success']} | {v['steps']} | {v['strict_decrease_fraction']:.4f} | {v['increase_fraction']:.4f} |" for name,v in cases.items() if name.startswith('task3_'))
    probe_table='\n'.join(f"| {name} | {v['mean']['rmse']:.3f} / {v['hard']['rmse']:.3f} | {v['mean']['teacher_regret']:.3f} / {v['hard']['teacher_regret']:.3f} | {v['mean']['selected_displacement']:.3f} / {v['hard']['selected_displacement']:.3f} |" for name,v in review['probes'].items())
    field_table='\n'.join(f"| {name} | {v['strict_descent_from_anchor']} | {v['minimum_peak_gap_over_iqr']:.6f} | {v['corridor_spearman']:.6f} | {v['snap_distance']:.5f} |" for name,v in review['fields'].items())
    report=f'''# E16：固定八候选池的最坏全局拟合监督

固定1000次更新后{sum(counts(cases))}/25成功，task3 {counts(cases)[2]}/5；E15为21/25、task3 3/5，E12为23/25、task3 3/5。相对E15新增{gains}，失去{losses}；失去原100k成功例{base_losses}。预定critic门槛{gate}。总{a['elapsed_seconds']:.2f}秒，训练核验{t['train_verify_seconds']:.2f}、独立评价{n['elapsed_seconds']:.2f}，低于360秒；候选前向额外计算，不声称等FLOP/墙钟。

## 固定学习问题与控制

同原100k IQE/value Adam/count/lr、E10教师、原始官方sample256/seed2001/1000更新。actor/dynamics/lam参数和Adam矩冻结。半批原始(s,g)保持，半批source及Sobol扰动同E12/E15；其goal从固定8候选中选当前双向MSE最大的一个，选择不反传。候选k编号((j+k*floor(K/8))*stride)%K，K6302、stride3895，同一行goal偏移相同，候选0逐位同E15，未扫描池大小/比例/相位。

新目标是半批原始平均MSE+半批有限池最大MSE的次梯度，不宣称原经验目标不变，也非原QRL局部风险top-k。只通过训练数据/固定教师/当前网络误差选目标，没有评价失败点、墙体、路线标签或模拟器生成转移；原轨迹/valid/未来目标语义不变。扩展查询可物理不可达，M5对角不可一致拟合限制保留，因此需检查是否追逐近对伪影，不能把高残差都叫真实最优距离误差。

三项测试通过：候选0/source/输入/人口循环/RNG身份，双向误差与并列取首，选中查询梯度等价显式max-pool目标（FP32 rtol2e−5/atol3e−5）。原E11优化器和损失不改。真实两步smoke完整保存加载后重置原模型/RNG，终点count101000/step101001、有限性/全状态往返/冻结参数与矩/同bank actor输出逐位核验通过。

原流SHA {t['paired_stream_sha256']}同E6；旧E12 query SHA {t['old_queries_stream_sha256']}、候选0完整query SHA {t['uniform_queries_stream_sha256']}同E15对应流，每批source逐位核验。池SHA {t['pool_stream_sha256']}，选择SHA {t['selection_stream_sha256']}，训练query SHA {t['queries_stream_sha256']}，target SHA {t['targets_stream_sha256']}。128000次选择覆盖{t['selected_unique_prototypes']}个原型、最多一个{t['selected_max_count']}次；选择后并非均匀分布，不将候选0每原型20–21次当实际选择人口。

## 全任务真实执行

| 任务 | 原100k | E15 | E16 | 固定教师 |
|---|---:|---:|---:|---:|
{table}

![全任务、普通验证、固定最坏池误差、同参考路线和采样诊断](hard_goal.png)

| task3 case | 成功 | 完整步数 | 严格下降比例 | 上坡比例 |
|---|---|---:|---:|---:|
{task3}

原独立Python评价进程仅载入QRL checkpoint与原forward IQE，无教师图导入/推理或对称包装；121真实后继/T1所选实际后继逐位核验、精确缓存、完整1000步/成功终止保持。参考墙体图只在episode之后用于场。原actor冻结，不能以神经贪心当官方actor成绩，控制E12/E15没有重跑episode。

## 误差与距离场

固定验证池的mean max paired MSE：原100k {pool_controls['original']['pool_max_squared_error']:.5f}，E12 {pool_controls['e12']['pool_max_squared_error']:.5f}，E15 {pool_controls['e15']['pool_max_squared_error']:.5f}，E16 {float(last['pool_heldout/pool_max_squared_error']):.5f}；对应池最大RMSE，E12/E15 {pool_controls['e12']['pool_max_rmse']:.5f}/{pool_controls['e15']['pool_max_rmse']:.5f}，E16 {float(last['pool_heldout/pool_max_rmse']):.5f}。这是固定非评价数据池的双向误差，不是最坏全域保证。

原验证forward/reverse RMSE {float(last['heldout/forward_rmse']):.5f}/{float(last['heldout/reverse_rmse']):.5f}，E15 {previous_stats['original']['forward_rmse']:.5f}/{previous_stats['original']['reverse_rmse']:.5f}；旧E12扩展{float(last['old_augmented_heldout/forward_rmse']):.5f}/{float(last['old_augmented_heldout/reverse_rmse']):.5f}，E15 {previous_stats['old_augmented']['forward_rmse']:.5f}/{previous_stats['old_augmented']['reverse_rmse']:.5f}；E15均衡扩展{float(last['augmented_heldout/forward_rmse']):.5f}/{float(last['augmented_heldout/reverse_rmse']):.5f}，E15 {previous_stats['augmented']['forward_rmse']:.5f}/{previous_stats['augmented']['reverse_rmse']:.5f}。同组比较，不混不同分布数值。

最后记录批次所选source-goal L∞小于1比例{float(last['selection/selected_separation_below_one_fraction']):.4f}、候选0比例{float(last['selection/selected_column_zero_fraction']):.4f}；记录批次曲线在图中，不将11个记录批当全部128000选择分布，也不能据近对比例直接证明/排除所有教师伪影。

canonical通道{n['field']['strict_descent_from_anchor']}，峰差/IQR {n['field']['minimum_peak_gap_over_iqr']:.6f}，全图/corridor相关{n['field']['all_graph_spearman']:.6f}/{n['field']['corridor_spearman']:.6f}；0.2参考图/0.14451吸附误差限制仍在。原pair/尾风险{float(last['original_risk/pair_mean_squared_positive']):.6f}/{float(last['original_risk/pair_tail_squared_positive']):.6f}、goal风险{float(last['original_risk/goal_squared_positive']):.6f}，lambda固定{float(last['original_risk/lam']):.3f}，不作新收敛证明。

![E15/E16全部五个task3轨迹](task3_hard_goal.png)

## 出口及复现

本轮通过预定critic门槛，E15的两个task3和两个task4失败均恢复，无失去；相比原100k新增全部五个task3和四个task4。五个task3分别68/76/64/67/65步、每步神经距离下降，原actor仍未利用新critic，不能据此标记总目标完成。当前成功库已多次用于方法选择，不是独立泛化验证。

## 通过门槛后的只读复核

另用{review['elapsed_seconds']:.2f}秒（独立60秒上限）核查五个既有task3起点及五个E15已存结束状态，每点121真实后继，无新episode/训练；随后计算五个实际目标的参考场。该项为事后机制解释，不冒充冻结主实验或独立评价。两个模型参数摘要不变。

| 固定探针 | 候选RMSE mean / hard | 教师regret mean / hard | 所选真实位移L∞ mean / hard |
|---|---:|---:|---:|
{probe_table}

两个E15失败终点，teacher-regret从约1.635/1.530降到0，所选位移从近零变成0.264/0.238；在相同真实候选上恢复了有用排序。原局部RMSE12.5/13.5降至7.4/8.3仍明显不为零，说明正确行动不需要完美数值校准。其他点误差或regret并非全部单调改善，表格完整保留，不能用一处指标代表全域。

| 实际task3目标 | 起点严格通道 | 峰差/IQR | corridor相关 | 吸附误差 |
|---|---|---:|---:|---:|
{field_table}

五个实际目标均有通道，与canonical和真实执行方向一致；仍仅0.2离散图证据，既有真实成功不能扩张成连续最优性证明。每个所选实际后继在主评价中逐位核验，碰撞释放下位移可大于0.2命令增量，表中保存真实后继位移而非强行截断。

同原表示/原Adam、相同梯度更新数与source/候选0流的对照，支持有限池高残差全局监督能恢复当前失败区域。它增加候选前向和目标多样性，未分别消融这些组成，也不单独证明原QRL唯一根因是平均MSE；原QRL本来就不使用此教师回归目标。新增教师及其几何/局部可逆先验和M5不连续限制仍需明示。本项不扫描池大小/比例/打分缩放/lr/预算，不换seed或运行1M；下一项另冻结官方策略/动力学适配对照，验证actor能否实际利用新critic。

协议[HARD_GOAL_PROTOCOL.md](../../../experiments/ogbench_critic/HARD_GOAL_PROTOCOL.md)，测试test_hard_goal.py，入口run_hard_goal.py <根目录> --output <全新目录>，事后probe_hard_goal.py；报告report_hard_goal.py <根目录>只读重建。raw results/ogbench_critic_20261029/hard_goal_1000，日志同级.log，checkpoint model/params_101000.pkl SHA {t['checkpoint_sha256']}。原模型/教师/E12/E15/源码/全部同bank轨迹/结束状态已核验；只提交代码/测试/协议/本报告及引用图，原始模型/数据/日志不入库。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
