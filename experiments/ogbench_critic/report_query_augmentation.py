"""File-only comparison of fixed E11/E12 endpoints, including source identity."""
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
    root=Path(root).resolve();out=root/'query_augmentation_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];n=a['neural']
    before=read(root/'iqe_distillation_1000/analysis.json');old=before['neural']['cases'];cases=n['cases']
    base=read(root/'baseline_greedy/summary.json')['cases'];teacher=read(root/'roadmap_metric/analysis.json')['cases']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==300 and t['steps']==1000 and t['adam_count']==101000
    assert t['passed'] and t['full_state_roundtrip'] and t['frozen_parameters_and_moments'] and t['frozen_bank_actor_outputs']
    assert t['paired_stream_sha256']==p['expected_stream_sha256']==before['training']['paired_stream_sha256']
    assert sha(out/'model/params_101000.pkl')==t['checkpoint_sha256']==n['checkpoint_sha256']
    assert sha(root/'iqe_distillation_1000/analysis.json')==p['previous_analysis_sha256']
    assert sha(root/'iqe_distillation_1000/model/params_101000.pkl')==p['previous_checkpoint_sha256']
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
            with np.load(root/'iqe_distillation_1000/neural_evaluation'/f'{name}.npz') as previous:
                np.testing.assert_array_equal(f['goal'],previous['goal']);np.testing.assert_array_equal(f['observations'][0],previous['observations'][0])
    rows=list(csv.DictReader((out/'metrics.csv').open()));assert int(rows[-1]['step'])==1000;last=rows[-1]
    previous_stats=read(out/'previous_statistics.json');initial=read(out/'initial_statistics.json');aug_initial=read(out/'initial_augmented_statistics.json')
    gains=[x for x in old if cases[x]['success'] and not old[x]['success']];losses=[x for x in old if old[x]['success'] and not cases[x]['success']]
    base_losses=[x for x in base if base[x]['success'] and not cases[x]['success']];gate=counts(cases)[2]==5 and not base_losses
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,(label,c) in enumerate([('Original QRL',base),('E11 MSE',old),('E12 query expansion',cases)]):
        axes[0,0].bar(np.arange(5)+(i-1)*.25,counts(c),width=.25,label=label)
    axes[0,0].set(title='Same fixed 25 cases',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.7),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    for key,label,stat in [('heldout','Original queries',initial),('augmented_heldout','Expanded queries',aug_initial)]:
        line,=axes[0,1].plot([0]+[float(r['step']) for r in rows],[stat['forward_rmse']]+[float(r[f'{key}/forward_rmse']) for r in rows],label=label)
        axes[0,1].axhline(previous_stats['original' if key=='heldout' else 'augmented']['forward_rmse'],color=line.get_color(),ls=':',label=f'E11 endpoint: {label}')
    axes[0,1].set(title='Fixed heldout forward fit',xlabel='Extra value updates',ylabel='RMSE',yscale='log');axes[0,1].legend(fontsize=7)
    with np.load(root/'baseline_field/field.npz') as f:nodes=f['nodes'].copy();idx=cKDTree(nodes).query(f['path'])[1]
    for label,path in [('E11',root/'iqe_distillation_1000/neural_evaluation/field.npz'),('E12',out/'neural_evaluation/field.npz'),('Teacher',root/'roadmap_metric/canonical_field.npz')]:
        with np.load(path) as f:np.testing.assert_array_equal(f['nodes'],nodes);v=f['values'][idx];walls=f['walls'].copy()
        axes[1,0].plot(v-v[0],label=label)
    axes[1,0].set(title='SAME canonical reference route',xlabel='Reference step',ylabel='d - d_start');axes[1,0].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'neural_evaluation'/f'task3_seed{seed}.npz') as f:axes[1,1].plot(f['state_values'],label=str(seed))
    axes[1,1].set(title='E12 all five task3 episodes',xlabel='Actual step',ylabel='Neural forward distance');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'query_augmentation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for ax,seed in zip(axes,range(3100,3105)):
        name=f'task3_seed{seed}'
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path in [('E11',root/'iqe_distillation_1000/neural_evaluation'/f'{name}.npz'),('E12',out/'neural_evaluation'/f'{name}.npz')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=f'{seed}: {"success" if cases[name]["success"] else "timeout"}',aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=8);fig.savefig(out/'task3_query_augmentation.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(old)[k-1]}/5 | {counts(cases)[k-1]}/5 | {counts(teacher)[k-1]}/5 |' for k in range(1,6))
    task3='\n'.join(f"| {name} | {v['success']} | {v['steps']} | {v['strict_decrease_fraction']:.4f} | {v['increase_fraction']:.4f} |" for name,v in cases.items() if name.startswith('task3_'))
    report=f'''# E12：任务无关查询扩展与原IQE距离学习

固定1000次更新完成，全部{sum(counts(cases))}/25成功，task3为{counts(cases)[2]}/5；E11为22/25、task3 2/5。相对E11新增{gains}，失去{losses}；失去原100k成功例{base_losses}。预定critic门槛为{gate}，不能以平均拟合误差替代此门槛。总{a['elapsed_seconds']:.2f}秒，其中训练/核验{t['train_verify_seconds']:.2f}秒，独立评价{n['elapsed_seconds']:.2f}秒，低于300秒。

## 固定方法与控制

E10教师、原IQE/Adam/count/lr、双向等权MSE与E11完全相同；只改变训练查询分布。每批偶数行保留原(s,g)，奇数行分别加四维非scramble Sobol偏移[-1,1]，不截断/过滤。幅度1来自固定官方maze.py的noise=1和maze_unit=4公式，不来自评价case调参。训练使用前128000个Sobol偏移，验证接下来1024个；没有新随机seed、不改变NumPy官方采样RNG。原数据转移和轨迹/未来目标语义不变，扩展只作教师查询，不把虚构点当真实转移。这些点可能在物理不可达处，标签仅抽象数据/几何先验，非真实MDP最优成本。

正式从同原100k出发1000步batch256/seed2001，只更新value；actor/dynamics/lam及矩冻结。原采样流SHA {t['paired_stream_sha256']}与E6/E11相同，扩展查询流SHA {t['queries_stream_sha256']}，标签流SHA {t['targets_stream_sha256']}。E11与原损失控制直接复用、不重训或重评。两项采样器测试通过，原损失/梯度/优化器测试复用E11通过证据；实际两步smoke保存加载后正式重置原模型，终点count101000/全部状态往返、数值有限、冻结参数/矩/actor输出通过。

## 全任务执行与拟合

| 任务 | 原100k | E11 | E12 | 固定教师 |
|---|---:|---:|---:|---:|
{table}

![固定终点全任务、验证误差、同参考路线与五个task3值](query_augmentation.png)

| task3 case | 成功 | 完整步数 | 严格下降比例 | 上坡比例 |
|---|---|---:|---:|---:|
{task3}

独立子进程仅载入原QRL checkpoint、原forward神经评分，不导入roadmap_metric/data_graph，不使用教师路径或对称包装。25例完整官方1000步/成功终止，121真实后继、实际选中后继逐位核验与精确缓存保持；参考墙体图仅末尾用于场评价。不是官方actor评价，原actor仍冻结；25-case是已有方法选择库，不是独立跨seed验证。

原验证forward/reverse RMSE终点{float(last['heldout/forward_rmse']):.5f}/{float(last['heldout/reverse_rmse']):.5f}；E11为{previous_stats['original']['forward_rmse']:.5f}/{previous_stats['original']['reverse_rmse']:.5f}。扩展验证终点{float(last['augmented_heldout/forward_rmse']):.5f}/{float(last['augmented_heldout/reverse_rmse']):.5f}，E11为{previous_stats['augmented']['forward_rmse']:.5f}/{previous_stats['augmented']['reverse_rmse']:.5f}。两种heldout独立保留，未据误差或成功率追加训练/选择中间终点。

canonical严格下降通道{n['field']['strict_descent_from_anchor']}，峰差/IQR {n['field']['minimum_peak_gap_over_iqr']:.6f}，全图/corridor相关{n['field']['all_graph_spearman']:.6f}/{n['field']['corridor_spearman']:.6f}。仍为0.2参考图/吸附误差0.14451的离散检查，零峰差不保证严格通道。原pair平均/尾风险{float(last['original_risk/pair_mean_squared_positive']):.6f}/{float(last['original_risk/pair_tail_squared_positive']):.6f}，goal风险{float(last['original_risk/goal_squared_positive']):.6f}，lambda固定{float(last['original_risk/lam']):.3f}；不把冻结lambda作新优化收敛证明。

![E11/E12全部五个task3完整轨迹](task3_query_augmentation.png)

## 出口与复现

本轮门槛未通过：新增3104，两个旧成功保留，但3100/3101仍停在墙边。扩展验证误差降低（原验证基本不变）伴随一例真实收益，支持查询分布参与恢复，但不足以解释或消除剩余距离场错误。canonical通道由False变True仍有实际失败，单锚点不足以验收。停止本扩展幅度/比例/排列/学习率/预算搜索，不启动actor训练，不据小随机误差宣称解决；下一项应针对目标距离的局部形状信号或表示机制，而非继续采样幅度搜索。不换seed、不运行1M、不中途挑模型。查询分布扩展只检验这一固定方案，不能将失败泛化为所有采样或距离表示均无效。

协议[QUERY_AUGMENT_PROTOCOL.md](../../../experiments/ogbench_critic/QUERY_AUGMENT_PROTOCOL.md)，测试`test_query_augmentation.py`；正式入口`run_query_augmentation.py <根目录> --output <全新目录>`，报告`report_query_augmentation.py <根目录>`只读重建。raw `results/ogbench_critic_20261029/query_augmentation_1000`，日志同级`.log`，checkpoint `model/params_101000.pkl` SHA {t['checkpoint_sha256']}。已核验固定教师、E11/控制、原模型/源码快照、同bank全部轨迹/结束状态；只提交代码、协议、测试、报告及引用图，原始结果不提交。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
