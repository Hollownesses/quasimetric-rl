"""File-only fixed E12/E13 signed-shape endpoint comparison."""
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
    root=Path(root).resolve();out=root/'local_shape_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];n=a['neural']
    before=read(root/'query_augmentation_1000/analysis.json');old=before['neural']['cases'];cases=n['cases']
    base=read(root/'baseline_greedy/summary.json')['cases'];teacher=read(root/'roadmap_metric/analysis.json')['cases']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==300 and t['steps']==1000 and t['adam_count']==101000
    assert t['passed'] and t['full_state_roundtrip'] and t['frozen_parameters_and_moments'] and t['frozen_bank_actor_outputs']
    assert t['paired_stream_sha256']==p['expected_stream_sha256']==before['training']['paired_stream_sha256']
    assert t['queries_stream_sha256']==before['training']['queries_stream_sha256']
    assert t['targets_stream_sha256']==before['training']['targets_stream_sha256']
    assert sha(out/'model/params_101000.pkl')==t['checkpoint_sha256']==n['checkpoint_sha256']
    assert sha(root/'query_augmentation_1000/analysis.json')==p['previous_analysis_sha256']
    assert sha(root/'query_augmentation_1000/model/params_101000.pkl')==p['previous_checkpoint_sha256']
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
            with np.load(root/'query_augmentation_1000/neural_evaluation'/f'{name}.npz') as previous:
                np.testing.assert_array_equal(f['goal'],previous['goal']);np.testing.assert_array_equal(f['observations'][0],previous['observations'][0])
    rows=list(csv.DictReader((out/'metrics.csv').open()));assert int(rows[-1]['step'])==1000;last=rows[-1]
    previous_stats=read(out/'previous_statistics.json');initial=read(out/'initial_statistics.json');aug_initial=read(out/'initial_augmented_statistics.json')
    gains=[x for x in old if cases[x]['success'] and not old[x]['success']];losses=[x for x in old if old[x]['success'] and not cases[x]['success']]
    base_losses=[x for x in base if base[x]['success'] and not cases[x]['success']];gate=counts(cases)[2]==5 and not base_losses
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,(label,c) in enumerate([('Original QRL',base),('E12 query expansion',old),('E13 signed shape',cases)]):
        axes[0,0].bar(np.arange(5)+(i-1)*.25,counts(c),width=.25,label=label)
    axes[0,0].set(title='Same fixed 25 cases',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.7),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    for key,label,stat in [('heldout','Original queries',initial),('augmented_heldout','Expanded queries',aug_initial)]:
        line,=axes[0,1].plot([0]+[float(r['step']) for r in rows],[stat['forward_rmse']]+[float(r[f'{key}/forward_rmse']) for r in rows],label=label)
        axes[0,1].axhline(previous_stats['original' if key=='heldout' else 'augmented']['forward_rmse'],color=line.get_color(),ls=':',label=f'E12 endpoint: {label}')
    axes[0,1].set(title='Fixed heldout forward fit',xlabel='Extra value updates',ylabel='RMSE',yscale='log');axes[0,1].legend(fontsize=7)
    for key,label,stat in [('heldout','Original queries',initial),('augmented_heldout','Expanded queries',aug_initial)]:
        line,=axes[0,2].plot([0]+[float(r['step']) for r in rows],[stat['forward_shape_rmse']]+[float(r[f'{key}/forward_shape_rmse']) for r in rows],label=label)
        axes[0,2].axhline(previous_stats['original' if key=='heldout' else 'augmented']['forward_shape_rmse'],color=line.get_color(),ls=':',label=f'E12: {label}')
    axes[0,2].set(title='Signed finite-difference fitting',xlabel='Extra value updates',ylabel='RMSE / 0.2',yscale='log');axes[0,2].legend(fontsize=7)
    for key,label in [('pair_mean_squared_positive','Original pair risk'),('pair_tail_squared_positive','Original tail risk'),('goal_squared_positive','Original goal risk')]:
        axes[1,2].plot([float(r['step']) for r in rows],[float(r[f'original_risk/{key}']) for r in rows],label=label)
    axes[1,2].set(title='Original heldout risks (lambda frozen)',xlabel='Extra value updates',ylabel='Squared positive risk',yscale='log');axes[1,2].legend(fontsize=7)
    with np.load(root/'baseline_field/field.npz') as f:nodes=f['nodes'].copy();idx=cKDTree(nodes).query(f['path'])[1]
    for label,path in [('E12',root/'query_augmentation_1000/neural_evaluation/field.npz'),('E13',out/'neural_evaluation/field.npz'),('Teacher',root/'roadmap_metric/canonical_field.npz')]:
        with np.load(path) as f:np.testing.assert_array_equal(f['nodes'],nodes);v=f['values'][idx];walls=f['walls'].copy()
        axes[1,0].plot(v-v[0],label=label)
    axes[1,0].set(title='SAME canonical reference route',xlabel='Reference step',ylabel='d - d_start');axes[1,0].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'neural_evaluation'/f'task3_seed{seed}.npz') as f:axes[1,1].plot(f['state_values'],label=str(seed))
    axes[1,1].set(title='E13 all five task3 episodes',xlabel='Actual step',ylabel='Neural forward distance');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'local_shape.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for ax,seed in zip(axes,range(3100,3105)):
        name=f'task3_seed{seed}'
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path in [('E12',root/'query_augmentation_1000/neural_evaluation'/f'{name}.npz'),('E13',out/'neural_evaluation'/f'{name}.npz')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=f'{seed}: {"success" if cases[name]["success"] else "timeout"}',aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=8);fig.savefig(out/'task3_local_shape.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(old)[k-1]}/5 | {counts(cases)[k-1]}/5 | {counts(teacher)[k-1]}/5 |' for k in range(1,6))
    task3='\n'.join(f"| {name} | {v['success']} | {v['steps']} | {v['strict_decrease_fraction']:.4f} | {v['increase_fraction']:.4f} |" for name,v in cases.items() if name.startswith('task3_'))
    report=f'''# E13：有符号局部形状监督的固定终点对照

固定1000更新后{sum(counts(cases))}/25成功、task3 {counts(cases)[2]}/5，E12为23/25、task3 3/5。相对E12新增{gains}，失去{losses}；失去原100k成功例{base_losses}。预定critic门槛为{gate}。总{a['elapsed_seconds']:.2f}秒，训练/核验{t['train_verify_seconds']:.2f}秒、独立评价{n['elapsed_seconds']:.2f}秒，低于300秒；仅固定终点，没有追加或挑最好。

## 单变量学习问题

同原100k IQE/Adam/count/lr、E10近似教师、E12半批原查询/半批±1确定Sobol扩展；只增加有符号局部距离差监督。F=d(s,g)、F'=d(s_next,g)，R=d(g,s)、R'=d(g,s_next)，Delta=t(s,g)−t(s_next,g)。新loss为原双向绝对MSE加mean[0.5*((F−F'−Delta)/0.2)^2+0.5*((R−R'−Delta)/0.2)^2]，系数1固定、0.2取官方每轴最大位移。Delta允许正负，不截断，不要求行为最优/距离每步减1。教师含数据图/局部可逆/几何先验，不是官方最优成本。

扩展行的next与source加相同偏移，只作抽象查询，不声称平移后还是合法物理转移；不使用评价墙体/目标/路线。原始官方valid轨迹/目标采样保留。训练1000步batch256、采样seed2001不变，从原100k起步而非E12续训；E12直接复用、不重训或重评。actor/dynamics/lam参数和Adam矩冻结，只更新value。

原始流SHA {t['paired_stream_sha256']}，当前查询SHA {t['queries_stream_sha256']}及当前教师标签SHA {t['targets_stream_sha256']}与E12相同；新next查询SHA {t['next_queries_stream_sha256']}、新标签SHA {t['next_targets_stream_sha256']}。因此绝对监督查询未变，新增变量是局部形状信号。

四项测试最终通过：有符号公式/梯度且无其他模块梯度、next=s时形状零/退化原MSE、非零Adam矩冻结/count、当前查询逐位同E12与next同平移。首次共享/分开编码器调用的FP32梯度累加差约9.54e−6使2e−6阈值失败，保留两个失败日志及修改前源码；loss逐位相同，梯度核验rtol2e−5/atol3e−5，训练公式未改。真实两步smoke完整保存加载后重置原模型/RNG正式运行；终点count101000、完整agent/Adam/RNG往返、有限性、冻结参数/矩/actor输出核验通过。

## 完整真实执行

| 任务 | 原100k | E12 | E13 | 固定教师 |
|---|---:|---:|---:|---:|
{table}

![固定终点执行、绝对验证误差、同参考路线与五个task3值](local_shape.png)

| task3 case | 成功 | 完整步数 | 严格下降比例 | 上坡比例 |
|---|---|---:|---:|---:|
{task3}

独立Python子进程仅载入原QRL checkpoint与原forward IQE评分，未导入roadmap_metric/data_graph、无教师推理或对称包装。原121动作真实后继、完整episode1000步/成功终止、实际所选后继逐位核验与T1精确缓存不变。参考墙体图只在episode结束后评价。该控制是critic诊断，不是官方actor评价；原actor输出保持不变。

## 拟合与形状

原验证forward/reverse绝对RMSE {float(last['heldout/forward_rmse']):.5f}/{float(last['heldout/reverse_rmse']):.5f}，E12为{previous_stats['original']['forward_rmse']:.5f}/{previous_stats['original']['reverse_rmse']:.5f}；扩展验证{float(last['augmented_heldout/forward_rmse']):.5f}/{float(last['augmented_heldout/reverse_rmse']):.5f}，E12为{previous_stats['augmented']['forward_rmse']:.5f}/{previous_stats['augmented']['reverse_rmse']:.5f}。

原验证形状forward/reverse RMSE {float(last['heldout/forward_shape_rmse']):.5f}/{float(last['heldout/reverse_shape_rmse']):.5f}，E12 {previous_stats['original']['forward_shape_rmse']:.5f}/{previous_stats['original']['reverse_shape_rmse']:.5f}；扩展验证{float(last['augmented_heldout/forward_shape_rmse']):.5f}/{float(last['augmented_heldout/reverse_shape_rmse']):.5f}，E12 {previous_stats['augmented']['forward_shape_rmse']:.5f}/{previous_stats['augmented']['reverse_shape_rmse']:.5f}。这些统计记录新目标是否生效，不能单独代替实际成功集和距离通道。

canonical严格下降通道{n['field']['strict_descent_from_anchor']}，最低峰差/IQR {n['field']['minimum_peak_gap_over_iqr']:.6f}，全图/corridor相关{n['field']['all_graph_spearman']:.6f}/{n['field']['corridor_spearman']:.6f}。0.2离散图与0.14451锚点吸附误差限制仍在，零峰差不代表严格通道。原pair平均/尾风险{float(last['original_risk/pair_mean_squared_positive']):.6f}/{float(last['original_risk/pair_tail_squared_positive']):.6f}、goal风险{float(last['original_risk/goal_squared_positive']):.6f}，lambda固定{float(last['original_risk/lam']):.3f}，不作新优化收敛结论。

![全部五个task3的E12/E13完整轨迹](task3_local_shape.png)

## 结论范围与复现

本轮成功集合与E12完全相同，3100/3101仍失败；3101轨迹虽改向另一个墙边，未形成真实成功。原/扩展验证forward形状RMSE分别下降约9.5%/5.2%，说明新信号生效，但这种有限改善未扩大执行成功集。canonical通道保持True亦不足验收。停止该形状系数/缩放/lr/预算搜索，不训练actor、不把风险压低当解决。有限预算的负结果不能证明原IQE表达不了目标距离；下一步可先核验另一种全局距离坐标表示是否保持教师的执行性质，再决定是否训练，避免无依据扩大当前网络训练。

结论仅限现有模型/已知25-case方法选择库；不换seed、不运行1M、不恢复已停止程序，不宣称跨seed稳定、官方actor改善或论文完整预算复现。总目标仍未完成。

协议[LOCAL_SHAPE_PROTOCOL.md](../../../experiments/ogbench_critic/LOCAL_SHAPE_PROTOCOL.md)，测试test_local_shape.py；入口run_local_shape.py <根目录> --output <全新目录>，报告report_local_shape.py <根目录>只读重建。raw results/ogbench_critic_20261029/local_shape_1000，日志同级.log；checkpoint model/params_101000.pkl SHA {t['checkpoint_sha256']}。教师/控制/E12/原模型/源码快照/全部同bank轨迹与结束状态已核验。只提交源码、测试、协议、报告和两张引用图，原始产物不提交。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
