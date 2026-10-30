"""File-only goal-support experiment with reused E12 control."""
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
    root=Path(root).resolve();out=root/'goal_support_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];n=a['neural']
    before=read(root/'query_augmentation_1000/analysis.json');old=before['neural']['cases'];cases=n['cases']
    base=read(root/'baseline_greedy/summary.json')['cases'];teacher=read(root/'roadmap_metric/analysis.json')['cases']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==300 and t['steps']==1000 and t['adam_count']==101000
    assert t['passed'] and t['full_state_roundtrip'] and t['frozen_parameters_and_moments'] and t['frozen_bank_actor_outputs']
    assert t['paired_stream_sha256']==p['expected_stream_sha256']==before['training']['paired_stream_sha256']
    assert t['old_queries_stream_sha256']==before['training']['queries_stream_sha256']
    assert sha(out/'support_counts.npy')==t['support_counts_sha256']
    support=np.load(out/'support_counts.npy');assert support.shape==(6302,) and support.sum()==128000 and support.min()==20 and support.max()==21
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
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,(label,c) in enumerate([('Original QRL',base),('E12 query expansion',old),('E15 goal support',cases)]):
        axes[0,0].bar(np.arange(5)+(i-1)*.25,counts(c),width=.25,label=label)
    axes[0,0].set(title='Same fixed 25 cases',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.7),ylabel='Successes / 5');axes[0,0].legend(fontsize=8)
    for key,label,stat in [('heldout','Original queries',initial),('augmented_heldout','Uniform-goal expanded queries',aug_initial)]:
        line,=axes[0,1].plot([0]+[float(r['step']) for r in rows],[stat['forward_rmse']]+[float(r[f'{key}/forward_rmse']) for r in rows],label=label)
        axes[0,1].axhline(previous_stats['original' if key=='heldout' else 'augmented']['forward_rmse'],color=line.get_color(),ls=':',label=f'E12 endpoint: {label}')
    axes[0,1].set(title='Fixed heldout forward fit',xlabel='Extra value updates',ylabel='RMSE',yscale='log');axes[0,1].legend(fontsize=7)
    with np.load(root/'baseline_field/field.npz') as f:nodes=f['nodes'].copy();idx=cKDTree(nodes).query(f['path'])[1]
    for label,path in [('E12',root/'query_augmentation_1000/neural_evaluation/field.npz'),('E15',out/'neural_evaluation/field.npz'),('Teacher',root/'roadmap_metric/canonical_field.npz')]:
        with np.load(path) as f:np.testing.assert_array_equal(f['nodes'],nodes);v=f['values'][idx];walls=f['walls'].copy()
        axes[1,0].plot(v-v[0],label=label)
    axes[1,0].set(title='SAME canonical reference route',xlabel='Reference step',ylabel='d - d_start');axes[1,0].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'neural_evaluation'/f'task3_seed{seed}.npz') as f:axes[1,1].plot(f['state_values'],label=str(seed))
    axes[1,1].set(title='E15 all five task3 episodes',xlabel='Actual step',ylabel='Neural forward distance');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'goal_support.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,4),constrained_layout=True);points=[]
    for ax,seed in zip(axes,range(3100,3105)):
        name=f'task3_seed{seed}'
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path in [('E12',root/'query_augmentation_1000/neural_evaluation'/f'{name}.npz'),('E15',out/'neural_evaluation'/f'{name}.npz')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=f'{seed}: {"success" if cases[name]["success"] else "timeout"}',aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0].legend(fontsize=8);fig.savefig(out/'task3_goal_support.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(base)[k-1]}/5 | {counts(old)[k-1]}/5 | {counts(cases)[k-1]}/5 | {counts(teacher)[k-1]}/5 |' for k in range(1,6))
    task3='\n'.join(f"| {name} | {v['success']} | {v['steps']} | {v['strict_decrease_fraction']:.4f} | {v['increase_fraction']:.4f} |" for name,v in cases.items() if name.startswith('task3_'))
    report=f'''# E15：固定全局教师的目标支持均衡

固定1000步后{sum(counts(cases))}/25成功、task3 {counts(cases)[2]}/5；E12为23/25、task3 3/5。相对E12新增{gains}，失去{losses}；失去原100k成功例{base_losses}。预定critic门槛{gate}，不能用随机对拟合误差替代。总{a['elapsed_seconds']:.2f}秒（训练/核验{t['train_verify_seconds']:.2f}、独立神经评价{n['elapsed_seconds']:.2f}），低于300秒。

## 唯一变量与训练证据

同E12原100k IQE/value Adam/count/lr、双向等权教师MSE、source查询、batch256/1000步、原官方采样seed2001。偶数行仍原(s,g)，奇数source仍原数据s加同Sobol偏移，goal中心改为均匀遍历E10所有6302个实际数据原型，再加同Sobol goal偏移[-1,1]。幅度/比例/损失和source分布不变，没有新的随机seed、评价墙体/目标/路线标签或原型筛选。不是E7原QRL源权重实验；此处改变的是新全局监督问题的goal分布。

原型j编号乘固定互素stride={p['stride']}再模K；stride由ceil(K*(sqrt(5)−1)/2)起取首个互素整数，不扫描。128000个均衡goal每原型20或21次，全部保留，非挑任务邻域；扩展位置可能物理不可达，仅作抽象教师查询，不是新真实转移。M5已证明教师对角跳变，仍不要求全域一致拟合、不把近似先验当真实MDP最优距离。

原始流SHA {t['paired_stream_sha256']}同E6/E12；每批source逐位同重建E12，重建旧查询流SHA {t['old_queries_stream_sha256']}与E12相同。新查询SHA {t['queries_stream_sha256']}，标签SHA {t['targets_stream_sha256']}；控制E12直接复用，没有重训/重评episode。原/旧扩展/新均衡扩展三组验证同时记录，未按曲线改预算或挑最好。

三项采样器测试通过（人口质量/互素遍历，source/偶数goal/输入不变，分块复现/NumPy RNG不变）；原E11损失与优化器代码不改。实际两步smoke完整保存加载，正式回到原100k及RNG；终点count101000/step101001，全部状态有限/往返、actor/dynamics/lam参数和Adam矩冻结、同bank actor输出逐位核验通过。

## 完整真实执行

| 任务 | 原100k | E12 | E15 | 固定教师 |
|---|---:|---:|---:|---:|
{table}

![固定成功集、两组验证、同参考路线与五个task3距离](goal_support.png)

| task3 case | 成功 | 完整步数 | 严格下降比例 | 上坡比例 |
|---|---|---:|---:|---:|
{task3}

全部25例独立Python推理子进程仅载入QRL checkpoint、原forward IQE，无教师路径或图导入、无对称包装。121真实后继/T1所选实际后继逐位核验和精确缓存保持，完整1000步/成功终止，不按停滞早停。参考墙体图只在episode后评价；这不是官方actor成绩，原actor仍冻结。

## 三种验证分布及场

原验证forward/reverse RMSE {float(last['heldout/forward_rmse']):.5f}/{float(last['heldout/reverse_rmse']):.5f}，E12 {previous_stats['original']['forward_rmse']:.5f}/{previous_stats['original']['reverse_rmse']:.5f}。旧E12扩展验证{float(last['old_augmented_heldout/forward_rmse']):.5f}/{float(last['old_augmented_heldout/reverse_rmse']):.5f}，E12 {previous_stats['old_augmented']['forward_rmse']:.5f}/{previous_stats['old_augmented']['reverse_rmse']:.5f}。新均衡扩展验证{float(last['augmented_heldout/forward_rmse']):.5f}/{float(last['augmented_heldout/reverse_rmse']):.5f}，E12在同新批次只读评分{previous_stats['augmented']['forward_rmse']:.5f}/{previous_stats['augmented']['reverse_rmse']:.5f}。它们的目标分布不同，必须同组比较，不跨组当收敛优劣。

canonical通道{n['field']['strict_descent_from_anchor']}、峰差/IQR {n['field']['minimum_peak_gap_over_iqr']:.6f}，全图/corridor相关{n['field']['all_graph_spearman']:.6f}/{n['field']['corridor_spearman']:.6f}。仍为0.2参考图、锚点吸附误差0.14451的离散证据，不能替代真实失败。原pair平均/尾风险{float(last['original_risk/pair_mean_squared_positive']):.6f}/{float(last['original_risk/pair_tail_squared_positive']):.6f}、goal风险{float(last['original_risk/goal_squared_positive']):.6f}，lambda固定{float(last['original_risk/lam']):.3f}，不作新优化收敛宣称。

![全部五个task3的E12/E15轨迹](task3_goal_support.png)

## 出口与复现

本轮门槛未通过：均衡后task3成功集合没变，丢失E12的两个task4成功。三组验证RMSE均略改善而执行下降，说明即使改善另一种广泛查询分布上的平均拟合，也不足以保护关键动作排序；不能据此认定目标频次是充分根因，也不能泛化为所有目标采样均无效。停止本均衡比例/stride/幅度/lr/预算搜索，不启动actor训练。

下一项可另冻结针对高拟合残差的全局监督机制，检验平均损失是否掩盖少量关键误差；不能使用已有评价失败点回填训练，也不能扫描本均衡的参数挽救。此处只是后续假设，尚未运行。现有bank用于方法选择，不是独立跨seed验证；不换seed、不运行1M、不恢复已停程序，不宣称论文完整预算复现或未验证的稳定改善。对原QRL的失败解释不能偷换成“目标数据缺失”，本项没有重新做覆盖审计。当前保留的较好神经方法仍E12/E13的23/25、task3 3/5，总目标未完成。

协议[GOAL_SUPPORT_PROTOCOL.md](../../../experiments/ogbench_critic/GOAL_SUPPORT_PROTOCOL.md)，测试test_goal_support.py，入口run_goal_support.py <根目录> --output <全新目录>，报告report_goal_support.py <根目录>只读重建。raw results/ogbench_critic_20261029/goal_support_1000，日志同级.log，checkpoint model/params_101000.pkl SHA {t['checkpoint_sha256']}。原模型/教师/E12/源码/全部同bank轨迹/结束状态已核验。只提交代码、协议、测试、报告与引用图，原始数组/模型/日志不入库。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
