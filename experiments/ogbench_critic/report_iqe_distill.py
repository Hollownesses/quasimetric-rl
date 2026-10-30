"""File-only verification and report of E11; no training or inference."""
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
def counts(cases):return [sum(v['success'] for n,v in cases.items() if n.startswith(f'task{k}_')) for k in range(1,6)]


def run(root):
    root=Path(root).resolve();out=root/'iqe_distillation_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];n=a['neural']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==300
    assert t['passed'] and t['steps']==1000 and t['adam_count']==101000
    assert t['full_state_roundtrip'] and t['frozen_parameters_and_moments'] and t['frozen_bank_actor_outputs']
    assert t['paired_stream_sha256']==p['expected_stream_sha256']
    assert sha(out/'model/params_101000.pkl')==t['checkpoint_sha256']==n['checkpoint_sha256']
    assert n['complete'] and n['only_neural_scores'] and n['candidate_evaluator_source_verified']
    assert not n['roadmap_module_imported'] and not n['data_graph_module_imported']
    assert sha(root/'evaluation_bank.npz')==n['bank_sha256']
    assert sha(root/'roadmap_metric/analysis.json')==p['teacher_analysis_sha256']
    assert sha(root/'roadmap_metric/graph.npz')==p['teacher_graph_sha256']
    assert sha(root/'roadmap_metric/distances.npy')==p['teacher_distance_sha256']
    assert sha(root/'tail_1000/analysis.json')==p['control_analysis_sha256']
    assert sha(root/'tail_1000/mean/params_101000.pkl')==p['control_checkpoint_sha256']
    assert sha(out/'validation_targets.npy')==p['validation_targets_sha256']
    for name,digest in p['source_sha256'].items():
        directory=Path(__file__).parents[1]/'ogbench_diagnostics' if name in ['core.py','fast_pointmaze.py'] else Path(__file__).parent
        assert sha(out/'source_snapshot'/name)==digest==sha(directory/name)
    assert read(out/'smoke/verification.json')['passed']
    assert not (out/'failure.json').exists() and not (out/'neural_evaluation/failure.json').exists()
    old=read(root/'baseline_greedy/summary.json')['cases'];cases=n['cases'];teacher=read(root/'roadmap_metric/analysis.json')['cases']
    assert list(old)==list(cases)==list(teacher) and len(cases)==25
    for name,info in cases.items():
        with np.load(out/'neural_evaluation'/f'{name}.npz') as trace:
            assert trace['observations'].shape==(info['steps']+1,2) and trace['actions'].shape==(info['steps'],2)
            assert trace['state_values'].shape==(info['steps']+1,) and trace['final_integration'].ndim==1
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            np.testing.assert_array_equal(trace['observations'][-1],info['final_xy'])
            with np.load(root/'baseline_greedy'/f'{name}.npz') as before:
                np.testing.assert_array_equal(trace['observations'][0],before['observations'][0]);np.testing.assert_array_equal(trace['goal'],before['goal'])
    probe=read(out/'local_probe/analysis.json');assert probe['post_hoc'] and probe['new_episodes']==0 and probe['seconds']<30
    assert probe['source_sha256']==sha(Path(__file__).with_name('probe_distillation.py'))
    assert len(probe['probes'])==10
    rows=list(csv.DictReader((out/'metrics.csv').open()));assert int(rows[-1]['step'])==1000
    steps=np.array([float(r['step']) for r in rows]);initial=read(out/'initial_statistics.json');control=read(out/'control_statistics.json');last=rows[-1]
    gains=[name for name in old if cases[name]['success'] and not old[name]['success']]
    losses=[name for name in old if old[name]['success'] and not cases[name]['success']]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for i,(label,c) in enumerate([('Original QRL',old),('Fixed data teacher',teacher),('Neural IQE student',cases)]):
        ax.bar(np.arange(5)+(i-1)*.25,counts(c),width=.25,label=label)
    ax.set(title='Same 25-case true-successor greedy',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.7),ylabel='Successes / 5');ax.legend(fontsize=8)
    for direction in ['forward','reverse']:
        axes[0,1].plot(np.r_[0,steps],np.r_[initial[f'{direction}_rmse'],[float(r[f'heldout/{direction}_rmse']) for r in rows]],label=direction)
    axes[0,1].axhline(control['forward_rmse'],color='gray',ls=':',label='Original-loss control endpoint')
    axes[0,1].set(title='Fixed 2048 heldout pairs',xlabel='Extra value updates',ylabel='Teacher-fit RMSE',yscale='log');axes[0,1].legend(fontsize=8)
    with np.load(root/'baseline_field/field.npz') as f:nodes=f['nodes'].copy();idx=cKDTree(nodes).query(f['path'])[1]
    for label,path in [('Original-loss control',root/'tail_1000/mean_field.npz'),('Fixed teacher',root/'roadmap_metric/canonical_field.npz'),('Neural student',out/'neural_evaluation/field.npz')]:
        with np.load(path) as f:
            np.testing.assert_array_equal(f['nodes'],nodes);v=f['values'][idx];walls=f['walls'].copy()
        axes[1,0].plot(v-v[0],label=label)
    axes[1,0].set(title='SAME canonical route; evaluation only',xlabel='Reference graph step',ylabel='d - d_start');axes[1,0].legend(fontsize=8)
    for seed in range(3100,3105):
        with np.load(out/'neural_evaluation'/f'task3_seed{seed}.npz') as f:
            axes[1,1].plot(f['state_values'],label=f'{seed}: {"success" if cases[f"task3_seed{seed}"]["success"] else "timeout"}')
    axes[1,1].set(title='Student: all five task3 cases',xlabel='Actual environment step',ylabel='Neural forward distance');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'iqe_distillation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(17,7),constrained_layout=True);points=[]
    for col,seed in enumerate(range(3100,3105)):
        name=f'task3_seed{seed}';ax=axes[0,col]
        for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
        for label,path in [('Teacher',root/'roadmap_metric/greedy'/f'{name}.npz'),('Student',out/'neural_evaluation'/f'{name}.npz')]:
            with np.load(path) as f:ax.plot(*f['observations'].T,label=label,lw=1.2);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
        ax.scatter(*goal,marker='*',color='green',s=70);ax.set(title=f'{seed}: {"success" if cases[name]["success"] else "timeout"}',aspect='equal',xlim=(2,14),ylim=(6,20))
        if col==0:ax.legend(fontsize=7)
        with np.load(out/'local_probe'/f'{name}_end.npz') as f:
            tv=f['teacher'];sv=f['student'];axes[1,col].scatter(tv-tv.min(),sv-sv.min(),s=5)
        axes[1,col].set(title=f'End candidates: RMSE {probe["probes"][name+"_end"]["rmse"]:.1f}',xlabel='Teacher excess distance',ylabel='Student excess distance')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes[0]:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    fig.savefig(out/'task3_distillation.png',dpi=160);plt.close(fig)
    table='\n'.join(f'| task{k} | {counts(old)[k-1]}/5 | {counts(teacher)[k-1]}/5 | {counts(cases)[k-1]}/5 |' for k in range(1,6))
    probes='\n'.join(f"| {name} | {v['rmse']:.3f} | {v['spearman']:.3f} | {v['teacher_regret_of_student']:.3f} | {v['teacher_best_displacement']:.3f} / {v['student_best_displacement']:.3f} |" for name,v in probe['probes'].items())
    report=f'''# E11：原IQE学习全局数据距离，恢复部分task3执行

固定1000次更新后，独立神经推理22/25成功，task3为2/5；原16个成功全部保留。教师为25/25。神经学生未通过预定task3五例全部成功门槛，因此不启动actor训练、不追加本MSE方案预算，也不宣称总目标完成。训练、核验及独立推理共{a['elapsed_seconds']:.2f}秒，其中训练/核验{t['train_verify_seconds']:.2f}秒、独立评价{n['elapsed_seconds']:.2f}秒，低于预定300秒。

## 学习问题与对照

原100k QRL的IQE网络、原value Adam矩/count、lr3e−4不变，只更新value。目标替换为固定E10教师的双向等权平方误差；教师包含数据图、局部可逆及几何连接先验，是近似距离而非官方最优步数标签。没有原push/pair/dual更新，actor/dynamics/lam参数与其Adam矩均冻结。故这是新监督学习问题，不是官方算法原样续训或单纯数值校准。

原官方train.sample(256)采样1000次、seed2001；s/s_next/value_goal流SHA {t['paired_stream_sha256']}与E6控制相同。A复用原损失1000终点，不重训或重评；其教师拟合forward RMSE为{control['forward_rmse']:.5f}。训练目标100%原随机目标，不拿评价目标作标签。标签流SHA {t['targets_stream_sha256']}。先两步完整保存/加载smoke，再从原100k重置正式运行，不恢复smoke；终点Adam count101000。

三个方法测试通过：双向MSE定义、无actor/dynamics/lam梯度、带非零Adam矩的冻结和计数。终点全agent/Adam/RNG数值有限且保存加载完全一致；同25起点actor输出逐位不变，不能声称原actor成功率提高。

## 完整真实执行与距离场

| 任务 | 原QRL/原损失控制 | 数据教师 | 神经学生 |
|---|---:|---:|---:|
{table}

新增{gains}，失去{losses}。task3成功3102/3103分别69/74步；其余完整1000步超时，没有停滞早停。学生仍使用原forward QRL value，不做对称包装。独立Python子进程只接收原模型路径，完成全部121动作真实后继枚举；启动及结束检查未导入roadmap_metric/data_graph，教师不在决策链中。T1源码身份核验、每个所选实际后继逐位核验、首次缓存命中复算均保留。墙体图只在episode结束后评价，不能混作推理用图。

![全任务、拟合曲线、同参考路线与全部task3距离](iqe_distillation.png)

固定验证2048对的forward/reverse RMSE从{initial['forward_rmse']:.5f}/{initial['reverse_rmse']:.5f}降至{float(last['heldout/forward_rmse']):.5f}/{float(last['heldout/reverse_rmse']):.5f}，paired MSE {initial['loss']:.5f}→{float(last['heldout/loss']):.5f}；目标均值47.9683，学生均值48.1788/48.1936，方向差均值0.49773。原pair平均/尾部风险0.01211/0.09326、goal风险0.000642；lambda固定183.248，没有优化含义上的新收敛结论。

canonical全图/corridor Spearman {n['field']['all_graph_spearman']:.6f}/{n['field']['corridor_spearman']:.6f}，最低峰差/IQR为0，但锚点严格下降通道仍为False。锚点吸附误差0.14451；无需超过起点高度不等于全程下降，不能以相关性和峰差替代实际三个失败。

## 事后只读教师/学生动作排序检查

完整评价后另用{probe['seconds']:.2f}秒（独立30秒上限）检查五个已存起点和五个已存终点，每点121个真实后继，无新episode或训练。该探针是事后解释，不是冻结主实验的独立验证。

| 已存状态 | 局部绝对RMSE | 动作评分Spearman | 学生选择的教师regret | 教师/学生选择位移L∞ |
|---|---:|---:|---:|---:|
{probes}

三个失败终点的局部拟合RMSE约18.9/42.3/30.2，明显大于随机验证1.1；学生选中位移近零，教师同候选仍有0.16–0.20位移并降低教师距离。3101/3104起点学生动作甚至增加教师距离。两个成功终点RMSE仅0.31/0.21。绝对误差不能单独解释排序，表中regret和相关性补充了动作方向证据；这支持存在任务相关查询上的拟合/推广与局部排序缺口，尚未区分采样、表示能力和优化原因。

![全部task3轨迹与终点动作评分，横纵均扣除各自最小值](task3_distillation.png)

## 结论与复现

全局监督可以让原IQE在推理完全无图时恢复两个task3并保留原成功集；网络并非完全不能表达任何有效绕行。但普通随机对上的低MSE不足以保证任务查询的距离梯度和完整执行。教师成功、神经部分恢复、官方actor未变必须分开。结果限已有模型和已知25-case方法选择库，不是跨seed稳定性或论文完整预算复现；seed1及1M保持停止。

后续应检验任务无关的查询分布/局部形状学习机制，不继续延长本loss、搜索学习率或针对三个评价目标回填标签。协议[IQE_DISTILL_PROTOCOL.md](../../../experiments/ogbench_critic/IQE_DISTILL_PROTOCOL.md)，入口`run_iqe_distill.py <根目录> --output <全新目录>`；独立推理`evaluate_neural.py`由主入口调用；事后探针`probe_distillation.py`；本报告`report_iqe_distill.py <根目录>`只读重建。原始目录`results/ogbench_critic_20261029/iqe_distillation_1000`，日志同级`.log`，checkpoint `model/params_101000.pkl` SHA {t['checkpoint_sha256']}。原模型、教师、控制、源码快照、数据流、同bank轨迹及结束状态均已核对。只提交代码/测试/协议/本报告和两张引用图，原始结果/模型不提交。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');run(parser.parse_args().root)
