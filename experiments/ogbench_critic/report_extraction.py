"""File-only paired extraction report; never imports a model or environment."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from report_iqe_distill import read,sha,counts


def run(root):
    root=Path(root).resolve();repo=Path(__file__).resolve().parents[2];out=root/'extraction_1000';a=read(out/'analysis.json');p=a['protocol'];t=a['training'];ev=a['evaluations']
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==600 and a['paired_250_initials_and_goals']
    assert t['passed'] and t['frozen_critic_and_lam'] and t['full_state_roundtrip'] and p['stage_steps']==1000 and p['batch_size']==256
    assert p['config']['alpha']==.0003 and p['config']['actor_loss']=='ddpgbc' and p['config']['actor_p_trajgoal']==1
    assert sha(root/'hard_goal_1000/analysis.json')==p['prior_analysis_sha256'] and sha(root/'hard_goal_1000/model/params_101000.pkl')==p['improved_sha256']
    assert t['stage_streams']['dynamics']['original_three_fields_sha256']==p['expected_dynamics_stream_sha256']
    assert sha(root/'evaluation_bank.npz')==p['fixed_bank_sha256']
    aliases={'official_qrl.py':repo/'experiments/ogbench_official_qrl/vendor/ogbench/impls/agents/qrl.py',
        'official_evaluation.py':repo/'experiments/ogbench_official_qrl/vendor/ogbench/impls/utils/evaluation.py',
        'core.py':repo/'experiments/ogbench_diagnostics/core.py','fixed_evaluate.py':Path(__file__).with_name('evaluate.py')}
    for name,digest in p['source_sha256'].items():assert sha(out/'source_snapshot'/name)==digest==sha(aliases.get(name,Path(__file__).with_name(name)))
    assert read(out/'smoke/verification.json')['passed'] and not (out/'failure.json').exists()
    fixed_base=read(root/'baseline_actor/summary.json')['cases'];paired_gains={};paired_losses={}
    for arm,n in ev.items():
        assert n['complete'] and n['only_neural_actor'] and not n['teacher_modules_imported'] and n['evaluation_rng']==0
        assert n['critic_digest']==p['initial_frozen'][arm]['modules_value']['params']
        assert n['official_evaluation_sha256']==p['source_sha256']['official_evaluation.py'] and n['official_qrl_sha256']==p['source_sha256']['official_qrl.py']
        assert len(n['official_episodes'])==250 and len(n['fixed_cases'])==25 and list(n['fixed_cases'])==list(fixed_base)
        for stage in ['dynamics','actor']:
            ck=t['checkpoints'][arm+'/'+stage];assert ck['full_state_roundtrip'] and ck['frozen'] and t['actual'][arm][stage]==1000
            assert ck['epoch']==p['initial_counts'][arm]+(1000 if stage=='dynamics' else 2000)
            assert sha(out/arm/stage/f"params_{ck['epoch']}.pkl")==ck['sha256']
        assert n['checkpoint_sha256']==t['checkpoints'][arm+'/actor']['sha256']
        for name,info in n['official_episodes'].items():
            with np.load(out/f'{arm}_evaluation/official'/f'{name}.npz') as f:
                assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2) and f['done'][-1]
                assert 0<info['steps']<=1000 and all(np.isfinite(f[k]).all() for k in f.files)
                np.testing.assert_array_equal(f['observations'][0],info['initial_xy']);np.testing.assert_array_equal(f['goal'],info['goal']);np.testing.assert_array_equal(f['observations'][-1],info['final_xy'])
        for name,info in n['fixed_cases'].items():
            with np.load(out/f'{arm}_evaluation/fixed_bank'/f'{name}.npz') as f,np.load(root/'baseline_actor'/f'{name}.npz') as original:
                assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2)
                np.testing.assert_array_equal(f['observations'][0],original['observations'][0]);np.testing.assert_array_equal(f['goal'],original['goal'])
                assert all(np.isfinite(f[k]).all() for k in f.files)
        for task in range(1,6):
            episodes=[v for v in n['official_episodes'].values() if v['task']==task];stats=n['official'][str(task)]
            assert len(episodes)==50 and sum(v['success'] for v in episodes)==stats['successes']
            assert stats['success_rate']==stats['official_statistics']['success']==stats['successes']/50
    aa=ev['original']['official_episodes'];bb=ev['improved']['official_episodes'];assert list(aa)==list(bb)
    for name in aa:
        np.testing.assert_array_equal(aa[name]['initial_xy'],bb[name]['initial_xy']);np.testing.assert_array_equal(aa[name]['goal'],bb[name]['goal'])
    for task in range(1,6):
        paired_gains[task]=[n for n in aa if aa[n]['task']==task and bb[n]['success'] and not aa[n]['success']]
        paired_losses[task]=[n for n in aa if aa[n]['task']==task and aa[n]['success'] and not bb[n]['success']]
    official_counts={arm:[n['official'][str(k)]['successes'] for k in range(1,6)] for arm,n in ev.items()}
    fixed_counts={arm:counts(n['fixed_cases']) for arm,n in ev.items()}
    fixed_gains=[n for n in fixed_base if ev['improved']['fixed_cases'][n]['success'] and not fixed_base[n]['success']]
    fixed_losses=[n for n in fixed_base if fixed_base[n]['success'] and not ev['improved']['fixed_cases'][n]['success']]
    gate=official_counts['improved'][2]>official_counts['original'][2] and all(official_counts['improved'][k]>=official_counts['original'][k] for k in [0,1,3,4])
    dyn=list(csv.DictReader((out/'dynamics_metrics.csv').open()));act=list(csv.DictReader((out/'actor_metrics.csv').open()))
    assert all(max(int(r['step']) for r in rows if r['arm']==arm)==1000 for rows in [dyn,act] for arm in ev)
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for i,arm in enumerate(ev):
        axes[0,0].bar(np.arange(5)+(i-.5)*.35,official_counts[arm],width=.35,label=f'{arm} critic + extraction')
        d=[r for r in dyn if r['arm']==arm];v=[r for r in act if r['arm']==arm]
        axes[0,2].plot([int(r['step']) for r in d],[float(r['heldout/dynamics_loss']) for r in d],label=arm)
        axes[1,0].plot([int(r['step']) for r in v],[float(r['heldout/actor/mse']) for r in v],label=arm)
        axes[1,1].bar(np.arange(5)+(i-.5)*.35,fixed_counts[arm],width=.35,label=arm)
    axes[0,0].set(title='Official evaluator; same 250 starts/goals',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylabel='Successes / 50',ylim=(0,55));axes[0,0].legend(fontsize=7)
    axes[0,1].bar(np.arange(5),[len(paired_gains[k]) for k in range(1,6)],label='B gains over A');axes[0,1].bar(np.arange(5),[-len(paired_losses[k]) for k in range(1,6)],label='B loses A successes');axes[0,1].axhline(0,color='gray',lw=.8)
    axes[0,1].set(title='Paired official successful-set differences',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylabel='Episodes');axes[0,1].legend(fontsize=7)
    axes[0,2].set(title='Dynamics-only stage; critic fixed',xlabel='Dynamics updates',ylabel='Official heldout dynamics loss',yscale='log');axes[0,2].legend(fontsize=7)
    axes[1,0].set(title='Actor-only stage; critic/dynamics fixed',xlabel='Actor updates',ylabel='Official behavior-action MSE');axes[1,0].legend(fontsize=7)
    axes[1,1].set(title='Existing 25-case actor bank',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylabel='Successes / 5',ylim=(0,5.5));axes[1,1].legend(fontsize=7)
    labels=['Dyn A+B','Actor A+B','Evaluate A','Evaluate B'];times=[t['stage_seconds']['dynamics'],t['stage_seconds']['actor'],ev['original']['elapsed_seconds'],ev['improved']['elapsed_seconds']]
    axes[1,2].bar(range(4),times);axes[1,2].set(title='Measured stage times (setup excluded)',xticks=range(4),xticklabels=labels,ylabel='Seconds');axes[1,2].tick_params(axis='x',labelsize=8)
    fig.savefig(out/'extraction.png',dpi=160);plt.close(fig)
    with np.load(root/'hard_goal_1000/neural_evaluation/field.npz') as f:walls=f['walls'].copy()
    fig,axes=plt.subplots(2,5,figsize=(17,8),constrained_layout=True);points=[]
    for row in range(2):
        for col,ax in enumerate(axes[row]):
            name=f'task3_seed{3100+col}' if row==0 else f'task3_episode{col:03d}';subdir='fixed_bank' if row==0 else 'official'
            for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
            for arm in ev:
                with np.load(out/f'{arm}_evaluation'/subdir/f'{name}.npz') as f:
                    ax.plot(*f['observations'].T,label=arm,lw=1.1);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
            ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=('Fixed ' if row==0 else 'Official ')+name.split('_')[-1].replace('seed','case'),aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes.flat:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0,0].legend(fontsize=7);fig.savefig(out/'task3_extraction.png',dpi=160);plt.close(fig)
    official_table='\n'.join(f"| task{k} | {official_counts['original'][k-1]}/50 | {official_counts['improved'][k-1]}/50 | {len(paired_gains[k])} | {len(paired_losses[k])} |" for k in range(1,6))
    fixed_table='\n'.join(f"| task{k} | {counts(fixed_base)[k-1]}/5 | {fixed_counts['original'][k-1]}/5 | {fixed_counts['improved'][k-1]}/5 |" for k in range(1,6))
    report=f'''# E17：冻结critic的官方动力学与actor提取对照

两组各完成1000次动力学及1000次actor更新，全流程{a['elapsed_seconds']:.2f}秒/600秒上限。官方评价器在相同250起点/目标上，A原critic提取{sum(official_counts['original'])}/250（{sum(official_counts['original'])/2.5:.1f}%），B改进critic提取{sum(official_counts['improved'])}/250（{sum(official_counts['improved'])/2.5:.1f}%）；task3分别{official_counts['original'][2]}/50、{official_counts['improved'][2]}/50。预定逐任务改善门槛为{gate}；固定bank的B相对原actor新增{fixed_gains}、失去{fixed_losses}。不能将已通过的E16神经贪心门槛当成actor利用成功。

## 受控提取与冻结证据

A为原100k QRL，B为E16固定1000终点；开始时actor/dynamics/lam参数/Adam矩及agent RNG完全相同，value不同。B已有额外1000次价值更新，不声称同总训练预算。本项匹配额外提取更新和批次；两组先仅官方dynamics_loss各1000，再仅官方actor_loss(ddpgbc)各1000，critic/lam全程冻结，actor阶段dynamics也冻结。

原alpha0.0003、lr3e−4、512×3网络、const_std、Q归一化/BC和100%未来actor目标均不改；batch256/seed2001连续2000个原官方批次，阶段间不重置，两组共享每批。无教师/图/评价目标/模拟器搜索训练标签。阶段化与冻结不同于官方main联合优化，属于提取适配实验，不是论文完整预算。

优化器原矩/count保留，冻结模块不经Adam。A/B最后计数102000/103000，分别代表这套分阶段更新次数，不能称官方联合102k/103k。测试确认当前FP32范围的bias correction均为1，1000计数差不改变相同梯度的Adam更新；没有伪造count。agent RNG按官方split顺序更新，阶段NumPy sampler状态也保存本地。

三项测试通过：官方模块梯度/非零矩冻结与RNG/count、FP32计数偏差、记录适配器透明转发。实际smoke每组各两步动力学/两步actor，保存加载完整一致，再重置正式初态；各阶段固定终点全agent/Adam/RNG有限/往返与冻结检查通过。固定2048验证上的critic输出逐位未变，已验证E16距离函数因而保留，未浪费时间重跑其贪心评价。

动力学前三字段流SHA {t['stage_streams']['dynamics']['original_three_fields_sha256']}同E16；含actions/actor_goals全流SHA {t['stage_streams']['dynamics']['full_actor_fields_sha256']}。actor阶段全流SHA {t['stage_streams']['actor']['full_actor_fields_sha256']}。没有查看中间成功率或按最好模型选终点。

## 官方全部五任务，每任务50episode

| 任务 | A原critic+同预算提取 | B改进critic+提取 | B新增A失败 | B失去A成功 |
|---|---:|---:|---:|---:|
{official_table}

每组全新独立Python进程直接调用固定官方utils.evaluation.evaluate和原QRLAgent.sample_actions，temperature0、gaussian=None、video0。只加deadline/reset/终止记录，不改数值语义；原训练数据加载返回但不用于决策，未导入教师/图/候选筛选模块。全部250起点/目标逐位匹配，不只是相同seed声明。评价RNG0是同样样本流控制，不是换训练seed或独立初始化复现。

上述A为新提取控制，不冒充原100k的那次评价逐例重放；原100k官方结果[50,50,0,50,50]/50、80%仅为历史背景。每个task保存官方全部统计、50条完整观测/动作/goal/奖励/done和末状态。均值为五任务等权，配对得失同时报告。

![全任务官方计数、配对得失、训练统计、原bank与耗时](extraction.png)

## 既有25-case actor bank

| 任务 | 原100k actor | A提取 | B提取 |
|---|---:|---:|---:|
{fixed_table}

既有扰动case全部保留，完整1000步/成功终止，无新增seed或重跑原基线。E16 critic的25/25是真实后继贪心，表中是直接actor，两者必须分开。上方官方250例与此bank也不可混为同一成功率。

![全部五个既有task3，以及预定官方task3前五episode；无挑最好](task3_extraction.png)

## 结论范围及复现

本轮通过预定逐任务改善门槛：task3从0/50到31/50，其他四任务的200例成功全部保留；B官方成功集合对A严格扩展31例、无失去。固定bank恢复三个task3、原actor20成功均保留。说明改进critic可被同一官方提取损失实际利用，效果不是额外actor预算单独解释：A接受同样提取预算，task3仍0/50。

仍有19/50 task3失败，既有bank两例也未恢复，不能宣称task3完全解决或把这些失败全部归于actor。E16贪心25/25来自已知方法选择库；本次250例是冻结终点的新评价流，未用于调E16/E17，但仍为同训练初始化/固定环境范围。下一项应只读区分这些新查询上的critic和提取缺口，不继续调BC权重/lr/阶段长度/预算。

不声明跨seed稳定或官方完整预算最终缺陷，不把没有新初始化/1M当未完成的必需补跑。方法包含新增全局教师先验和阶段化适配，不能称官方QRL算法原样成绩；保留原100k、MQE和负结果对照，最终机制结论仍须限定证据强度。

协议[EXTRACTION_PROTOCOL.md](../../../experiments/ogbench_critic/EXTRACTION_PROTOCOL.md)，test_extraction.py；入口run_extraction.py <根目录> --output <全新目录>，独立评价evaluate_extraction.py由主入口调用，report_extraction.py只读重建。raw results/ogbench_critic_20261029/extraction_1000，日志同级.log与两个子评价.log。A checkpoint original/actor/params_102000.pkl SHA {t['checkpoints']['original/actor']['sha256']}；B improved/actor/params_103000.pkl SHA {t['checkpoints']['improved/actor']['sha256']}。原100k/E16源文件、官方源码、全部阶段checkpoint/源码/轨迹身份已核验；只提交代码/测试/协议/本报告和两张引用图，原始模型/CSV/JSON/NPZ/日志不提交。
'''
    (out/'report.md').write_text(report);print(out/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');run(p.parse_args().root)
