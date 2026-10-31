"""Read-only E6 mean extraction integrity audit and three-way Chinese report."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from report_iqe_distill import read,sha,counts


def difference(old,new):
    return dict(gains=[k for k in old if new[k]['success'] and not old[k]['success']],
                losses=[k for k in old if old[k]['success'] and not new[k]['success']])


def run(root,output):
    root=Path(root).resolve();out=Path(output).resolve();repo=Path(__file__).resolve().parents[2]
    a=read(out/'analysis.json');p=a['protocol'];t=a['training'];c=a['evaluation'];e17=read(root/'extraction_1000/analysis.json')
    assert a['complete'] and a['elapsed_seconds']<p['time_limit_seconds']==600 and a['paired_250_initials_and_goals_with_e17'] and not a['e17_rerun']
    assert t['passed'] and t['actual']==dict(dynamics=1000,actor=1000) and t['stage_streams']==e17['training']['stage_streams']==p['expected_stage_streams']
    assert t['frozen_critic_and_lam'] and t['full_state_roundtrip'] and p['stage_steps']==1000 and p['batch_size']==256
    assert sha(p['source_checkpoint'])==p['source_sha256'] and sha(root/'tail_1000/analysis.json')==p['prior_analysis_sha256']
    assert sha(root/'extraction_1000/analysis.json')==p['e17_analysis_sha256'] and sha(root/'evaluation_bank.npz')==p['fixed_bank_sha256']
    for name,item in e17['training']['checkpoints'].items():
        assert sha(root/'extraction_1000'/name/f"params_{item['epoch']}.pkl")==item['sha256']
    for arm in ['original','improved']:
        assert read(root/f'extraction_1000/{arm}_evaluation/analysis.json')==e17['evaluations'][arm]
    assert read(out/'smoke/verification.json')['passed'] and not (out/'failure.json').exists()
    aliases={'official_qrl.py':repo/'experiments/ogbench_official_qrl/vendor/ogbench/impls/agents/qrl.py',
             'official_evaluation.py':repo/'experiments/ogbench_official_qrl/vendor/ogbench/impls/utils/evaluation.py',
             'core.py':repo/'experiments/ogbench_diagnostics/core.py','fixed_evaluate.py':Path(__file__).with_name('evaluate.py')}
    for name,digest in p['source_files_sha256'].items():assert sha(out/'source_snapshot'/name)==digest==sha(aliases.get(name,Path(__file__).with_name(name)))
    for stage,epoch in [('dynamics',102000),('actor',103000)]:
        ck=t['checkpoints'][stage];assert ck['epoch']==epoch and ck['frozen'] and ck['full_state_roundtrip']
        assert sha(out/'mean'/stage/f'params_{epoch}.pkl')==ck['sha256']
    assert c['complete'] and c['only_neural_actor'] and not c['teacher_modules_imported'] and c['evaluation_rng']==0
    assert c['checkpoint_sha256']==t['checkpoints']['actor']['sha256'] and c['critic_digest']==p['initial_frozen']['modules_value']['params']
    assert c['official_evaluation_sha256']==p['source_files_sha256']['official_evaluation.py'] and c['official_qrl_sha256']==p['source_files_sha256']['official_qrl.py']
    assert len(c['official_episodes'])==250 and len(c['fixed_cases'])==25
    ev={'A 原100k':e17['evaluations']['original'],'C E6 mean':c,'B E16':e17['evaluations']['improved']}
    for group in ev.values():
        assert list(group['official_episodes'])==list(c['official_episodes']) and group['bank_sha256']==p['fixed_bank_sha256']
        for name,info in c['official_episodes'].items():
            np.testing.assert_array_equal(info['initial_xy'],group['official_episodes'][name]['initial_xy'])
            np.testing.assert_array_equal(info['goal'],group['official_episodes'][name]['goal'])
    for task in range(1,6):
        episodes=[v for v in c['official_episodes'].values() if v['task']==task];stats=c['official'][str(task)]
        assert len(episodes)==50 and sum(v['success'] for v in episodes)==stats['successes']
        assert stats['success_rate']==stats['official_statistics']['success']==stats['successes']/50
    for name,info in c['official_episodes'].items():
        with np.load(out/'mean_evaluation/official'/f'{name}.npz') as f:
            assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2) and f['done'][-1]
            assert 0<info['steps']<=1000 and all(np.isfinite(f[k]).all() for k in f.files)
            np.testing.assert_array_equal(f['observations'][0],info['initial_xy']);np.testing.assert_array_equal(f['goal'],info['goal']);np.testing.assert_array_equal(f['observations'][-1],info['final_xy'])
    for name,info in c['fixed_cases'].items():
        with np.load(out/'mean_evaluation/fixed_bank'/f'{name}.npz') as f:
            assert f['observations'].shape==(info['steps']+1,2) and f['actions'].shape==(info['steps'],2) and all(np.isfinite(f[k]).all() for k in f.files)
            for arm in ['original','improved']:
                with np.load(root/f'extraction_1000/{arm}_evaluation/fixed_bank'/f'{name}.npz') as reference:
                    np.testing.assert_array_equal(f['observations'][0],reference['observations'][0]);np.testing.assert_array_equal(f['goal'],reference['goal'])
    official={label:[v['official'][str(k)]['successes'] for k in range(1,6)] for label,v in ev.items()}
    fixed={label:counts(v['fixed_cases']) for label,v in ev.items()}
    paired={'C-A':difference(ev['A 原100k']['official_episodes'],c['official_episodes']),
            'B-C':difference(c['official_episodes'],ev['B E16']['official_episodes'])}
    bank_paired={'C-A':difference(ev['A 原100k']['fixed_cases'],c['fixed_cases']),
                 'B-C':difference(c['fixed_cases'],ev['B E16']['fixed_cases'])}
    paired_task={key:{str(task):{kind:[n for n in names if c['official_episodes'][n]['task']==task] for kind,names in values.items()} for task in range(1,6)} for key,values in paired.items()}
    rows={}
    for label in ev:
        directory=out if label=='C E6 mean' else root/'extraction_1000'
        arm='mean' if label=='C E6 mean' else 'original' if label=='A 原100k' else 'improved'
        for stage in ['dynamics','actor']:
            with (directory/f'{stage}_metrics.csv').open() as handle:rows[label,stage]=[r for r in csv.DictReader(handle) if r['arm']==arm]
            assert len(rows[label,stage])==11 and int(rows[label,stage][-1]['step'])==1000
    colors=['#888888','#d69331','#297aa3'];labels=['A: Original 100k','C: E6 mean','B: E16']
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for i,(label,short,color) in enumerate(zip(ev,labels,colors)):
        axes[0,0].bar(np.arange(5)+(i-1)*.25,official[label],width=.25,label=short,color=color)
        axes[1,1].bar(np.arange(5)+(i-1)*.25,fixed[label],width=.25,label=short,color=color)
        dyn=rows[label,'dynamics'];act=rows[label,'actor']
        axes[0,2].plot([int(r['step']) for r in dyn],[float(r['heldout/dynamics_loss']) for r in dyn],label=short,color=color)
        axes[1,0].plot([int(r['step']) for r in act],[float(r['heldout/actor/mse']) for r in act],label=short,color=color)
    axes[0,0].set(title='Official actor; matched 250 starts/goals',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,56),ylabel='Successes / 50');axes[0,0].legend(fontsize=8)
    for i,(key,color) in enumerate(zip(paired,['#d69331','#297aa3'])):
        x=np.arange(5)+(i-.5)*.34;task=paired_task[key]
        axes[0,1].bar(x,[len(task[str(k)]['gains']) for k in range(1,6)],width=.34,label=f'{key}: gains',color=color)
        axes[0,1].bar(x,[-len(task[str(k)]['losses']) for k in range(1,6)],width=.34,label=f'{key}: losses',color=color,alpha=.45)
    axes[0,1].axhline(0,color='gray',lw=.7);axes[0,1].set(title='Paired successful-set differences',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylabel='Episodes');axes[0,1].legend(fontsize=8)
    axes[0,2].set(title='Official heldout dynamics loss',xlabel='Dynamics updates',ylabel='Loss',yscale='log');axes[0,2].legend(fontsize=8)
    axes[1,0].set(title='Official heldout behavior-action MSE',xlabel='Actor updates',ylabel='MSE');axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Existing 25-case actor bank',xticks=range(5),xticklabels=[f'T{k}' for k in range(1,6)],ylim=(0,5.6),ylabel='Successes / 5');axes[1,1].legend(fontsize=8)
    times=[t['stage_seconds']['dynamics'],t['stage_seconds']['actor'],c['elapsed_seconds']]
    axes[1,2].bar(['Dynamics C','Actor C','Evaluate C'],times,color='#d69331');axes[1,2].set(title='Only new arm C was run',ylabel='Seconds')
    fig.savefig(out/'mean_extraction.png',dpi=160);plt.close(fig)
    with np.load(root/'hard_goal_1000/neural_evaluation/field.npz') as f:walls=f['walls'].copy()
    fig,axes=plt.subplots(2,5,figsize=(17,8),constrained_layout=True);points=[]
    for row in range(2):
        for col,ax in enumerate(axes[row]):
            name=f'task3_seed{3100+col}' if row==0 else f'task3_episode{col:03d}';subdir='fixed_bank' if row==0 else 'official'
            for wall in walls:ax.add_patch(Rectangle(wall[:2]-wall[2:],*(2*wall[2:]),color='.85'))
            for label,short,color in zip(ev,labels,colors):
                path=out/'mean_evaluation' if label=='C E6 mean' else root/f"extraction_1000/{'original' if label=='A 原100k' else 'improved'}_evaluation"
                with np.load(path/subdir/f'{name}.npz') as f:
                    ax.plot(*f['observations'].T,label=short,color=color,lw=1.1);goal=f['goal'];points.extend([f['observations'].copy(),goal[None]])
            ax.scatter(*goal,marker='*',s=70,color='green');ax.set(title=('Fixed ' if row==0 else 'Official ')+name.split('_')[-1],aspect='equal')
    points=np.concatenate(points);lo=points.min(0)-1;hi=points.max(0)+1
    for ax in axes.flat:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
    axes[0,0].legend(fontsize=7);fig.savefig(out/'task3_mean_extraction.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| task{k} | {official['A 原100k'][k-1]}/50 | {official['C E6 mean'][k-1]}/50 | {official['B E16'][k-1]}/50 |" for k in range(1,6))
    table+='\n| 平均 | '+' | '.join(f'{sum(v)/2.5:.1f}%' for v in official.values())+' |'
    fixed_table='\n'.join(f"| task{k} | {fixed['A 原100k'][k-1]}/5 | {fixed['C E6 mean'][k-1]}/5 | {fixed['B E16'][k-1]}/5 |" for k in range(1,6))
    pair_table='\n'.join(f"| task{k} | {len(paired_task['C-A'][str(k)]['gains'])} / {len(paired_task['C-A'][str(k)]['losses'])} | {len(paired_task['B-C'][str(k)]['gains'])} / {len(paired_task['B-C'][str(k)]['losses'])} |" for k in range(1,6))
    lam=float(read(root/'tail_1000/initial_statistics.json')['lam'])
    with (root/'tail_1000/metrics.csv').open() as handle:mean_final=[r for r in csv.DictReader(handle) if r['arm']=='mean' and int(r['step'])==1000][0]
    mean_lam=float(mean_final['heldout/lam'])
    c3=official['C E6 mean'][2];b3=official['B E16'][2]
    interpretation=(f'普通mean续训在相同提取下task3为{c3}/50，E16为{b3}/50；因此不能仅用“额外1000次普通价值更新加相同提取”解释E16在此流上的全部收益。' if c3<b3 else
                    f'普通mean续训在相同提取下task3为{c3}/50，达到或超过E16的{b3}/50；原先把额外价值续训排除为收益来源的表述需要收窄，不能只凭E17-A就把收益归于新增方法。')
    summary=dict(passed=True,official_counts=official,fixed_counts=fixed,paired=paired,paired_per_task=paired_task,bank_paired=bank_paired,
                 initial_lambda=lam,mean_heldout_lambda=mean_lam,matching_e17_training_streams=True,matched_official_inputs=250,
                 verified_new_trajectories=275,e17_models_or_evaluations_rerun=False,checkpoint_sha256=t['checkpoints']['actor']['sha256'])
    (out/'report_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    report=f'''# E18：E6 mean接E17同预算动力学与actor提取

2026-11-01。仅新增C组，从已保存E6 mean终点进行1000次dynamics及1000次actor更新，{a['elapsed_seconds']:.2f}秒完成，低于600秒上限；A/B直接复用原E17，没有重训或重新评价。{interpretation} 逐任务与配对得失如下，不能只比较task3或平均数。

## 起点、控制与核验

C为tail_1000/mean/params_101000.pkl，SHA {p['source_sha256']}，已完成官方value_loss1000次续训，actor/dynamics及其Adam矩、agent RNG逐位同原100k。完整原Adam/count101000保留；value/lam参数及矩全程冻结，actor阶段动力学亦冻结。两阶段固定终点count102000/103000、TrainState.step102001/103001，均有限、全状态保存加载一致，2048验证critic输出逐位不变。此次没有新增critic更新或重做贪心诊断。

直接复用且核对E17源码SHA的extraction.py、evaluate_extraction.py及官方QRL/evaluate。batch256、lr0.0003、alpha0.0003、512×3、const_std、官方100%未来actor目标不变；连续2000批np采样2001，阶段间不重置。dynamics和actor两个阶段的三字段及含actions/actor_goals完整流均逐位同E17。固定2048验证SHA {p['validation_sha256']}亦同。每100步训练/验证日志，未做中间成功率选择或追加预算。

新增一项测试通过：改变小网络lambda参数，官方dynamics/actor的损失、统计与活动模块梯度逐位不变。实际两阶段各2步smoke后完整保存加载，再回到E6 mean原状态/采样2001正式运行。C沿用E6更新过的lambda，不冒充同E16起始完整状态；原网络乘子约{lam:.6f}，E6 mean价值终点约{mean_lam:.6f}。本轮lambda参数及Adam矩自C起点起逐位冻结，未重置为原值。

## 全部官方任务，各50episode

| 任务 | A 原100k+提取（复用） | C E6 mean+提取（新增） | B E16+提取（复用） |
|---|---:|---:|---:|
{table}

独立子进程原官方actor/evaluate，temperature0、gaussian=None、video0，无图/教师/模拟器搜索参与决策。所有250起点及goal与A/B逐位匹配，不只是声明相同seed。评价rng0是同输入控制，不是换训练seed；原25-case bank也未增加扰动种子。全部275条新增完整轨迹的观测/动作/goal、步数、有限性、done及末状态通过只读报告审计。

| 任务 | C相对A：新增 / 失去 | B相对C：新增 / 失去 |
|---|---:|---:|
{pair_table}

C-A合计新增{len(paired['C-A']['gains'])}、失去{len(paired['C-A']['losses'])}；B-C合计新增{len(paired['B-C']['gains'])}、失去{len(paired['B-C']['losses'])}。完整逐例名单保存在report_audit.json，不根据结果筛去任务或失败例。

![三组官方成功数、配对得失、同验证损失、bank与C组耗时](mean_extraction.png)

## 原25-case直接actor bank

| 任务 | A 原100k+提取 | C E6 mean+提取 | B E16+提取 |
|---|---:|---:|---:|
{fixed_table}

C-A新增{bank_paired['C-A']['gains']}，失去{bank_paired['C-A']['losses']}；B-C新增{bank_paired['B-C']['gains']}，失去{bank_paired['B-C']['losses']}。这里是直接actor，与E6当时的mean神经贪心16/25分开，也不与官方250例混为一项指标。

![全部五个既有task3及预定官方task3前五例，三组同输入；未挑成功](task3_mean_extraction.png)

## 解释范围

{interpretation} 这是在看到E17结果后补做的控制，不能冒充原预注册第三组或新的独立验证；所有设置在C运行前冻结，C没有按结果选终点、调参数或追加预算。

C与B均为原100k加1000价值更新、1000动力学更新、1000actor更新，补齐了价值更新次数的控制。前期价值目标/目标查询/教师先验/候选前向成本不同，lambda更新也不同；λ不被这两个官方提取损失读取且本轮冻结，不将其改成另一个起点。此项不单独分离候选多样性与最大误差选择，不声称等总计算量、唯一训练因果、跨seed稳定或完整论文复现。其他四任务及配对失去必须与task3一起解释，保留所有负结果。

## 复现与原始产物

协议[MEAN_EXTRACTION_PROTOCOL.md](../../../experiments/ogbench_critic/MEAN_EXTRACTION_PROTOCOL.md)，源码run_mean_extraction.py、report_mean_extraction.py，新增测试test_mean_extraction.py。从仓库根目录：

```bash
P=experiments/ogbench_official_qrl/.venv/bin/python
E=experiments/ogbench_critic
G=results/ogbench_critic_20261029
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
$P "$E/test_mean_extraction.py"
# 提供复现命令，本次只执行了一次正式C，不自动重复：
$P "$E/run_mean_extraction.py" "$G" --output "$G/mean_extraction_new"
# 当前报告只读重建，不训练/评价：
$P "$E/report_mean_extraction.py" "$G" --output "$G/mean_extraction_1000"
```

raw目录results/ogbench_critic_20261029/mean_extraction_1000，同级mean_extraction_1000.log及mean_extraction_tests.log；子评价evaluation.log。dynamics checkpoint为mean/dynamics/params_102000.pkl，SHA {t['checkpoints']['dynamics']['sha256']}；actor终点mean/actor/params_103000.pkl，SHA {t['checkpoints']['actor']['sha256']}。协议/source_snapshot、阶段CSV、sampler状态、275轨迹和report_audit.json保留本地。原E6/原100k/E17文件未改变。只提交源码/测试/协议/报告/引用图，不提交checkpoint/JSON/CSV/NPZ/日志。
'''
    (out/'report.md').write_text(report);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
