"""Analyze only fixed E1 endpoints, including both constraints and all five tasks."""
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


ARMS=('baseline','pair','goal')
NAMES={'baseline':'100k','pair':'A / pair','goal':'B / goal'}
COLORS={'baseline':'#5f6875','pair':'#3478b5','goal':'#c84c45'}


def endpoints(root):
    training=root/'paired_10k'
    protocol=json.loads((training/'protocol.json').read_text())
    assert protocol['steps']==10000 and protocol['seed']==2001
    assert hashlib.sha256((training/'protocol.json').read_bytes()).hexdigest()==(training/'protocol.sha256').read_text().strip()
    done=json.loads((training/'training_complete.json').read_text())
    assert done['steps']==10000
    verified=json.loads((training/'checkpoint_verification.json').read_text())
    for arm in ['pair','goal']:
        assert done['final'][arm]['adam_count']==110000
        assert done['final'][arm]['frozen']==protocol['initial_frozen']
        assert verified[arm]['passed'] and verified[arm]['bank_actor_actions_equal']
        assert (training/arm/'params_110000.pkl').is_file()
    with (training/'metrics.csv').open(newline='') as f:
        logs=[{k:(v if k=='arm' else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]
    expected=[1,*range(500,10001,500)]
    series={arm:[r for r in logs if r['arm']==arm] for arm in ['pair','goal']}
    for arm,rows in series.items():
        assert [int(r['step']) for r in rows]==expected
        assert all(np.isfinite(v) for row in rows for k,v in row.items() if k!='arm')
    greedy={arm:json.loads((root/f'{arm}_greedy/summary.json').read_text()) for arm in ARMS}
    fields={arm:json.loads((root/f'{arm}_field/summary.json').read_text()) for arm in ARMS}
    arrays={arm:np.load(root/f'{arm}_field/field.npz') for arm in ARMS}
    bank=hashlib.sha256((root/'evaluation_bank.npz').read_bytes()).hexdigest()
    for arm in ARMS:
        assert greedy[arm]['bank_sha256']==bank
        assert greedy[arm]['controller']=='real-greedy'
        assert len(greedy[arm]['cases'])==25
        assert list(greedy[arm]['cases'])==list(greedy['baseline']['cases'])
        assert greedy[arm]['action_grid_sha256']==greedy['baseline']['action_grid_sha256']
        assert fields[arm]['spacing']==.2 and fields[arm]['radius']==.7
        assert fields[arm]['reference_replay_success']
        assert np.array_equal(arrays[arm]['nodes'],arrays['baseline']['nodes'])
        if arm!='baseline':
            assert greedy[arm]['checkpoint']==str(training/arm)
            assert fields[arm]['checkpoint']==str(training/arm)
            expected_sha=hashlib.sha256((training/arm/'params_110000.pkl').read_bytes()).hexdigest()
            for evaluation in [greedy[arm],fields[arm]]:
                assert evaluation['epoch']==110000
                assert evaluation['checkpoint_sha256']==expected_sha
                assert evaluation['critic_digest']==done['final'][arm]['critic_digest']
    return protocol,done,verified,series,greedy,fields,arrays


def paired_counts(reference,treatment):
    counts=dict(wins=0,losses=0,both_success=0,both_failure=0,common_success_step_deltas=[])
    assert reference.keys()==treatment.keys()
    for name,a in reference.items():
        b=treatment[name]
        if b['success'] and not a['success']:counts['wins']+=1
        elif a['success'] and not b['success']:counts['losses']+=1
        elif a['success']:
            counts['both_success']+=1
            counts['common_success_step_deltas'].append(b['steps']-a['steps'])
        else:counts['both_failure']+=1
    return counts


def run(root):
    root=Path(root).resolve()
    protocol,done,verified,series,greedy,fields,arrays=endpoints(root)
    output=root/'e1_comparison';output.mkdir(exist_ok=True)
    pairings={f'{a}_to_{b}':paired_counts(greedy[a]['cases'],greedy[b]['cases'])
              for a,b in [('baseline','pair'),('baseline','goal'),('pair','goal')]}
    task4_pairing=paired_counts(
        {k:v for k,v in greedy['pair']['cases'].items() if k.startswith('task4_')},
        {k:v for k,v in greedy['goal']['cases'].items() if k.startswith('task4_')})
    idx=cKDTree(arrays['baseline']['nodes']).query(arrays['baseline']['path'])[1]
    fixed_path={arm:arrays[arm]['values'][idx] for arm in ARMS}
    fixed_probe={arm:dict(gap=float(v[15]-v[0]),
        relative_gap=float((v[15]-v[0])/((v[15]+v[0])/2+1e-6))) for arm,v in fixed_path.items()}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specs=[('heldout/pair_squared_positive','Held-out pair penalty',True),
           ('heldout/goal_squared_positive','Held-out goal penalty',True),
           ('heldout/d_goal_mean','Held-out mean goal distance',False),
           ('heldout/lam','Dual multiplier',False)]
    for ax,(key,title,log) in zip(axes.flat,specs):
        for arm,rows in series.items():
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],color=COLORS[arm],label=NAMES[arm])
        if 'penalty' in title:ax.axhline(.05,c='black',ls='--',lw=1,label='eps=0.05')
        if log:ax.set_yscale('log')
        ax.set(title=title,xlabel='Extra critic updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for arm in ARMS:
        normalized=(fixed_path[arm]-fixed_path[arm][0])/fields[arm]['distance_iqr']
        axes[1,1].plot(normalized,label=NAMES[arm],color=COLORS[arm])
    axes[1,1].set(title='Same reference route; distance change / IQR',xlabel='Route steps',ylabel='(d - d_start) / IQR')
    axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.2)
    x=np.arange(5)
    for i,arm in enumerate(ARMS):
        rates=[greedy[arm]['tasks'][f'task{t}']['successes']/5 for t in range(1,6)]
        axes[1,2].bar(x+(i-1)*.24,rates,width=.24,label=NAMES[arm],color=COLORS[arm])
    axes[1,2].set(title='True-successor greedy; 5 fixed cases / task',xticks=x,xticklabels=[f'T{t}' for t in range(1,6)],ylim=(0,1.05),ylabel='Success fraction')
    axes[1,2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,arm in zip(axes,ARMS):
        nodes=arrays[arm]['nodes'];v=arrays[arm]['values']
        norm=(v-np.quantile(v,.25))/fields[arm]['distance_iqr']
        sc=ax.scatter(nodes[:,0],nodes[:,1],c=norm,s=5,vmin=-1,vmax=3,cmap='viridis')
        for x,y,hx,hy in arrays[arm]['walls']:
            ax.add_patch(Rectangle((x-hx,y-hy),2*hx,2*hy,color='#39414c'))
        path=arrays['baseline']['path'];ax.plot(path[:,0],path[:,1],c='white',lw=1)
        ax.scatter(*fields[arm]['anchor'],c='red',s=25);ax.scatter(*fields[arm]['goal'],c='cyan',marker='*',s=70)
        ax.set(title=NAMES[arm],aspect='equal',xlim=(-2,22),ylim=(-2,22),xlabel='x',ylabel='y')
    fig.colorbar(sc,ax=list(axes),shrink=.8,label='(distance - Q25) / IQR')
    fig.savefig(output/'normalized_fields.png',dpi=180,bbox_inches='tight');plt.close(fig)
    task_rows=['| 任务 | 原始100k | A：pair续训10k | B：goal续训10k |','|---|---:|---:|---:|']
    for task in range(1,6):
        task_rows.append('| task'+str(task)+' | '+' | '.join(f"{greedy[arm]['tasks'][f'task{task}']['successes']}/5" for arm in ARMS)+' |')
    metric_rows=['| 指标 | 原始100k | A | B |','|---|---:|---:|---:|']
    field_metrics=[('minimum_peak_gap','全部路线最低峰值差'),('minimum_peak_gap_over_iqr','峰值差 / 距离IQR'),
        ('distance_iqr','整图距离IQR'),('strict_descent_reachable_fraction','整图下降可达比例'),
        ('shortest_route_descent_opportunity_fraction','存在下降最短路后继的节点比例'),
        ('all_graph_spearman','整图参考成本/距离Spearman'),('corridor_spearman','出口走廊Spearman')]
    for key,label in field_metrics:
        metric_rows.append('| '+label+' | '+' | '.join(f'{fields[arm][key]:.5g}' for arm in ARMS)+' |')
    metric_rows.append('| 锚点严格下降通道 | '+' | '.join(str(fields[a]['strict_descent_from_anchor']) for a in ARMS)+' |')
    loss_rows=['| Held-out终点统计 | A | B |','|---|---:|---:|']
    for key,label in [('pair_squared_positive','pair平方正部'),('goal_squared_positive','goal平方正部'),
                      ('pair_positive_fraction','pair正残差比例'),('goal_positive_fraction','goal正残差比例'),
                      ('goal_uphill_fraction','随机目标上坡比例'),('d_goal_mean','平均目标距离'),('lam','lambda')]:
        loss_rows.append('| '+label+' | '+' | '.join(f'{series[a][-1]["heldout/"+key]:.6g}' for a in ['pair','goal'])+' |')
    a3=greedy['pair']['tasks']['task3']['successes'];b3=greedy['goal']['tasks']['task3']['successes']
    new_channel=fields['goal']['strict_descent_from_anchor'] and not fields['pair']['strict_descent_from_anchor']
    if b3>a3 and new_channel:
        decision='B同时改善task3执行及锚点下降通道，A未达到同样结果；支持继续独立复核目标相关梯度方向，但还不能称为稳定改善。必须检查其他任务损失和B放松pair约束的代价。'
    elif a3>0 and b3>0:
        decision='A、B都改善了task3执行，额外critic预算是重要因素。不能把收益全部归于目标相关约束；需要进一步分离预算和目标改变。'
    elif b3<=a3 and not fields['goal']['strict_descent_from_anchor']:
        decision='B没有同时改善task3执行和锚点下降通道。本轮停止继续搜索这个约束的权重，下一项转向有明确离线目标的多步学习，并纳入MQE比较。'
    else:
        decision='执行与距离场指标没有同时支持明确结论；应根据具体配对轨迹及排序差异限定解释，不能只凭一个指标宣布成功。'
    summary=dict(endpoint_extra_updates=10000,protocol_sha256=hashlib.sha256((root/'paired_10k/protocol.json').read_bytes()).hexdigest(),
        checkpoint_verification=verified,task_results={arm:greedy[arm]['tasks'] for arm in ARMS},
        fields=fields,heldout_endpoint={arm:series[arm][-1] for arm in ['pair','goal']},
        paired_execution=pairings,decision=decision,training_elapsed_seconds=done['elapsed_seconds'])
    summary['fixed_M1_probe_endpoint']=fixed_probe
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    text=f'''# E1：目标相关距离差与官方pair约束的固定终点对照

本报告只使用预先固定的额外10,000更新终点（总Adam count=110000），不选择中间checkpoint。A/B从相同100k模型和Adam状态恢复，同一采样调用产生批次与目标；actor/dynamics参数和Adam矩冻结。两组计数、全状态有限性、冻结证明及同库actor输出检查均通过。正式配对训练耗时{done['elapsed_seconds']/3600:.3f}小时。

## 五任务真实后继贪心执行

{chr(10).join(task_rows)}

这是固定11×11真实后继贪心诊断，全部episode遵循官方1000步TimeLimit。每任务5个固定扰动，不是官方actor评价或统计稳定性证明。actor被冻结，因此本轮不可能改善它在同一状态的动作。

A相对原始模型：新增成功{pairings['baseline_to_pair']['wins']}、失去{pairings['baseline_to_pair']['losses']}。B相对原始模型：新增成功{pairings['baseline_to_goal']['wins']}、失去{pairings['baseline_to_goal']['losses']}。B相对A：新增成功{pairings['pair_to_goal']['wins']}、失去{pairings['pair_to_goal']['losses']}。配对分母均为同一25例。

A/B的task1成功数分别为{greedy['pair']['tasks']['task1']['successes']}/5和{greedy['goal']['tasks']['task1']['successes']}/5；task4中B相对A新增{task4_pairing['wins']}、失去{task4_pairing['losses']}。任务级计数必须结合配对得失解释，不能只报有利的净计数。完整配对名单与轨迹保留本地。

## 距离场

{chr(10).join(metric_rows)}

三模型使用同一0.2间距、半径0.7的碰撞检查图；参考路线均经真实MuJoCo执行成功。锚点吸附和连续最优性限制沿用E0报告。最低峰值差既报告原尺度，也按IQR归一化；严格下降可达性与秩相关不受统一正缩放和平移影响。曲线使用同一条E0参考路线，不为每个模型选择最有利路线。

B的局部峰值差在归一化后仍明显下降，说明不能把它完全解释成统一尺度变化；但峰值差仍为正，且固定贪心执行没有恢复task3。局部地形变化不足以构成稳定闭环改善。

![两类约束、距离尺度、同参考路线与五任务执行](comparison.png)

![按各模型IQR归一化的距离场](normalized_fields.png)

## 两类约束与距离尺度

{chr(10).join(loss_rows)}

Held-out为固定验证批次2048条；随机目标下的上坡比例只作诊断，任意行为转移远离任意目标本来可能合理，不能将其全部当作失败。B的约束由三角不等式可知弱于pair约束，所以B自身penalty下降并不证明保留了原局部可行性。必须结合pair统计、尺度和执行结果解释。

B的终点goal平方正部虽在eps=0.05内，但仍高于A；正残差比例较低并不等于平方正部更低。其pair平方正部约为A的{series['goal'][-1]['heldout/pair_squared_positive']/series['pair'][-1]['heldout/pair_squared_positive']:.2f}倍，这一代价与执行退化必须同时报告。

## 与M1梯度诊断的关系

在M1相同的两个固定probe节点上，原始/A/B的终点距离差分别为{fixed_probe['baseline']['gap']:.4f}、{fixed_probe['pair']['gap']:.4f}、{fixed_probe['goal']['gap']:.4f}，相对上坡分别为{fixed_probe['baseline']['relative_gap']:.6f}、{fixed_probe['pair']['relative_gap']:.6f}、{fixed_probe['goal']['relative_gap']:.6f}。M1的一步B反事实平均增大该上坡，而10k后的B反而减小了它；这直接说明不能将一次局部梯度投影线性外推为长期结论。长期终点仍必须看通道和执行。

另一个范围限制：M1只分解value_loss，E1也只优化value_loss或其goal替换；官方端到端QRL的value参数还会受到dynamics_loss的梯度。A因此不是完整官方联合训练的预算对照。原方法完整预算是否恢复，仍由独立官方1M运行检验；不能凭E1把完整算法失败唯一归因于某个损失分量。

## 本轮决策与尚未完成的目标

{decision}

该结论只覆盖当前100k初始化、续训采样seed2001、单环境及固定25例诊断。稳定改善仍需独立采样/初始化、其他目标、官方完整预算基线以及actor利用新critic的单独验证。用户的总研究目标尚未完成；不凭这次单对照收尾。

若进入多步，遵循[MQE官方论文与代码](https://mqe-paper.github.io/)所涉及的多步准度量学习背景，不将任意行为路线长度强行作为最优距离等式，也不把多步本身视为新贡献。

源码、测试、协议、报告及引用PNG提交Git；模型、CSV/JSON/NPZ、轨迹、日志和其他中间产物保留本地。完整数值和配对结果在同目录analysis.json，原始日志见paired_10k/metrics.csv。
'''
    (output/'report.md').write_text(text)
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
