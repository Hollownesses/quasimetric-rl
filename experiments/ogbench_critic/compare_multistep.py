"""Analyze verified E2 endpoints; do not confuse new critic state with QRL state."""
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

from compare import paired_counts

ARMS=('baseline','one','multi')
NAMES={'baseline':'QRL 100k','one':'E2 one-step','multi':'E2 multistep'}
COLORS={'baseline':'#636d79','one':'#3478b5','multi':'#c84c45'}
UNIT=-np.log(.995)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_evaluation(info,arm,training,done,unit):
    """Reject intermediate, wrong-format, wrong-unit or mismatched V evaluations."""
    assert info['checkpoint']==str(training/arm)
    assert info['checkpoint_kind']=='multistep' and info['mode']==arm
    assert info['epoch']==10000 and info['training_extra_updates']==10000
    assert info['distance_unit']=='log-discount' and info['unit']==unit
    assert info['checkpoint_sha256']==digest(training/arm/'critic_10000.pkl')
    assert info['critic_digest']==done['final'][arm]['value_digest']


def endpoints(root):
    training=root/'multistep_10k'
    protocol=json.loads((training/'protocol.json').read_text())
    assert digest(training/'protocol.json')==(training/'protocol.sha256').read_text().strip()
    assert protocol['steps']==10000 and protocol['seed']==4501
    assert protocol['batch_size']==256 and protocol['uniform_batch_goal_samples']==16
    assert protocol['gamma']==.995 and protocol['distance_unit']==UNIT
    # Freeze proof and full optimizer/RNG roundtrip are produced by actual loading.
    done=json.loads((training/'training_complete.json').read_text())
    verified=json.loads((training/'checkpoint_verification.json').read_text())
    assert done['steps']==10000 and done['source_transitions_per_arm']==2560000
    for arm in ['one','multi']:
        assert done['final'][arm]['step']==done['final'][arm]['adam_count']==10000
        check=verified[arm]
        assert check['passed'] and check['extra_updates']==check['adam_count']==10000
        assert check['full_state_roundtrip'] and check['frozen_parameters_and_moments']
        assert check['bank_actor_actions_equal'] and check['initial_rank_spearman']>.999999
    # The running training source is retained even when later analysis files are added.
    for name in ['multistep.py','train_multistep.py','common.py','evaluate.py','field.py','verify_multistep.py']:
        assert digest(training/'source_snapshot'/name)==protocol['source_sha256'][name],name
    assert digest(training/'source_snapshot/MULTISTEP_PROTOCOL.md')==protocol['protocol_sha256']
    with (training/'metrics.csv').open(newline='') as f:
        rows=[{k:(v if k=='arm' else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]
    series={a:[r for r in rows if r['arm']==a] for a in ['one','multi']}
    for arm,logs in series.items():
        assert [int(r['step']) for r in logs]==[1,*range(500,10001,500)]
        assert all(np.isfinite(v) for r in logs for k,v in r.items() if k!='arm')
    greedy={a:json.loads((root/('baseline_greedy' if a=='baseline' else f'e2_{a}_greedy')/'summary.json').read_text()) for a in ARMS}
    fields={a:json.loads((root/('baseline_field' if a=='baseline' else f'e2_{a}_field')/'summary.json').read_text()) for a in ARMS}
    arrays={a:np.load(root/('baseline_field' if a=='baseline' else f'e2_{a}_field')/'field.npz') for a in ARMS}
    bank=digest(root/'evaluation_bank.npz')
    for arm in ARMS:
        g=greedy[arm];f=fields[arm]
        assert g['bank_sha256']==bank and g['controller']=='real-greedy'
        assert list(g['cases'])==list(greedy['baseline']['cases']) and len(g['cases'])==25
        assert g['action_grid_sha256']==greedy['baseline']['action_grid_sha256']
        assert f['spacing']==.2 and f['radius']==.7 and f['reference_replay_success']
        assert np.array_equal(arrays[arm]['nodes'],arrays['baseline']['nodes'])
        assert np.isfinite(arrays[arm]['values']).all() and f['distance_iqr']>0
        for task in range(1,6):
            cases=[v for k,v in g['cases'].items() if k.startswith(f'task{task}_')]
            assert len(cases)==g['tasks'][f'task{task}']['episodes']==5
            assert sum(v['success'] for v in cases)==g['tasks'][f'task{task}']['successes']
        trajectory_dir=root/('baseline_greedy' if arm=='baseline' else f'e2_{arm}_greedy')
        for name,case in g['cases'].items():
            assert 1<=case['steps']<=1000
            with np.load(trajectory_dir/f'{name}.npz') as t:
                assert t['observations'].shape==(case['steps']+1,2)
                assert t['actions'].shape==(case['steps'],2)
                assert np.isfinite(t['observations']).all() and np.isfinite(t['actions']).all()
        if arm!='baseline':
            for info in [g,f]:check_evaluation(info,arm,training,done,protocol['distance_unit'])
            assert f['descent_tolerance']==1e-6*UNIT
    return protocol,done,verified,series,greedy,fields,arrays


def interpret(greedy,fields):
    one=greedy['one']['tasks']['task3']['successes']
    multi=greedy['multi']['tasks']['task3']['successes']
    one_channel=fields['one']['strict_descent_from_anchor']
    multi_channel=fields['multi']['strict_descent_from_anchor']
    pairing=paired_counts(greedy['one']['cases'],greedy['multi']['cases'])
    if multi>one and multi_channel and not one_channel:
        return ('多步组比一步组改善task3执行并新增锚点下降通道，支持进一步检验备份长度的作用。'
                f'其他任务仍须检查配对得失（新增{pairing["wins"]}、失去{pairing["losses"]}），单个续训seed不构成稳定改善。')
    if one>0 and multi>0 and one_channel and multi_channel:
        return ('两组均恢复部分task3执行和锚点下降通道，不能将共同收益归于多步长度；'
                '共同改变的备份目标、状态动作表示及优化器重置仍需进一步分离。')
    if one==0 and multi==0 and not one_channel and not multi_channel:
        return ('两组均未恢复task3执行或锚点下降通道；本次固定预算不支持该IQE备份实现已经解决距离恢复。'
                '不挑中间模型或调权重挽救本轮；转而结合已完成的原生MQE和既有失败轨迹检验机制，不追加换seed或1M QRL。')
    return ('执行与通道没有同时支持清楚的多步收益，需按具体配对轨迹和排序限定解释。'
            '不能只凭较低loss、较小峰差或一个任务净计数宣称改善。')


def run(root):
    root=Path(root).resolve()
    protocol,done,verified,series,greedy,fields,arrays=endpoints(root)
    output=root/'e2_comparison';output.mkdir(exist_ok=True)
    pairings={f'{a}_to_{b}':paired_counts(greedy[a]['cases'],greedy[b]['cases'])
              for a,b in [('baseline','one'),('baseline','multi'),('one','multi')]}
    changes={f'{a}_to_{b}':dict(
        wins=[k for k,v in greedy[b]['cases'].items() if v['success'] and not greedy[a]['cases'][k]['success']],
        losses=[k for k,v in greedy[a]['cases'].items() if v['success'] and not greedy[b]['cases'][k]['success']])
        for a,b in [('baseline','one'),('baseline','multi'),('one','multi')]}
    idx=cKDTree(arrays['baseline']['nodes']).query(arrays['baseline']['path'])[1]
    path_values={a:arrays[a]['values'][idx] for a in ARMS}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    specs=[('train/backup','LINEX backup (log-discount units)',False),
           ('train/invariance','Action invariance penalty',False),
           ('heldout/pair_squared_positive_step_units','Held-out pair penalty (step units)',True),
           ('heldout/goal_squared_positive_step_units','Held-out goal penalty (step units)',True)]
    for ax,(key,title,log) in zip(axes.flat,specs):
        for a,rows in series.items():
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=NAMES[a],color=COLORS[a])
        if log:ax.set_yscale('symlog',linthresh=1e-6)
        ax.set(title=title,xlabel='New critic updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for a in ARMS:
        axes[1,1].plot((path_values[a]-path_values[a][0])/fields[a]['distance_iqr'],label=NAMES[a],color=COLORS[a])
    axes[1,1].set(title='Same original reference route',xlabel='Route steps',ylabel='(d - d_start) / IQR')
    axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=8)
    x=np.arange(5)
    for i,a in enumerate(ARMS):
        axes[1,2].bar(x+(i-1)*.24,[greedy[a]['tasks'][f'task{t}']['successes']/5 for t in range(1,6)],
                      width=.24,label=NAMES[a],color=COLORS[a])
    axes[1,2].set(title='True-successor greedy; 5 fixed cases / task',xticks=x,
                  xticklabels=[f'T{t}' for t in range(1,6)],ylim=(0,1.05),ylabel='Success fraction')
    axes[1,2].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,a in zip(axes,ARMS):
        v=arrays[a]['values'];nodes=arrays[a]['nodes']
        sc=ax.scatter(nodes[:,0],nodes[:,1],c=(v-np.quantile(v,.25))/fields[a]['distance_iqr'],
                      s=5,vmin=-1,vmax=3,cmap='viridis')
        for x,y,hx,hy in arrays[a]['walls']:
            ax.add_patch(Rectangle((x-hx,y-hy),2*hx,2*hy,color='#39414c'))
        path=arrays['baseline']['path'];ax.plot(path[:,0],path[:,1],c='white',lw=1)
        ax.scatter(*fields[a]['anchor'],c='red',s=25);ax.scatter(*fields[a]['goal'],c='cyan',marker='*',s=70)
        ax.set(title=NAMES[a],aspect='equal',xlim=(-2,22),ylim=(-2,22),xlabel='x',ylabel='y')
    fig.colorbar(sc,ax=list(axes),shrink=.8,label='(distance - Q25) / IQR')
    fig.savefig(output/'normalized_fields.png',dpi=180,bbox_inches='tight');plt.close(fig)
    tasks=['| 任务 | 原始QRL 100k | E2 one | E2 multi |','|---|---:|---:|---:|']
    for t in range(1,6):tasks.append(f'| task{t} | '+' | '.join(f"{greedy[a]['tasks'][f'task{t}']['successes']}/5" for a in ARMS)+' |')
    metrics=['| 指标 | 原始QRL 100k | E2 one | E2 multi |','|---|---:|---:|---:|']
    for key,label in [('minimum_peak_gap_over_iqr','全部路线最低峰差 / IQR'),
        ('strict_descent_reachable_fraction','整图严格下降可达比例'),
        ('shortest_route_descent_opportunity_fraction','有下降最短路后继的节点比例'),
        ('all_graph_spearman','整图成本/距离Spearman'),('corridor_spearman','出口走廊Spearman')]:
        metrics.append('| '+label+' | '+' | '.join(f'{fields[a][key]:.6g}' for a in ARMS)+' |')
    metrics.append('| 锚点下降通道 | '+' | '.join(str(fields[a]['strict_descent_from_anchor']) for a in ARMS)+' |')
    residual=['| 固定held-out终点 | one | multi |','|---|---:|---:|']
    for key,label in [('pair_squared_positive_step_units','pair平方正部（步数单位）'),
                      ('goal_squared_positive_step_units','goal平方正部（步数单位）'),
                      ('mean_goal_log_units','平均距离（log-discount）'),('mean_goal_step_units','平均距离（除以c）'),
                      ('goal_uphill_fraction','随机目标上坡比例')]:
        residual.append('| '+label+' | '+' | '.join(f'{series[a][-1]["heldout/"+key]:.6g}' for a in ['one','multi'])+' |')
    decision=interpret(greedy,fields)
    summary=dict(protocol_sha256=digest(root/'multistep_10k/protocol.json'),endpoint_new_updates=10000,
        source_transitions_per_arm=done['source_transitions_per_arm'],elapsed_seconds=done['elapsed_seconds'],
        verified=verified,task_results={a:greedy[a]['tasks'] for a in ARMS},fields=fields,
        heldout={a:series[a][-1] for a in ['one','multi']},paired_execution=pairings,paired_changes=changes,decision=decision)
    (output/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    pairing_text='\n'.join(f'- {a}：新增{p["wins"]}、失去{p["losses"]}；新增名单{changes[a]["wins"]}；失去名单{changes[a]["losses"]}。' for a,p in pairings.items())
    report=f'''# E2：MQE-style IQE的一步／多步固定预算对照

仅评价预先冻结的新critic各10,000次优化终点，每组{done['source_transitions_per_arm']:,}条源转移；配对训练共耗时{done['elapsed_seconds']/3600:.3f}小时。不是原官方110k模型，也不是原生MQE。完整新Adam计数、参数有限性、保存加载/RNG往返、原actor/dynamics参数与矩冻结及25例actor动作完全相同均已通过验证。

## 五任务真实后继贪心

{chr(10).join(tasks)}

原始模型仅作起点背景。因果备份长度对照是one与multi：两组共享原始100k初始化、缩放后的IQE、状态动作编码器、全新Adam、转移、目标及批内抽样；两者只改变waypoint/备份步长。与E1相比存在多处改变，不能将跨实验收益只归于多步。

{pairing_text}

同一25例完整integration state和目标、11×11动作网格，候选真实后继恢复与实际执行逐位一致；全部episode使用官方1000步TimeLimit。每任务5例是诊断样本，不能称稳定成功率或官方actor50episode评价。完整逐步轨迹均保留本地，形状、长度和有限性已经核验。

## 距离恢复与约束

{chr(10).join(metrics)}

同一0.2网格、半径0.7，参考路线均经真实MuJoCo执行成功。下降容差随单位乘c，峰差按各自IQR归一化，同路线曲线使用原模型预先固定路线。IQE输出为log-discount距离，不能把其原始数值直接与QRL步数尺度比较；下表除以c只是确定性换单位，不证明校准为最短到达成本。

最低峰差为零只说明存在一条不超过起点高度的图路线，不等于该路线每一步都严格下降；局部先下降再上升或平台仍可阻断贪心执行。此处两组仍无锚点严格下降通道，不能把零峰差称为陷阱已消除。

{chr(10).join(residual)}

两类正残差都是固定2048条验证样本的诊断；E2没有直接优化原pair/goal约束或lambda。随机行为转移可能远离随机目标，不把任意上坡都当作错误。LINEX loss和动作不变性也不能替代通道与执行检验。

one组终点平均距离仅1.18个原步数单位，pair/goal残差很小，但全图排序和执行明显退化。这是距离尺度/辨别能力收缩的证据，不是恢复长程最优距离。multi保留较大距离尺度并优于one的部分排序和task1执行，但仍比原始模型失去11个成功例，task3没有恢复。仅用16个批内目标估计、IQE与单critic等适配也限制了跨方法归因，不能据此宣布原生MQE无效。

![终点对照及训练曲线](comparison.png)

![IQR归一化距离场](normalized_fields.png)

## 决策与范围

{decision}

采用[MQE原文及官方代码](https://mqe-paper.github.io/)的备份思路，不将多步本身当作新贡献；本实现保留IQE、单critic、冻结actor及固定16个批内目标无偏估计，因此不能标记为原生MQE复现。终点数值、逐步轨迹与两张图已审阅，原生MQE 100k的既有结果另行比较；当前总研究目标未完成。

2026-10-30用户明确禁止以换seed证明非偶然性，也不运行1M QRL；两项对应程序已停止，按[CURRENT_LIMITS.md](../../../experiments/ogbench_critic/CURRENT_LIMITS.md)覆盖旧计划，不再把它们当作后续必要门槛。结论范围限定于现有模型、目标及执行证据，不声称跨seed/完整预算泛化。

原始日志与新critic bundle位于multistep_10k，逐步轨迹和图数组在e2_one/multi_greedy与e2_one/multi_field；机器可读比较位于本目录analysis.json。仅源码、测试、协议、说明、报告及引用PNG进入Git。
'''
    (output/'report.md').write_text(report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();run(a.root)
