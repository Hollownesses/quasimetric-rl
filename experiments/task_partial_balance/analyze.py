"""Post-budget common evaluation of reused R0/R1 and the sole new Rhalf arm."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import evaluate,graph_inputs,graph_structure,decisions,statistics
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.task_source_balance.analyze import decomposition
from experiments.full_replay_four_path.train import EpochSampler
from experiments.four_path_factorial.run import file_digest,tensor_digest
from .prepare import verify_existing,FIELDS
from .train import load_training_data,mixed_weights,equal,write

ARMS=('R0','Rhalf','R1')


def verify_half(root,p,data):
    folder=root/'Rhalf';cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'COMPLETE').read_text())
    weights,stats,base=mixed_weights(data,.5)
    sig=dict(protocol_sha256=file_digest(root/'protocol.json'),task_weight_mode='partial-source',alpha=.5,
             weights_sha256=stats['weights_sha256'],fingerprints=data.fingerprint)
    assert cfg['training_signature']==done['training_signature']==sig
    assert cfg['alpha']==.5 and cfg['mixed_weights']==stats
    assert done['step']==50000 and not done['reference_loaded']
    assert done['final_sha256']==file_digest(folder/'final.pt')
    initial=torch.load(folder/'initial.pt',weights_only=True);final=torch.load(folder/'final.pt',weights_only=True)
    for name,step in [('initial.pt',0),('phr.pt',45000),('final.pt',50000)]:
        ck=torch.load(folder/name,weights_only=True)
        assert ck['step']==step and ck['training_signature']==sig and ck['fingerprints']==data.fingerprint
        assert ck['alpha']==.5 and tensor_digest(ck['task_weights'])==stats['weights_sha256']
    old=Path(p['prior_root'])
    for arm in ('R0','R1'):
        i=torch.load(old/arm/'initial.pt',weights_only=True);f=torch.load(old/arm/'final.pt',weights_only=True)
        for key in FIELDS:assert equal(initial[key],i[key]),key
        for key in ('sampler','visits','rng','push_counters','sampling_chain_sha256'):assert equal(final[key],f[key]),key
    lines=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
    scans=[r for r in lines if 'connected_expert' in r]
    assert [r['step'] for r in scans]==list(range(0,50001,1000))
    lr_logs=[r for r in lines if 'lr' in r]
    assert [r['step'] for r in lr_logs]==list(range(100,50001,100))
    from .train import learning_rate
    for r in lr_logs:
        assert r['lr']==learning_rate(r['step'],45000,5000)
        c=r['connected_expert_components'];assert c['optimized_pair_rows']==c['pair_rows']==c['task_rows']
    sampler=EpochSampler(len(data.original),2307);visits=torch.zeros_like(final['visits']);chain='0'*64
    for _ in range(50000):
        ids=sampler.next(256);visits[ids]+=1
        chain=hashlib.sha256(bytes.fromhex(chain)+data.original[ids].numpy().tobytes()).hexdigest()
    assert chain==final['sampling_chain_sha256']==done['sampling_chain_sha256']
    assert torch.equal(visits,final['visits']) and equal(sampler.state_dict(),final['sampler'])
    return dict(initial_complete_state_exact_to_both_old_arms=True,zero_dual=not bool(initial['dual'].any()),
        final_sampler_visits_rng_counters_exact=True,whole_ordered_sequence_reconstructed=True,
        sampling_chain_sha256=chain,steps=50000,visits_sum=int(visits.sum()),
        visits_min=int(visits.min()),visits_max=int(visits.max()),full_audits=len(scans),
        alpha=.5,weights_mean=stats['mean']),scans


def repeat_groups(data,ref,predictions,bins):
    rows=data.original.numpy();src=data.src.numpy();unique,inverse,counts=np.unique(src[rows],return_inverse=True,return_counts=True)
    groups={}
    for spec in bins:
        mask=counts>=spec['min']
        if spec['max'] is not None:mask &= counts<=spec['max']
        ids=unique[mask];record_ids=src[rows][mask[inverse]]
        if not len(ids):continue
        stats={}
        for arm,pred in predictions.items():
            e=pred[ids]-ref[ids]
            stats[arm]=dict(unique_sources=statistics(pred[ids],ref[ids]),records=statistics(pred[record_ids],ref[record_ids]),
                overestimated_sources=int((e>1e-5).sum()),underestimated_sources=int((e < -1e-5).sum()),
                max_absolute_error=float(abs(e).max()),p90_absolute_error=float(np.quantile(abs(e),.9)))
        changes={}
        for other in ('R0','R1'):
            d=abs(predictions['Rhalf'][ids]-ref[ids])-abs(predictions[other][ids]-ref[ids])
            changes['Rhalf_minus_'+other]=dict(improved_sources=int((d < -1e-5).sum()),worsened_sources=int((d>1e-5).sum()),
                unchanged_sources=int((abs(d)<=1e-5).sum()),mean_absolute_error_delta=float(d.mean()),
                largest_improvement=float(d.min()),largest_deterioration=float(d.max()))
        record_mass=len(record_ids)/len(rows);source_mass=len(ids)/len(unique)
        groups[spec['label']]=dict(n_sources=len(ids),n_records=len(record_ids),min_multiplicity=int(counts[mask].min()),
            max_multiplicity=int(counts[mask].max()),reference_mean=float(ref[ids].mean()),
            task_mass=dict(R0=record_mass,Rhalf=.5*(record_mass+source_mass),R1=source_mass),metrics=stats,changes=changes)
    assert sum(g['n_sources'] for g in groups.values())==len(unique)
    assert sum(g['n_records'] for g in groups.values())==len(rows)
    return groups,unique,counts


def validate_saved(data,ref,starts,results,predictions,details,arrays):
    src=data.src.numpy();dst=data.dst.numpy();cost=data.cost.double().numpy();max_decomp=0.;max_regret=0.
    unique=np.unique(src[data.original.numpy()])
    groups=dict(records=src[data.original.numpy()],unique_sources=unique,unique_starts=np.unique(starts),starts=starts,
        longest_quartile=unique[ref[unique]>=np.quantile(ref[unique],.75)])
    for arm in ARMS:
        pred=predictions[arm];r=results[arm]
        for group,ids in groups.items():assert statistics(pred[ids],ref[ids])==r['distance'][group]
        gap=arrays[arm+'_gap'];edge=arrays[arm+'_edge_slack'];triangle=arrays[arm+'_triangle_slack']
        assert np.array_equal(gap,ref-pred)
        error=float(abs(gap-edge-triangle).max());assert error<1e-7;max_decomp=max(error,max_decomp)
        for b in details[arm]['branches']:
            aa=b['alternatives'];scores=[cost[a['row']]+pred[dst[a['row']]] for a in aa]
            assert aa[int(np.argmin(scores))]['row']==b['chosen_row']
        unique_rollouts={}
        for e in details[arm]['episodes']:
            rr=np.array(e['rows']);assert e['status']=='success'
            assert src[rr[0]]==e['start'] and np.array_equal(dst[rr[:-1]],src[rr[1:]])
            assert ref[dst[rr[-1]]]==0 and abs(cost[rr].sum()-e['cost'])<1e-9
            advantage=float((cost[rr]+ref[dst[rr]]-ref[src[rr]]).sum())
            err=abs(advantage-e['regret']);assert err<1e-8;max_regret=max(max_regret,err)
            unique_rollouts[e['start']]=e
        assert len(unique_rollouts)==165
        assert abs(np.mean([e['regret'] for e in unique_rollouts.values()])-r['decisions']['unique_start_rollouts']['mean_successful_regret'])<1e-9
    return dict(passed=True,distance_statistics_recomputed=True,greedy_choices_checked=True,
        path_cost_and_advantage_regret_checked=True,decomposition_identity_max_error=max_decomp,
        path_regret_identity_max_error=max_regret)


def run(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());torch.set_num_threads(1)
    assert json.loads((root/'Rhalf/COMPLETE').read_text())['step']==50000
    for path,h in p['source_files'].items():assert file_digest(path)==h,path
    for path,h in p['reused_artifacts'].items():assert file_digest(path)==h,path
    data=load_training_data(p['data']);verify_existing(data);pairing,logs=verify_half(root,p,data)
    # All references below are strictly post-budget. No training module imports this file.
    refroot=Path('results/connected_reference_fix_20261024/main');info=json.loads((refroot/'summary.json').read_text())
    assert info['passed'] and info['fingerprints']==data.fingerprint
    assert file_digest(refroot/'graph_reference_audit_only.npz')==info['reference_sha256']
    with np.load(refroot/'graph_reference_audit_only.npz') as f:
        ref=f['distance'].copy();next_edge=f['next_edge'].copy()
        for k,t in [('source',data.src),('target',data.dst),('physical',data.physical)]:assert np.array_equal(f[k],t.numpy())
    validate_reference(data,ref);graph,starts=graph_inputs(data);structure=graph_structure(data,graph,starts)
    assert structure['acyclic'] and structure['longest_possible_recorded_rollout']==275
    assert structure['branches']==58 and structure['unique_starts']==165
    oracle,_=decisions(data,graph,ref,ref,starts,300)
    assert oracle['summary']['optimal_branch_choices']==58 and oracle['summary']['mean_successful_regret']<1e-8
    out=root/'comparison';out.mkdir(exist_ok=False)
    oldroot=Path(p['prior_root']);prior=json.loads((oldroot/'comparison/summary.json').read_text())
    results={};stages={};predictions={};arrays={};details={};all_logs={}
    for arm in ARMS:
        folder=root/arm if arm=='Rhalf' else oldroot/arm
        for step,name in [(50000,'final.pt'),(45000,'phr.pt')]:
            path=folder/name;before=file_digest(path)
            result,pred,detail=evaluate(data,graph,ref,starts,path)
            result['decisions']['strict_reference_pairs']=sum(b['strict_reference_pairs'] for b in detail['branches'])
            result['decisions']['correctly_ordered_pairs']=sum(b['correctly_ordered_pairs'] for b in detail['branches'])
            diag,pieces=decomposition(data,ref,next_edge,path,pred,starts);result['decomposition']=diag
            assert file_digest(path)==before
            if arm!='Rhalf':assert result==prior['results' if step==50000 else 'stage45000'][arm],('old evaluation changed',arm,step)
            if step==50000:
                results[arm]=result;predictions[arm]=pred;details[arm]=detail
                arrays[arm+'_prediction']=pred;arrays.update({arm+'_'+k:v for k,v in pieces.items()})
                write(out/(arm+'_decisions.json'),detail)
            else:stages[arm]=result
            print(json.dumps(dict(arm=arm,step=step,distance=result['distance'],decisions=result['decisions'])),flush=True)
        all_logs[arm]=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines() if 'connected_expert"' in s]
    groups,unique,counts=repeat_groups(data,ref,predictions,p['multiplicity_bins'])
    validation=validate_saved(data,ref,starts,results,predictions,details,arrays)
    summary=dict(results=results,assessment=assess(results),stage45000=stages,multiplicity=groups,pairing=pairing,validation=validation,
        graph_structure=structure,weights=p['mixed_weights'],fingerprints=data.fingerprint,
        protocol_sha256=file_digest(root/'protocol.json'),reference_sha256=info['reference_sha256'],
        reused_evaluations_exactly_match_prior=True,official_endpoint=50000,training=all_logs,
        evaluation_source_sha256=file_digest(__file__),scope='single seed606; 4 threads; fixed finite graph; only alpha=.5 added')
    write(out/'summary.json',summary);write(out/'multiplicity.json',groups)
    np.savez_compressed(out/'arrays.npz',reference=ref,starts=starts,source_nodes=unique,source_counts=counts,**arrays)
    shutil.copy2(__file__,out/'analysis_source.py')
    report(out,summary);plot(out,summary)
    write(out/'COMPLETE',dict(passed=True,summary_sha256=file_digest(out/'summary.json'),report_sha256=file_digest(out/'report.md'),validation=validation))
    artifacts=[root/'protocol.json',root/'runtime.json',root/'reuse_verification.json',root.parent/'tests.log',root.parent/'analysis_tests.log',
        out/'summary.json',out/'report.md',out/'arrays.npz',out/'comparison.png',out/'multiplicity.json',out/'analysis_source.py']
    artifacts += [root/'Rhalf'/name for name in ('initial.pt','phr.pt','final.pt','COMPLETE','config.json','training.jsonl')]
    write(root/'final_integrity.json',dict(passed=True,pairing=pairing,validation=validation,
        old_artifacts_unchanged=all(file_digest(path)==h for path,h in p['reused_artifacts'].items()),
        artifacts={str(path):file_digest(path) for path in artifacts}))
    write(root/'COMPLETE',dict(passed=True,official_endpoint=50000,final_integrity_sha256=file_digest(root/'final_integrity.json')))


def assess(results):
    """Descriptive multi-metric positions; no weighted score or checkpoint selection."""
    getters={
        'unique_source_nrmse':lambda x:x['distance']['unique_sources']['nrmse_percent'],
        'farthest_nrmse':lambda x:x['distance']['longest_quartile']['nrmse_percent'],
        'start165_nrmse':lambda x:x['distance']['unique_starts']['nrmse_percent'],
        'records_mae':lambda x:x['distance']['records']['mae'],
        'unique_source_abs_bias':lambda x:abs(x['distance']['unique_sources']['bias']),
        'farthest_abs_bias':lambda x:abs(x['distance']['longest_quartile']['bias']),
        'negative_optimal_branches':lambda x:-x['decisions']['optimal_branch_choices'],
        'negative_ranking_accuracy':lambda x:-x['decisions']['pairwise_order_accuracy'],
        'branch_regret':lambda x:x['decisions']['mean_branch_regret'],
        'path165_regret':lambda x:x['decisions']['unique_start_rollouts']['mean_successful_regret'],
        'negative_optimal_paths165':lambda x:-x['decisions']['unique_start_rollouts']['optimal_rollouts'],
        'maximum_violation':lambda x:x['local']['unique_edges']['max_positive'],
        'unique_edge_mean_violation':lambda x:x['local']['unique_edges']['mean_positive'],
        'record_mean_violation':lambda x:x['local']['records']['mean_positive'],
    }
    positions={}
    for name,get in getters.items():
        values={a:float(get(results[a])) for a in ARMS};h=values['Rhalf'];low=min(values['R0'],values['R1']);high=max(values['R0'],values['R1'])
        pos='better_than_both' if h<low-1e-8 else ('worse_than_both' if h>high+1e-8 else 'between_or_tied')
        positions[name]=dict(lower_is_better=True,values=values,position=pos)
    a=results['R0']['distance']['longest_quartile']['nrmse_percent']
    b=results['R1']['distance']['longest_quartile']['nrmse_percent']
    h=results['Rhalf']['distance']['longest_quartile']['nrmse_percent']
    return dict(metric_positions=positions,farthest_nrmse_gain_retained_fraction=(a-h)/(a-b) if abs(a-b)>1e-12 else None,
        interpretation='Descriptive tradeoffs without a composite objective; signed bias and exact choices must also be inspected')


def report(out,s):
    r=s['results'];half=r['Rhalf'];lines=['# Rhalf 部分源均衡：R0 / Rhalf / R1（正式50k）','',
        '本轮仅新增固定alpha=0.5；task损失为 `-0.6 mean((0.5+0.5 N/(M n(s))) d(s,G))`。'
        '这是记录均匀与去重源均匀task经验目标的混合，不是保持目标不变的重采样。'
        'B1、PHR、记录采样、dual时钟及全部共同配置保持一致。旧R0/R1直接复用，未重跑或追加seed/预算。','',
        '## 协议与验证','',
        'seed606，batch256，sampler seed2307，B1 seed42，task_mode=all，4线程，AdamW weight_decay0，'
        '逐记录PHR rho100；每访问用primal前残差以eta1更新dual。45k恒定lr2e-5，随后5k从2e-6几何衰减至2e-7。'
        '每1000步全量审计，保留epoch末尾40条小批次。所有组从相同零步model/optimizer/zero dual/visits/sampler/RNG/counters开始。','',
        f"N={s['weights']['N']}，M={s['weights']['M']}；Rhalf权重范围{s['weights']['min']:.6f}–{s['weights']['max']:.6f}，全量均值{s['weights']['mean']:.10f}。不做minibatch权重和归一化。",'',
        '6项训练pytest测试及2项分组分析测试全部通过：alpha端点相对旧源码快照更新一致，half全量值与梯度、B1/PHR/dual公式、'
        '全50k采样预算及真实4步训练/恢复和alpha/权重/配置拒绝。源码快照与环境指纹已冻结；'
        '运行环境同上轮，旧源码后续多seed支持未改变核心更新函数，本轮仍只用seed606。','',
        f"三组完整有序采样序列一致，每行访问{s['pairing']['visits_min']}–{s['pairing']['visits_max']}次，"
        f"总访问{s['pairing']['visits_sum']:,}次，Rhalf完成{s['pairing']['full_audits']}次全量审计（含0步）。"
        'R0/R1配置、checkpoint、数据和预算均核验；本次统一重算的R0/R1评价结果与旧报告完全相等。','',
        '## 任务距离','',
        'bias=预测−参考；NRMSE=RMSE/RMS(参考)。修正参考仅在50k完成后加载。','',
        '| 分组 | 模型 | MAE | RMSE | NRMSE % | bias |','|---|---|---:|---:|---:|---:|']
    h=half['distance'];d=half['decisions'];loc=half['local'];assessment=s.get('assessment',assess(r))
    retained=assessment['farthest_nrmse_gain_retained_fraction']
    lead=(f"本轮Rhalf的去重源NRMSE为{h['unique_sources']['nrmse_percent']:.4f}%，最远四分位为{h['longest_quartile']['nrmse_percent']:.4f}%，"
        f"相对R0保留R1远端NRMSE改善幅度的{100*retained:.2f}%。去重源bias从R1的{r['R1']['distance']['unique_sources']['bias']:+.4f}变为{h['unique_sources']['bias']:+.4f}，"
        f"去重边平均正违反从R1的{r['R1']['local']['unique_edges']['mean_positive']:.6f}变为{loc['unique_edges']['mean_positive']:.6f}。"
        f"但最优分支数为{d['optimal_branch_choices']}/58（R0/R1为{r['R0']['decisions']['optimal_branch_choices']}/{r['R1']['decisions']['optimal_branch_choices']}），"
        f"最大违反为{loc['unique_edges']['max_positive']:.6f}（R0/R1为{r['R0']['local']['unique_edges']['max_positive']:.6f}/{r['R1']['local']['unique_edges']['max_positive']:.6f}）。")
    positions=assessment['metric_positions'];best=any(v['position']=='better_than_both' for v in positions.values());worst=any(v['position']=='worse_than_both' for v in positions.values())
    if best and worst:
        judgement='它不是各项指标简单位于两端之间：部分指标优于两端，部分指标劣于两端。对于本轮结果，距离拟合、总体高估与平均违反形成了有利折中，但最坏违反和分支决策没有改善，不能宣布整体更优。'
    else:
        judgement='应按以下距离、偏差、决策和约束指标分别判断；没有预设综合效用，不能仅凭NRMSE下降宣布整体更优。'
    lines[2:2]=[lead,'',judgement,'']
    labels={'records':'42,536记录','unique_sources':'6,766去重源','longest_quartile':'最远四分位','unique_starts':'165独立起点','starts':'307原始起点'}
    for group,label in labels.items():
        for arm in ARMS:
            d=r[arm]['distance'][group];lines.append(f"| {label} | {arm} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['nrmse_percent']:.4f} | {d['bias']:+.4f} |")
    lines+=['','## 分支与完整路径','',
        '| 模型 | 最优分支/58 | 严格排序正确率 | 分支平均regret | 165路径平均regret | 165最优路径 | 307路径平均regret |','|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        d=r[arm]['decisions'];u=d['unique_start_rollouts']
        lines.append(f"| {arm} | {d['optimal_branch_choices']} | {d['pairwise_order_accuracy']:.4%} | {d['mean_branch_regret']:.4f} | {u['mean_successful_regret']:.4f} | {u['optimal_rollouts']} | {d['mean_successful_regret']:.4f} |")
    lines+=['','候选边和cost+预测后继距离贪心规则相同；严格排序排除参考并列，最优容差1e-6。'
        '图无环、唯一无出边节点为目标、最长可选路径275步；307/307到达主要由图结构保证，仅为完整性检查。','',
        '## 局部违反','',
        '残差=d(s,s′)−原始一步成本。违反数使用阈值1e-5；最大/平均正违反为relu残差，不截断容差。','',
        '| 模型 | 分布 | 违反数/总数 | 最大正违反 | 平均正违反 |','|---|---|---:|---:|---:|']
    for arm in ARMS:
        for group in ('records','unique_edges'):
            d=r[arm]['local'][group];lines.append(f"| {arm} | {group} | {d['violations']}/{d['count']} | {d['max_positive']:.6f} | {d['mean_positive']:.6f} |")
    lines+=['','## 长程有符号误差分解','',
        '`D*−J = Σ(cost−d_edge) + [Σd_edge−J]`，沿固定参考最短路径计算。两项分别是累计边项和累计三角松弛。'
        '负边项可掩盖正三角松弛；该恒等式不构成RMSE可加归因或因果中介识别。','',
        '| 分组 | 模型 | D*−J | 累计边项 | 累计三角松弛 |','|---|---|---:|---:|---:|']
    for group in labels:
        for arm in ARMS:
            d=r[arm]['decomposition']['groups'][group];lines.append(f"| {labels[group]} | {arm} | {d['gap']:.4f} | {d['edge_slack']:.4f} | {d['triangle_slack']:.4f} |")
    lines+=['','## 按源重复次数分组','',
        '边界在训练前固定，计数来自参与task push的物理行。下面先给每源一次的误差；记录加权结果另列。'
        '不同组参考距离分布不同，因此跨组NRMSE不能当成同尺度难度比较；组内三模型的参考尺度相同。','',
        '| n(s) | 去重源数 | 记录数 | 组内平均参考 | R0/Rhalf/R1 task目标权重占比 |','|---|---:|---:|---:|---|']
    for name,g in s['multiplicity'].items():
        masses='/'.join(f"{100*g['task_mass'][a]:.2f}%" for a in ARMS)
        lines.append(f"| {name} | {g['n_sources']} | {g['n_records']} | {g['reference_mean']:.4f} | {masses} |")
    for distribution,title in [('unique_sources','每源等权'),('records','记录加权')]:
        lines+=['',f'### {title}','', '| n(s) | 模型 | MAE | RMSE | NRMSE % | bias |','|---|---|---:|---:|---:|---:|']
        for name,g in s['multiplicity'].items():
            for arm in ARMS:
                d=g['metrics'][arm][distribution];lines.append(f"| {name} | {arm} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['nrmse_percent']:.4f} | {d['bias']:+.4f} |")
    lines+=['','### 逐源绝对误差的改善与退化','',
        '差值=Rhalf绝对误差−对照绝对误差，负为改善；计数容差1e-5。每个源只计一次。','',
        '| n(s) | 对照 | 改善源数 | 退化源数 | 近似不变 | 平均MAE差 |','|---|---|---:|---:|---:|---:|']
    for name,g in s['multiplicity'].items():
        for arm in ('R0','R1'):
            d=g['changes']['Rhalf_minus_'+arm]
            lines.append(f"| {name} | {arm} | {d['improved_sources']} | {d['worsened_sources']} | {d['unchanged_sources']} | {d['mean_absolute_error_delta']:+.4f} |")
    lines+=['','## Rhalf的折中判断','']
    assessment=s.get('assessment',assess(r));positions=assessment['metric_positions']
    metric_labels={'unique_source_nrmse':'去重源NRMSE','farthest_nrmse':'最远四分位NRMSE','start165_nrmse':'165起点NRMSE',
        'records_mae':'记录MAE','unique_source_abs_bias':'去重源绝对bias','farthest_abs_bias':'最远四分位绝对bias',
        'negative_optimal_branches':'最优分支数（越多越好）','negative_ranking_accuracy':'排序正确率（越高越好）',
        'branch_regret':'分支平均regret','path165_regret':'165起点平均路径regret',
        'negative_optimal_paths165':'165起点最优完整路径数（越多越好）','maximum_violation':'最大局部违反',
        'unique_edge_mean_violation':'去重边平均正违反','record_mean_violation':'记录平均正违反'}
    better=[metric_labels[k] for k,v in positions.items() if v['position']=='better_than_both']
    worse=[metric_labels[k] for k,v in positions.items() if v['position']=='worse_than_both']
    retained=assessment['farthest_nrmse_gain_retained_fraction']
    if retained is not None:lines += [f'相对R0，Rhalf保留R1最远四分位NRMSE改善幅度的 {100*retained:.2f}%。该比例仅描述这一指标，不是综合收益或因果归因。','']
    lines += ['优于两端的指标：'+(', '.join(better) if better else '无')+'。',
        '劣于两端的指标：'+(', '.join(worse) if worse else '无')+'。',
        '其余所列指标位于两端之间或并列；不能由alpha的线性插值假定训练结果也线性插值。','']
    for group in ('records','unique_sources','longest_quartile','unique_starts'):
        h=half['distance'][group];a=r['R0']['distance'][group];b=r['R1']['distance'][group]
        lines.append(f"- {labels[group]}：Rhalf NRMSE {h['nrmse_percent']:.4f}%，相对R0 {h['nrmse_percent']-a['nrmse_percent']:+.4f}个百分点、相对R1 {h['nrmse_percent']-b['nrmse_percent']:+.4f}个百分点；MAE {h['mae']:.4f}，bias {h['bias']:+.4f}。")
    for other in ('R0','R1'):
        a=r[other];d=half['decisions'];b=a['decisions'];h=half['local']['unique_edges'];l=a['local']['unique_edges']
        lines.append(f"- 相对{other}：最优分支数变化 {d['optimal_branch_choices']-b['optimal_branch_choices']:+d}，排序正确率变化 {(d['pairwise_order_accuracy']-b['pairwise_order_accuracy'])*100:+.4f}个百分点，分支平均regret变化 {d['mean_branch_regret']-b['mean_branch_regret']:+.4f}，165路径平均regret变化 {d['unique_start_rollouts']['mean_successful_regret']-b['unique_start_rollouts']['mean_successful_regret']:+.4f}；最大违反变化 {h['max_positive']-l['max_positive']:+.6f}，去重边平均正违反变化 {h['mean_positive']-l['mean_positive']:+.6f}。")
        h=half['decomposition']['groups']['longest_quartile'];d=a['decomposition']['groups']['longest_quartile']
        lines.append(f"- 最远四分位相对{other}的有符号gap变化 {h['gap']-d['gap']:+.4f} = 累计边项变化 {h['edge_slack']-d['edge_slack']:+.4f} + 三角松弛变化 {h['triangle_slack']-d['triangle_slack']:+.4f}。")
    lines+=['','### 改善与退化发生在哪里','']
    totals={a:dict(improved=0,worsened=0) for a in ('R0','R1')}
    for name,g in s['multiplicity'].items():
        a=g['changes']['Rhalf_minus_R0'];b=g['changes']['Rhalf_minus_R1']
        for arm,c in [('R0',a),('R1',b)]:
            totals[arm]['improved']+=c['improved_sources'];totals[arm]['worsened']+=c['worsened_sources']
        lines.append(f"- n(s)={name}，{g['n_sources']}个源：相对R0，组内MAE变化{a['mean_absolute_error_delta']:+.4f}，{a['improved_sources']}个源改善、{a['worsened_sources']}个源退化；相对R1，组内MAE变化{b['mean_absolute_error_delta']:+.4f}。")
    lines+=['',f"合计相对R0：{totals['R0']['improved']}个源绝对误差改善，{totals['R0']['worsened']}个源退化；相对R1：{totals['R1']['improved']}改善、{totals['R1']['worsened']}退化。"
        '平均误差下降不意味着大多数或每个状态都改善；尤其组内平均改善与退化源数量可能同时增加。各组规模差异很大，应同时看样本数与误差幅度。','',
        '相对R0的远端误差改善伴随累计三角松弛下降，且该组累计边项变得更不负；因此不能把该组改善简单解释为累积了更多局部违反。'
        '但局部最大违反和全体平均违反是不同统计量，仍须单独报告。']
    lines+=['','## 45k仅作事后过程分析','',
        '| 模型/步骤 | 去重源NRMSE % | 最远NRMSE % | 最优分支 | 165路径regret | 最大违反 |','|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for step,x in [(45000,s['stage45000'][arm]),(50000,r[arm])]:
            lines.append(f"| {arm}/{step} | {x['distance']['unique_sources']['nrmse_percent']:.4f} | {x['distance']['longest_quartile']['nrmse_percent']:.4f} | {x['decisions']['optimal_branch_choices']} | {x['decisions']['unique_start_rollouts']['mean_successful_regret']:.4f} | {x['local']['unique_edges']['max_positive']:.6f} |")
    lines+=['','50k始终为正式终点，不挑最好checkpoint。单seed606、4线程、固定有限训练图；'
        '不承诺多线程逐位复现，不证明多seed稳定性、连续控制或未见状态泛化。共享后缀下源/记录/路径也不是统计独立样本。'
        '本轮未搜索其他alpha、追加预算或新增seed。','',
        '命令和实现见 `experiments/task_partial_balance/PROTOCOL.md`；冻结配置 `../protocol.json`，'
        '复用核验 `../reuse_verification.json`，测试 `../../tests.log`，完整结果 `summary.json`，分组结果 `multiplicity.json`。',
        '', '![统一比较](comparison.png)','']
    (out/'report.md').write_text('\n'.join(lines))


def plot(out,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=s['results'];x=np.arange(3);fig,axes=plt.subplots(2,3,figsize=(15,8))
    for j,g in enumerate(('unique_sources','longest_quartile','unique_starts')):
        axes[0,0].bar(x+(j-1)*.25,[r[a]['distance'][g]['nrmse_percent'] for a in ARMS],width=.25,label=g)
    axes[0,0].set(xticks=x,xticklabels=ARMS,ylabel='NRMSE (%)');axes[0,0].legend(fontsize=8)
    axes[0,1].bar(x,[r[a]['decisions']['unique_start_rollouts']['mean_successful_regret'] for a in ARMS]);axes[0,1].set(xticks=x,xticklabels=ARMS,ylabel='165-start mean path regret')
    axes[0,2].bar(x,[r[a]['local']['unique_edges']['max_positive'] for a in ARMS]);axes[0,2].set(xticks=x,xticklabels=ARMS,ylabel='Maximum local violation')
    for j,k in enumerate(('edge_slack','triangle_slack','gap')):
        axes[1,0].bar(x+(j-1)*.25,[r[a]['decomposition']['groups']['longest_quartile'][k] for a in ARMS],width=.25,label=k)
    axes[1,0].axhline(0,color='k',lw=.6);axes[1,0].set(xticks=x,xticklabels=ARMS,ylabel='Farthest-quartile signed decomposition');axes[1,0].legend(fontsize=8)
    groups=list(s['multiplicity']);xx=np.arange(len(groups))
    for a in ARMS:
        axes[1,1].plot(xx,[s['multiplicity'][g]['metrics'][a]['unique_sources']['mae'] for g in groups],'-o',label=a)
        axes[1,2].plot(xx,[s['multiplicity'][g]['metrics'][a]['unique_sources']['bias'] for g in groups],'-o',label=a)
    axes[1,1].set(xticks=xx,xticklabels=groups,xlabel='Source multiplicity',ylabel='Unique-source MAE');axes[1,1].legend()
    axes[1,2].axhline(0,color='k',lw=.6);axes[1,2].set(xticks=xx,xticklabels=groups,xlabel='Source multiplicity',ylabel='Unique-source bias');axes[1,2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
