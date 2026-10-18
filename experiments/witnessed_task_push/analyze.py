"""Final-only paired diagnosis. Recorded graph quantities never enter training."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from experiments.four_path_execution.common import read,write,adapter,file_digest
from experiments.four_path_factorial.run import evaluate,tensor_digest
from experiments.full_replay_four_path.data import load_union
from experiments.full_replay_four_path.train import distances
from experiments.full_replay_four_path.analyze_values import stats


def lines(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def residual_stats(h):
    pos=np.maximum(h,0)
    return dict(count=len(h),violations=int((h>1e-5).sum()),max_violation=float(pos.max()),
                mean_positive=float(pos.mean()),p99_positive=float(np.quantile(pos,.99)))


def distribution(values):
    return dict(count=len(values),mean=float(values.mean()) if len(values) else None,
                quantiles=np.quantile(values,[0,.25,.5,.75,.99,1]).tolist() if len(values) else [])


@torch.no_grad()
def run(root):
    assert (root/'train/COMPLETE').exists(), 'Evaluate only the fixed-budget final model'
    torch.set_num_threads(4)
    protocol=read(root/'protocol.json');baseline=Path(protocol['baseline'])
    for p,h in protocol['source_hashes'].items():assert file_digest(p)==h,p
    for p,h in read(root/'dependency_audit.json')['source_hashes'].items():assert file_digest(p)==h,p
    for p,h in protocol['baseline_hashes'].items():assert file_digest(baseline/p)==h,p
    config=read(root/'train/config.json')
    for k,v in protocol['unchanged'].items():assert config[k]==v,k
    d=load_union(task_mode='witnessed');assert config['fingerprints']==d.fingerprint
    assert config['task_mask_sha256']==tensor_digest(d.raw.transition_infos['global_push_task_mask'])
    assert config['task_mode']=='witnessed'
    out=root/'comparison';out.mkdir(exist_ok=False)
    saved=torch.load(root/'train/final.pt',map_location='cpu',weights_only=True)
    old_saved=torch.load(baseline/'train/final.pt',map_location='cpu',weights_only=True)
    assert saved['step']==old_saved['step']==25000
    initial=torch.load(root/'train/initial.pt',map_location='cpu',weights_only=True)
    old_initial=torch.load(baseline/'train/initial.pt',map_location='cpu',weights_only=True)
    for k,v in initial['model'].items():assert torch.equal(v,old_initial['model'][k]),k
    assert torch.equal(saved['visits'],old_saved['visits'])
    for k,v in saved['sampler'].items():
        other=old_saved['sampler'][k]
        assert torch.equal(v,other) if torch.is_tensor(v) else v==other,k
    counters=saved['push_counters'];visits=saved['visits']
    witnessed=d.raw.transition_infos['global_push_task_witness_mask']
    assert counters['original_pair_visits']==int(visits.sum())
    assert counters['original_task_visits']==int(visits[witnessed[d.original]].sum())
    assert counters['path_pair_visits']==counters['path_task_visits']==335*25000
    verification=dict(initial_model_bitwise_equal=True,sampler_final_bitwise_equal=True,
        original_visits_bitwise_equal=True,source_hashes_verified=True,baseline_hashes_verified=True,
        task_mask_sha256=config['task_mask_sha256'],data_fingerprints=d.fingerprint,
        final_checkpoint_sha256=file_digest(root/'train/final.pt'),
        visits=dict(min=int(visits.min()),max=int(visits.max()),mean=float(visits.float().mean())),
        push_counters=counters,empty_task_batch_fraction=counters['original_empty_task_batches']/25000)
    write(out/'verification.json',verification)
    adapted,model=adapter(d.small,root/'train/final.pt')
    rec,values,arrays=evaluate(model,d.small,None,25000,0.)
    write(out/'four_path.json',rec);write(out/'four_path_values.json',values)
    np.savez_compressed(out/'four_path_arrays.npz',**arrays)
    rows=d.physical.nonzero().flatten().numpy()
    previous=np.load(baseline/'value_analysis/recorded_values.npz')
    np.testing.assert_array_equal(rows,previous['rows'])
    pred=[np.concatenate(distances(model,d,r)[1:]).reshape(2,-1) for r in torch.from_numpy(rows).split(1024)]
    task,pair=np.concatenate(pred,axis=1)
    assert np.isfinite(task).all() and np.isfinite(pair).all()
    # Reuse exact recorded graph and segment upper bounds after verifying unchanged inputs.
    # They are diagnostics, never targets or checkpoint selection criteria.
    ref=previous['graph_task'];upper=previous['b1_chain_upper']
    np.savez_compressed(out/'recorded_values.npz',rows=rows,task=task,pair=pair,graph_task=ref,b1_chain_upper=upper)
    physical=d.physical.numpy();origin=rows<d.nbase
    dev=d.raw.transition_infos['device_index'].numpy();witness=witnessed.numpy()[rows]
    observations=d.raw.all_observations[d.src_obs[rows]].numpy()
    u=(dev[rows]==0)&(observations[:,0]>2.8)&(observations[:,0]<5.2)&(observations[:,1]>1.9)&(observations[:,1]<5.4)
    groups={'original':origin,'paths':~origin,'original_U':origin&u,
            'original_witnessed':origin&witness,'original_unwitnessed':origin&~witness}
    groups.update({f'device_{i}':origin&(dev[rows]==i) for i in range(4)})
    values_report={}
    for name,mask in groups.items():
        item=dict(count=int(mask.sum()),witness_count=int((mask&witness).sum()),
                  finite_graph_count=int(np.isfinite(ref[mask]).sum()),arms={})
        for arm,t,p in [('all',previous['task'],previous['pair']),('witnessed',task,pair)]:
            item['arms'][arm]=dict(task=distribution(t[mask]),pair=distribution(p[mask]),
                graph_comparison=stats(t[mask],ref[mask]),
                b1_upper=dict(mean=float(upper[mask].mean()),
                    above_count=int((p[mask]>upper[mask]+1e-5).sum()),
                    max_excess=float(np.maximum(p[mask]-upper[mask],0).max())))
        values_report[name]=item
    write(out/'values.json',values_report)
    residuals={arm:np.load(p/'train/final_constraints.npz')['residual'] for arm,p in [('all',baseline),('witnessed',root)]}
    for h in residuals.values():assert np.isfinite(h).all() and np.max(np.abs(h[~physical]))==0
    fullrows=np.arange(len(physical));entry=physical&(d.dst.numpy()==d.goal.numpy())&(d.src.numpy()!=d.dst.numpy())
    masks=dict(original=physical&(fullrows<d.nbase),paths=physical&(fullrows>=d.nbase),
               original_terminal_entry=entry&(fullrows<d.nbase),path_terminal_entry=entry&(fullrows>=d.nbase))
    masks.update({f'device_{i}':physical&(fullrows<d.nbase)&(dev==i) for i in range(4)})
    constraints={name:{arm:residual_stats(h[mask]) for arm,h in residuals.items()} for name,mask in masks.items()}
    write(out/'constraints.json',constraints)
    worst=[]
    for row in np.argsort(-residuals['witnessed'])[:50]:
        worst.append(dict(row=int(row),device=int(dev[row]),source='original' if row<d.nbase else 'paths',
            residual=float(residuals['witnessed'][row]),cost=float(d.cost[row]),witness=bool(witnessed[row]),
            terminal_entry=bool(entry[row]),source_observation=d.nodes[d.src[row]].tolist(),
            target_observation=d.nodes[d.dst[row]].tolist()))
    write(out/'worst_edges.json',worst)
    # All final residuals can be reconstructed from the independent chunked prediction.
    check_rows=torch.from_numpy(rows[::97])
    direct=(distances(model,d,check_rows,False)-d.cost[check_rows]).numpy()
    np.testing.assert_allclose(direct,residuals['witnessed'][check_rows.numpy()],rtol=5e-4,atol=2e-4)
    select=np.random.default_rng(1704).choice(len(rows),2048,replace=False);r=rows[select]
    exported=adapted.batch_value(d.nodes[d.src[r]].numpy(),d.nodes[d.goal[r]].numpy())
    np.testing.assert_allclose(exported,task[select],rtol=5e-6,atol=1e-4)
    write(out/'adapter_check.json',dict(count=len(r),max_error=float(abs(exported-task[select]).max())))
    probes={k:torch.tensor(v) for k,v in read(root/'train/diagnostic_probes.json').items()}
    old_probes=[]
    for stage,step in [('initial',0),('phr',20000),('final',25000)]:
        _,oldmodel=adapter(d.small,baseline/f'train/{stage}.pt')
        bygroup={}
        for name,r in probes.items():
            _,t,p=distances(oldmodel,d,r)
            bygroup[name]=dict(task_mean=float(t.mean()),task_max=float(t.max()),pair_mean=float(p.mean()))
        old_probes.append(dict(step=step,model_only_probes=bygroup))
    write(out/'baseline_probes.json',old_probes)
    oldrec=read(baseline/'value_analysis/four_path.json')
    scans=[r for r in lines(root/'train/training.jsonl') if 'model_only_probes' in r]
    corridor=[dict(step=r['step'],**r['model_only_probes']['device_1']) for r in scans]
    write(out/'corridor_probe_trajectory.json',corridor)
    decay_start=next(r for r in corridor if r['step']==20000)
    corridor_change=dict(initial=corridor[0]['task_mean'],phr=decay_start['task_mean'],
        final=corridor[-1]['task_mean'],maximum=max(r['task_mean'] for r in corridor),
        decay_relative_change_percent=100*(corridor[-1]['task_mean']/decay_start['task_mean']-1))
    summary=dict(verification=verification,values=values_report,constraints=constraints,
                 four_path=dict(all=oldrec,witnessed=rec),corridor_probes=corridor,corridor_change=corridor_change)
    write(out/'summary.json',summary)
    plot(out,baseline,scans,old_probes,oldrec,rec,values)
    report(out,summary,baseline,root)
    frozen=out/'source_snapshot';frozen.mkdir()
    hashes={}
    for p in Path('experiments/witnessed_task_push').glob('*.py'):
        shutil.copy2(p,frozen/p.name);hashes[str(p)]=file_digest(p)
    write(out/'analysis_manifest.json',dict(sources=hashes,checkpoint_sha256=verification['final_checkpoint_sha256'],
        references='Existing graph_task and b1_chain_upper reused only after training COMPLETE; source fingerprints checked.'))
    write(out/'COMPLETE',dict(final_step=25000,diagnostics_complete=True,navigation_evaluated=False))


def plot(out,baseline,scans,oldprobes,oldrec,rec,values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    ax.plot([r['step'] for r in scans],[r['model_only_probes']['device_1']['task_mean'] for r in scans],label='witnessed')
    ax.plot([r['step'] for r in oldprobes],[r['model_only_probes']['device_1']['task_mean'] for r in oldprobes],'o--',label='all (3 saved stages)')
    ax.set(title='Corridor: same 512 source probes',xlabel='Update',ylabel='Mean task distance',yscale='log');ax.legend()
    oldscans=[r for r in lines(baseline/'train/training.jsonl') if 'original' in r]
    for arm,records in [('all',oldscans),('witnessed',scans)]:
        axes[0,1].plot([r['step'] for r in records],[r['original']['violation_count'] for r in records],label=arm)
    axes[0,1].set(title='All 200,000 original physical edges',xlabel='Update',ylabel='Violations > 1e-5');axes[0,1].legend()
    for arm,records in [('all',oldscans),('witnessed',scans)]:
        axes[1,0].plot([r['step'] for r in records],[r['paths']['max_violation'] for r in records],label=arm)
    axes[1,0].set(title='Four-path local constraints',xlabel='Update',ylabel='Maximum positive residual')
    axes[1,0].set_yscale('symlog',linthresh=1e-4);axes[1,0].legend()
    x=np.arange(4);ax=axes[1,1]
    ax.bar(x-.25,[r['start'] for r in oldrec['paths']],width=.25,label='all')
    ax.bar(x,[r['start'] for r in rec['paths']],width=.25,label='witnessed')
    ax.bar(x+.25,[r['reference'] for r in rec['paths']],width=.25,label='recorded graph')
    ax.set(title='Four-path starts: final checkpoint',xticks=x,xticklabels=['P1','P2','P3','P4'],ylabel='Task distance');ax.legend()
    fig.savefig(out/'diagnostics.png',dpi=170);plt.close(fig)
    oldvalues=read(baseline/'value_analysis/four_path_values.json')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for i,ax in enumerate(axes.flat):
        new=[v for v in values if v['path']==i];old=[v for v in oldvalues if v['path']==i]
        ax.plot([v['timestep'] for v in new],[v['reference'] for v in new],color='black',label='recorded graph')
        ax.plot([v['timestep'] for v in old],[v['value'] for v in old],label='all')
        ax.plot([v['timestep'] for v in new],[v['value'] for v in new],label='witnessed')
        ax.set(title=f'Path {i+1}',xlabel='Recorded timestep',ylabel='Distance to task goal');ax.legend()
    fig.savefig(out/'path_values.png',dpi=170);plt.close(fig)


def report(out,s,baseline,root):
    old,new=s['four_path']['all'],s['four_path']['witnessed'];v=s['values'];c=s['constraints'];verify=s['verification']
    corridor_old=v['device_1']['arms']['all']['task']['mean'];corridor_new=v['device_1']['arms']['witnessed']['task']['mean']
    text=['# Witnessed task push：单 seed 配对诊断','',
          '保持完整 20 万转移＋四路径、共享 encoder/IQE、采样、局部 PHR、B1 和学习率预算不变，仅 task push 选择真实合法成功后缀的源。seed 606 从与旧 all 实验逐位相同的初始参数重新训练，固定完成 20,000＋5,000 步；没有按参考值选择模型。','',
          '三个问题的判断：','',
          f'1. **此前的大幅数值膨胀受到抑制，但不能说完全停止增长。** corridor 最终全量均值下降 {100*(1-corridor_new/corridor_old):.2f}%，探针后半段呈平台附近波动，降步长阶段变化 {s["corridor_change"]["decay_relative_change_percent"]:+.2f}%。没有记录成功链，因此不能把最终较小预测解释为正确任务距离。',
          f'2. **四路径距离量级明显回升，但长程起点仍不完整。** NRMSE 从 {old["task_nrmse_percent"]:.2f}% 降到 {new["task_nrmse_percent"]:.2f}%；P1 接近参考，P3/P4 起点仍大幅低估。整体平均误差改善不能掩盖这些前段失败。',
          '3. **约束违反幅度明显缓解，尚未恢复全量可行性。** 原始边的违反数量、最大值和平均正残差均下降；四路径最大值和平均正残差下降，但违反边数增加，入终态边的部分指标也恶化。','',
          '| 指标 | 原 all task | witnessed task |','|---|---:|---:|',
          f'| corridor 全 50,000 源的 task 均值 | {corridor_old:.4f} | {corridor_new:.4f} |',
          f'| 四路径 task NRMSE | {old["task_nrmse_percent"]:.3f}% | {new["task_nrmse_percent"]:.3f}% |',
          f'| 四路径 push gap | {old["push_gap_percent"]:.3f}% | {new["push_gap_percent"]:.3f}% |']
    for name,label in [('original','原始边'),('paths','四路径边')]:
        a,b=c[name]['all'],c[name]['witnessed']
        text += [f'| {label}违反数（>1e-5） | {a["violations"]}/{a["count"]} | {b["violations"]}/{b["count"]} |',
                 f'| {label}最大正残差 | {a["max_violation"]:.6g} | {b["max_violation"]:.6g} |',
                 f'| {label}平均正残差 | {a["mean_positive"]:.6g} | {b["mean_positive"]:.6g} |']
    drift=s['corridor_change']
    text += ['',f'corridor 最终均值为旧 all 的 {100*corridor_new/corridor_old:.3f}%。同一组 512 探针均值：初始化 {drift["initial"]:.3f} → PHR 结束 {drift["phr"]:.3f} → 最终 {drift["final"]:.3f}；降步长阶段变化 {drift["decay_relative_change_percent"]:+.3f}%。其记录图到任务终态仍无有限路径，较小数值本身不是正确估值的证据。共享模型受其他 task 与 B1 更新影响，即使没有直接 task push，corridor 值也可以变化。旧 all 实验只有 initial/PHR/final 三个可恢复阶段，图中虚线仅连接这些阶段。','',
        '| 路径起点 | all | witnessed | 记录图参考 | edge slack | triangle slack |',
        '|---|---:|---:|---:|---:|---:|']
    for a,b in zip(old['paths'],new['paths']):
        text.append(f'| P{b["path"]+1} | {a["start"]:.3f} | {b["start"]:.3f} | {b["reference"]:.3f} | {b["edge_slack"]:.3f} | {b["triangle_slack"]:.3f} |')
    p3,p4=new['paths'][2:]
    text += ['', 'edge slack 是路径一步成本与预测边长之差的累加；triangle slack 是预测边长与相邻 task 值间的三角不等式余量。两者之和为起点低估。局部可行性、记录图长程恢复和未见状态导航是不同指标；不以约束数或 push gap 单独判断成功。',
        f'P3/P4 起点分别低估 {100*(1-p3["start"]/p3["reference"]):.1f}%/{100*(1-p4["start"]/p4["reference"]):.1f}%；其中三角余量占起点低估的 {100*p3["triangle_slack"]/(p3["reference"]-p3["start"]):.1f}%/{100*p4["triangle_slack"]/(p4["reference"]-p4["start"]):.1f}%。路径曲线前段沿真实成功路径前进时预测 task 距离反而升高，后段才贴近参考。这是剩余长程关系未恢复的直接证据；不能由后段拟合较好推断候选排序或闭环成功。','',
        '## 各任务与片段支持','',
        '| 原始任务 | task 见证源 | task 均值 all → witnessed | B1 均值 all → witnessed | 超出 B1 记录片段上界 all → witnessed |',
        '|---|---:|---:|---:|---:|']
    for i in range(4):
        g=v[f'device_{i}'];a,b=g['arms']['all'],g['arms']['witnessed']
        text.append(f'| device {i} | {g["witness_count"]}/{g["count"]} | {a["task"]["mean"]:.2f} → {b["task"]["mean"]:.2f} | {a["pair"]["mean"]:.2f} → {b["pair"]["mean"]:.2f} | {a["b1_upper"]["above_count"]} → {b["b1_upper"]["above_count"]} |')
    a,b=c['original_terminal_entry']['all'],c['original_terminal_entry']['witnessed']
    text += ['',f'原始真实入终态边共 {a["count"]} 条：违反数 {a["violations"]} → {b["violations"]}，最大正残差 {a["max_violation"]:.4f} → {b["max_violation"]:.4f}。这些边与恒等 anchor 分开统计，不能用 anchor 为零掩盖入终态边违反。',
        f'四路径入终态边的违反数 {c["path_terminal_entry"]["all"]["violations"]} → {c["path_terminal_entry"]["witnessed"]["violations"]}，最大正残差 {c["path_terminal_entry"]["all"]["max_violation"]:.6f} → {c["path_terminal_entry"]["witnessed"]["max_violation"]:.6f}。局部改善并非每一个子组都成立。',
        'B1 的有记录片段成本只是该对状态的距离上界，不自动等于最短距离。上表是所有记录源的预测均值，并非 witnessed 训练目标；没有见证的 task 预测仍可在评估中读取，但不会被 task loss 最大化。','',
        '## 掩码、数据与更新保留','',
        '原始 task 保留 2,575/200,000 个见证源，四路径保留 335/335；每步两个来源各占 0.5，内部 task 0.6、B1 0.4。task 对当前来源 batch 中保留条目求均值，空选择返回可微零；不回退到全 task，也不把空项权重交给 B1。条件均值会把原 task 权重集中到见证源，这与生产 witnessed 语义一致，并非用 200,000 作固定分母。',
        f'原始空 task batch 共 {verify["push_counters"]["original_empty_task_batches"]}/25,000（{100*verify["empty_task_batch_fraction"]:.2f}%）。所有原始物理边访问 {verify["visits"]["min"]}–{verify["visits"]["max"]} 次；原始 B1 总访问 {verify["push_counters"]["original_pair_visits"]:,}，四路径 B1 总访问 {verify["push_counters"]["path_pair_visits"]:,}。采样器最终状态和逐边访问数与旧实验逐位一致。失败片段未从约束或 B1 删除。','',
        '独立核验确认完整成本、动力学、终态 anchor 和 B1 episode 边界正确；模型初始参数、最终采样器状态及逐边访问数与旧实验逐位一致，合并数据、成本及配对指纹一致，旧结果哈希未变。另抽取2,048个配对检查推理适配，最大输出差为0。所有 201,087 边每 1,000 步扫描；752 个 quotient 恒等 anchor 保持零，原始物理边的乘子仍只按访问更新，四路径每步更新，没有加入 QP。','',
        '剩余瓶颈至少有两项：原始边每条只访问 31–32 次，特别是 14 条入终态边仍有大残差；四路径前段则主要缺少长程距离的紧致性，不能只靠进一步压低局部违反来保证恢复。当前数据支持限制消除了一个明显的无见证最大化方向，但没有解决这两项问题。本实验未改变采样或优化器来进一步验证其因果作用。','',
        '## 解释范围与复现','',
        '这是单个配对 seed 的训练目标诊断，不是跨 seed 稳健性结论。本轮未新增闭环 MPPI 或固定候选评估，因此不能据此宣称未见状态或导航成功率改善；此前 all 模型的闭环失败结果保留。所有残差、最坏 50 边、各任务预测分布及 B1 上界违反均输出，没有筛掉失败。图参考及 B1 链成本仅用于固定最终 checkpoint 的事后评估；无 Oracle 标签，无新采集。','',
        '数值限制：生产实现的 4 CPU 线程更新存在微小非确定性。相同初始状态及 batch 的两份模型，5 步后最大参数差约 2.16e-5；单线程同样检查逐位相同。端到端恢复 smoke 的访问计数一致，但浮点模型轨迹不保证逐位相同。本轮正式训练未中断恢复，并维持旧 4 线程设置；配对初始化和采样消除了这两项差异，但未消除多线程数值噪声。','',
        '复现入口：[实验说明与运行命令](../../../../experiments/witnessed_task_push/README.md)。checkpoint 含模型、AdamW、所有逐边乘子、访问计数、采样器及 RNG，并校验 task mode/mask，防止跨目标误恢复。','',
        '![诊断](diagnostics.png)','', '![路径值](path_values.png)','']
    (out/'report.md').write_text('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
