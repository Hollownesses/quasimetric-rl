"""Final-only A/B constraint and long-range value comparison; no controller runs."""
import argparse
import shutil
from pathlib import Path
import numpy as np
import torch
from experiments.four_path_execution.common import read,write,adapter,file_digest
from experiments.four_path_factorial.run import evaluate,graph_reference
from experiments.full_replay_four_path.data import load_union
from experiments.full_replay_four_path.train import distances,EpochSampler
from experiments.witnessed_task_push.analyze import lines,residual_stats,distribution
from experiments.large_phr_calibration.train import SOURCE,EXPECTED_SOURCE


def assert_nested(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested(x,y)
    else:assert a==b


def audits(path):
    # Preserve the raw log but remove replayed audit steps for plotting/statistics.
    records={r['continuation_step']:r for r in lines(path) if 'original' in r}
    return [records[k] for k in sorted(records)]


@torch.no_grad()
def run(root):
    for arm in ('A','B'):assert (root/arm/'COMPLETE').exists(),arm
    torch.set_num_threads(4);protocol=read(root/'protocol.json');parent=Path(protocol['parent'])
    assert file_digest(SOURCE)==EXPECTED_SOURCE
    for p,h in protocol['source_hashes'].items():assert file_digest(p)==h,p
    for p,h in protocol['parent_sha256'].items():assert file_digest(parent/p)==h,p
    d=load_union(task_mode='witnessed');assert d.fingerprint==protocol['data_fingerprints']
    out=root/'comparison';out.mkdir(exist_ok=False)
    source=torch.load(SOURCE,map_location='cpu',weights_only=True)
    states={};records={};paths={};path_values={};value_groups={};residuals={};dual_stats={}
    learner_keys=('model','optimizer','dual','visits','sampler','rng','push_counters')
    for arm in ('A','B'):
        initial=torch.load(root/arm/'initial.pt',map_location='cpu',weights_only=True)
        for key in learner_keys:assert_nested(initial[key],source[key])
        state=torch.load(root/arm/'final.pt',map_location='cpu',weights_only=True)
        assert state['step']==30000 and state['continuation_step']==5000
        assert state['task_mode']=='witnessed' and state['task_mask_sha256']==source['task_mask_sha256']
        assert state['calibration_config']['source_sha256']==EXPECTED_SOURCE
        assert state['calibration_config']==read(root/arm/'config.json')
        for param,item in source['optimizer']['state'].items():
            if 'step' in item:assert state['optimizer']['state'][param]['step']==item['step']+5000
        states[arm]=state;records[arm]=audits(root/arm/'training.jsonl')
        assert [r['continuation_step'] for r in records[arm]]==list(range(0,5001,250))
    for key in ('visits','sampler','rng','push_counters'):assert_nested(states['A'][key],states['B'][key])
    config_A=read(root/'A/config.json');config_B=read(root/'B/config.json')
    assert {k:v for k,v in config_A.items() if k!='arm'}=={k:v for k,v in config_B.items() if k!='arm'}
    sampler=EpochSampler(len(d.original),0);sampler.restore(source['sampler'])
    extra=torch.zeros_like(source['visits'])
    for _ in range(5000):extra[sampler.next(256)]+=1
    assert_nested(sampler.state_dict(),states['A']['sampler'])
    assert torch.equal(source['visits']+extra,states['A']['visits'])
    assert states['A']['outer_updates']==0 and states['B']['outer_updates']==20
    events=[r for r in records['B'] if 'outer_update' in r]
    assert len(events)==20 and all(r['outer_update']['rows']==200000 and r['outer_update']['eta']==100 for r in events)
    assert not any('outer_update' in r for r in records['A'])
    verification=dict(parent_hashes_unchanged=True,source_hashes_verified=True,
        complete_initial_learner_state_bitwise_equal_to_parent=True,final_sampler_rng_visits_push_counts_paired=True,
        optimizer_steps_increased_by_5000=True,source_checkpoint_sha256=EXPECTED_SOURCE,
        final_checkpoint_sha256={a:file_digest(root/a/'final.pt') for a in states},
        extra_visits=dict(min=int(extra.min()),max=int(extra.max()),mean=float(extra.float().mean())),
        extra_push_counters={k:states['A']['push_counters'][k]-source['push_counters'][k] for k in source['push_counters']},
        original_dual_updates=dict(A='once per sampled visit',B=20,B_influencing_subsequent_primal_steps=19))
    # First full outer update has a known previous dual: exactly the parent dual.
    # Independently recompute all latest residuals at its saved primal model.
    first=torch.load(root/'B/at_0250.pt',map_location='cpu',weights_only=True)
    _,firstmodel=adapter(d.small,root/'B/at_0250.pt')
    first_h=torch.cat([distances(firstmodel,d,r,False)-d.cost[r] for r in torch.arange(len(d.src)).split(1024)])
    expected=(source['dual'][d.original]+100*first_h[d.original]).relu()
    torch.testing.assert_close(first['dual'][d.original],expected,rtol=5e-5,atol=.003)
    verification['first_outer_full_formula_max_error']=float((first['dual'][d.original]-expected).abs().max())
    write(out/'verification.json',verification)
    rows=d.physical.nonzero().flatten().numpy();physical=d.physical.numpy();fullrows=np.arange(len(physical))
    origin=rows<d.nbase;device=d.raw.transition_infos['device_index'].numpy()
    entry=physical&(d.dst.numpy()==d.goal.numpy())&(d.src.numpy()!=d.dst.numpy())
    masks=dict(original=physical&(fullrows<d.nbase),paths=physical&(fullrows>=d.nbase),
        original_terminal_entry=entry&(fullrows<d.nbase),path_terminal_entry=entry&(fullrows>=d.nbase))
    masks.update({f'device_{i}':physical&(fullrows<d.nbase)&(device==i) for i in range(4)})
    witness=d.raw.transition_infos['global_push_task_witness_mask'].numpy()
    masks['original_witnessed']=physical&(fullrows<d.nbase)&witness
    masks['original_unwitnessed']=physical&(fullrows<d.nbase)&~witness
    old=np.load(parent/'comparison/recorded_values.npz');np.testing.assert_array_equal(rows,old['rows'])
    merged_path_ref=old['graph_task'][rows>=d.nbase]
    small_path_ref=graph_reference(d.small)[d.small.task_src].numpy()
    np.testing.assert_allclose(merged_path_ref,small_path_ref,atol=1e-8,rtol=1e-8)
    write(out/'reference_consistency.json',dict(path_physical_sources=len(merged_path_ref),
        merged_and_four_path_graph_reference_max_error=float(abs(merged_path_ref-small_path_ref).max())))
    for arm in ('parent','A','B'):
        folder=parent/'train' if arm=='parent' else root/arm
        state=source if arm=='parent' else states[arm]
        raw=np.load(folder/'final_constraints.npz');h=raw['residual'];dual=raw['dual']
        assert np.isfinite(h).all() and np.isfinite(dual).all() and abs(h[~physical]).max()==0
        np.testing.assert_array_equal(dual,state['dual'].numpy());residuals[arm]=h
        dual_stats[arm]={}
        for name,mask in masks.items():
            dual_stats[arm][name]=dict(mean=float(dual[mask].mean()),max=float(dual[mask].max()),
                complementarity_mean=float(np.abs(dual[mask]*h[mask]).mean()),
                effective_mean=float(np.maximum(dual[mask]+100*h[mask],0).mean()))
        if arm=='parent':
            paths[arm]=read(parent/'comparison/four_path.json');path_values[arm]=read(parent/'comparison/four_path_values.json')
            task,pair=old['task'],old['pair']
        else:
            target=out/arm;target.mkdir()
            adapted,model=adapter(d.small,folder/'final.pt')
            rec,values,arrays=evaluate(model,d.small,None,30000,0.)
            paths[arm]=rec;path_values[arm]=values
            write(target/'four_path.json',rec);write(target/'four_path_values.json',values)
            np.savez_compressed(target/'four_path_arrays.npz',**arrays)
            blocks=[torch.stack(distances(model,d,r)).numpy() for r in torch.from_numpy(rows).split(1024)]
            edge,task,pair=np.concatenate(blocks,axis=1)
            np.testing.assert_allclose(edge-d.cost[rows].numpy(),h[rows],rtol=5e-4,atol=2e-4)
            assert np.isfinite(task).all() and np.isfinite(pair).all()
            np.savez_compressed(target/'recorded_values.npz',rows=rows,task=task,pair=pair,
                                graph_task=old['graph_task'],b1_chain_upper=old['b1_chain_upper'])
            select=rows[::97];indices=np.arange(len(rows))[::97]
            actual=adapted.batch_value(d.nodes[d.src[select]].numpy(),d.nodes[d.goal[select]].numpy())
            np.testing.assert_allclose(actual,task[indices],rtol=5e-6,atol=1e-4)
            write(target/'adapter_check.json',dict(rows=len(select),max_error=float(abs(actual-task[indices]).max())))
            worst=[]
            for r in np.argsort(-h)[:50]:
                worst.append(dict(row=int(r),source='original' if r<d.nbase else 'paths',
                    device=int(device[r]),witnessed=bool(witness[r]),residual=float(h[r]),cost=float(d.cost[r]),
                    terminal_entry=bool(entry[r]),source_observation=d.nodes[d.src[r]].tolist()))
            write(target/'worst_edges.json',worst)
        groups=dict(original=origin,paths=~origin)
        groups.update({f'device_{i}':origin&(device[rows]==i) for i in range(4)})
        value_groups[arm]={name:dict(task=distribution(task[m]),pair=distribution(pair[m]),
            b1_above_chain_bound=int((pair[m]>old['b1_chain_upper'][m]+1e-5).sum())) for name,m in groups.items()}
    constraints={name:{arm:residual_stats(h[m]) for arm,h in residuals.items()} for name,m in masks.items()}
    summary=dict(verification=verification,constraints=constraints,four_path=paths,values=value_groups,dual=dual_stats)
    write(out/'summary.json',summary);write(out/'constraints.json',constraints);write(out/'dual.json',dual_stats)
    plot(out,records,paths,path_values);report(out,summary,protocol)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    analysis_sources=['experiments/large_phr_calibration/analyze.py','experiments/witnessed_task_push/analyze.py',
                      'experiments/four_path_execution/common.py']
    for p in analysis_sources:
        dest=snapshot/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(out/'analysis_manifest.json',dict(sources={p:file_digest(p) for p in analysis_sources},
        references='Recorded graph and B1 upper bounds used only after both fixed 5k runs completed. No Oracle.'))
    write(out/'COMPLETE',dict(extra_steps_per_arm=5000,analysis_complete=True,closed_loop=False))


def plot(out,records,paths,path_values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(A='#2878b5',B='#e27925',parent='#888888')
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,group,key,title in zip(axs.flat,['original','original','paths','paths'],
            ['mean_positive','max_violation','mean_positive','violation_count'],
            ['Original mean positive residual','Original maximum violation','Four-path mean positive residual','Four-path violation count']):
        for arm in ('A','B'):
            ax.plot([r['continuation_step'] for r in records[arm]],
                    [r[group][key] for r in records[arm]],label=arm,color=colors[arm])
        ax.set(title=title,xlabel='Extra primal updates',ylabel=key);ax.legend()
    fig.savefig(out/'constraints.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,ax in enumerate(axs.flat):
        reference=[v for v in path_values['parent'] if v['path']==i]
        ax.plot([v['timestep'] for v in reference],[v['reference'] for v in reference],color='black',label='recorded graph')
        for arm in ('parent','A','B'):
            v=[v for v in path_values[arm] if v['path']==i]
            ax.plot([x['timestep'] for x in v],[x['value'] for x in v],label=arm,color=colors[arm])
        ax.set(title=f'Path {i+1}',xlabel='Recorded timestep',ylabel='Task distance');ax.legend()
    fig.savefig(out/'path_values.png',dpi=170);plt.close(fig)


def report(out,s,protocol):
    c=s['constraints'];f=s['four_path'];v=s['verification'];arms=('parent','A','B')
    text=['# 大 replay PHR 校准：5k A/B 续训','',
        '两组均从同一个 witnessed final（step 25,000、seed 606）继承完整模型、AdamW、dual 和采样状态，续训到 step 30,000。只改变 B 的原始边 dual 更新机制；task push、B1、primal 采样、模型和环境保持不变。','',
        '**本次结果：B 明显改善原始边约束，但显著损害四路径长程价值，没有实现联合改善。A 的长程估值基本维持父模型水平，原始边残差仅小幅下降。当前 B 配置不能作为同时恢复可行性与价值的替代方案。** 以下违反数始终使用 h>1e-5 阈值。','',
        '| 指标 | witnessed 父模型 | A：按访问 dual | B：全量外层 dual |','|---|---:|---:|---:|']
    for group,label in [('original','原始 20 万边'),('paths','四路径边'),
                        ('original_terminal_entry','原始入终态边'),('path_terminal_entry','四路径入终态边')]:
        for key,metric in [('violations','违反数'),('max_violation','最大正残差'),('mean_positive','平均正残差')]:
            text.append('| '+label+' '+metric+' | '+' | '.join(f'{c[group][a][key]:.6g}' for a in arms)+' |')
    for key,label in [('task_nrmse_percent','四路径 task NRMSE (%)'),('push_gap_percent','四路径 push gap (%)'),('task_rmse','四路径 task RMSE')]:
        text.append('| '+label+' | '+' | '.join(f'{f[a][key]:.4f}' for a in arms)+' |')
    change=100*(c['original']['B']['mean_positive']/c['original']['A']['mean_positive']-1)
    text += ['',f'B 相对 A 的原始边平均正残差变化 {change:+.2f}%；四路径 NRMSE 差为 {f["B"]["task_nrmse_percent"]-f["A"]["task_nrmse_percent"]:+.3f} 个百分点。需同时检查长程估值与局部可行性，不能将压缩距离造成的残差下降自动视作恢复。','',
        '| 路径起点 | 父模型 | A | B | 记录图参考 | B edge slack | B triangle slack |','|---|---:|---:|---:|---:|---:|---:|']
    for i in range(4):
        p=f['B']['paths'][i]
        text.append(f'| P{i+1} | '+' | '.join(f'{f[a]["paths"][i]["start"]:.3f}' for a in arms)+
            f' | {p["reference"]:.3f} | {p["edge_slack"]:.3f} | {p["triangle_slack"]:.3f} |')
    text += ['', 'edge slack 与 triangle slack 之和为路径起点低估，完整路径曲线保留在图中。记录图近似恢复不是未见状态泛化或闭环成功，本轮没有候选或导航评估。',
        'B 的四条路径 edge slack 增至约 78–106，A 仅约 5–25，说明路径预测边长明显收缩。P3/P4 的 triangle slack 甚至比 A 略低，但 edge slack 的增加更大，导致起点距离继续下降；这不是原有长程关系已经恢复。合并图对这 335 个路径源的参考与四路径图逐点一致，参考误差为零，排除了两组使用不同图目标的解释。',
        f'原始入终态边 B 仍有 {c["original_terminal_entry"]["B"]["violations"]}/{c["original_terminal_entry"]["B"]["count"]} 条违反；四路径入终态边 A 为 {c["path_terminal_entry"]["A"]["violations"]}/4，B 为 {c["path_terminal_entry"]["B"]["violations"]}/4。B 的四路径总违反数很少，但其最大违反量更大，不能把“只剩几条”当成近似恢复。','',
        '## 剩余瓶颈与解释范围','',
        f'原始 dual 最终最大值 A={s["dual"]["A"]["original"]["max"]:.1f}、B={s["dual"]["B"]["original"]["max"]:.1f}；平均绝对互补残差 |lambda*h| 分别为 {s["dual"]["A"]["original"]["complementarity_mean"]:.3f} 与 {s["dual"]["B"]["original"]["complementarity_mean"]:.3f}。B 的值是在最终外层更新后统计；不同 dual 强度不能被当作等预算收敛指标，且两组均未满足全量可行性。',
        '在本配置下，B 施加更强且更及时的原始 dual 约束压力，进一步降低了残差；但在当前共享模型、采样和 5k 小学习率预算下，所选 eta=100/周期250 的校准伴随距离收缩。可能涉及外层步长与内层适应不足、不同来源对共享表示的竞争；本实验没有分别隔离这些原因。标准 PHR 系数并不保证有限内层更新已经充分优化，不能据此否定所有全量 dual 更新方案，也不应把当前 B 当作成功校准。','',
        '## 更新机制与预算','',
        'A 保持每次访问更新原始 dual：lambda ← relu(lambda+h_pre)，eta=1。B 在 primal 中固定原始 dual，每 250 步对当前模型全量扫描后同步更新：lambda ← relu(lambda+100*h_latest)。rho=100，因此外层 eta=100 使用标准 PHR 系数。四路径两组都每步更新、eta=1，未改 task push 或其空 batch 语义。',
        '两组都继承完整 Adam 状态，使用相同 2e-6→2e-7 指数降步长，共 5k 更新；这相对父模型末步 2e-7 是共同的低步长重启。A 控制这部分额外预算与学习率改变。两组每 250 步都全量审计，不根据参考值停止。',
        'B 共 20 次外层更新；最后一次在第 5,000 步末，只改变保存的 dual，未反馈到本轮最终 primal 模型。本轮模型受前 19 次更新影响。B 每边名义 eta 总和 2,000，A 约 6.4；因此这是更新及时性、频率和强度一起改变的校准机制对照，不是单纯刷新频率的因果实验。','',
        '## 完整性与保留的失败','',
        f'初始模型、Adam、dual、采样器/RNG 和计数与父 checkpoint 逐位相同；最终 A/B 采样、访问和 push 计数逐位一致。每条原始物理边新增访问 {v["extra_visits"]["min"]}–{v["extra_visits"]["max"]} 次，原始 B1 新增 {v["extra_push_counters"]["original_pair_visits"]:,} 次，四路径 B1 新增 {v["extra_push_counters"]["path_pair_visits"]:,} 次。原始空 task batch 新增 {v["extra_push_counters"]["original_empty_task_batches"]} 次。所有真实边和有效 B1 都保留。',
        f'独立重算第一个外层模型的全部残差，B dual 与 relu(父 dual+100*h_latest) 的最大误差为 {v["first_outer_full_formula_max_error"]:.6g}。checkpoint 恢复验证保证边界更新不重做。父模型及旧结果哈希未变。',
        '全部最终残差/dual 保存于各组 final_constraints.npz；最坏 50 边、各任务统计、B1 上界违反和完整曲线均保留。752 个 quotient anchor 恒为零，真实入终态边独立列出。源码快照、父文件哈希、配置及复现命令见 protocol.json 和 experiments/large_phr_calibration/README.md。',
        '正式训练固定 4 CPU 线程，沿用父实验已记录的微小数值非确定性；相同 seed 与采样不保证全部浮点轨迹逐位一致。结论仅覆盖这一个父 checkpoint 和所选 eta/周期，不代表外层参数最优或跨 seed 稳健。Oracle 未用于训练、校准或停止，无新数据和闭环评估。','',
        '![约束曲线](constraints.png)','', '![路径值](path_values.png)','']
    (out/'report.md').write_text('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
