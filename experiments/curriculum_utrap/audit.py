"""Post-collection audit: real replay, reset exclusions, gates, exact support, comparison."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from quasimetric_rl.data import MultiEpisodeData
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.eval.task_path_support_audit import physical_episodes,valid_anchor_rows,exact_reachable
from .common import ROOT,DEVICE,sha,write
from .collect import coverage_key,batch_metrics,sample_start
from .data import load_training_data,audit_training_data


def random_actions(seed,n,cfg,omega_max):
    rng=np.random.default_rng(seed);p=cfg['policy'];actions=[]
    probability=p['straight_action_probability']
    while len(actions)<n:
        hold=int(rng.integers(p['action_hold_min_steps'],p['action_hold_max_steps']+1))
        scale=float(rng.choice(np.asarray([-1.,-.5,0.,.5,1.],np.float32),
            p=[.25*(1-probability),.25*(1-probability),probability,.25*(1-probability),.25*(1-probability)]))
        actions.extend([[scale*omega_max]]*min(hold,n-len(actions)))
    return np.asarray(actions,np.float32)


def equal(a,b,name):
    if not np.array_equal(a,b):raise ValueError('bit-exact mismatch: '+name)


def state_bins(observations,cfg):
    return {coverage_key([o[0],o[1],np.arctan2(o[3],o[2])],cfg) for o in observations}


def wilson(success,n):
    if not n:return None
    z=1.96;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def descriptive(path,label,env,cfg):
    path=Path(path).resolve()
    payload=torch.load(path,map_location='cpu',weights_only=True);raw=MultiEpisodeData(**payload['raw'])
    assert replay_fingerprint(raw)==payload['sha256']
    eps,anchors,sources=physical_episodes(raw)
    eps=[e for e in eps if e['device']==0]
    anchors=[a for a in anchors if int(raw.transition_infos['device_index'][a])==0]
    valid,_=valid_anchor_rows(raw,anchors,sources,env)
    reach=exact_reachable(raw,eps,valid,sources,scope='strict_context')
    rows=np.concatenate([e['rows'] for e in eps]);obs=raw.all_observations.numpy()[sources[rows]]
    outcomes=Counter(e['outcome'] for e in eps);bins=state_bins(obs,cfg)
    return dict(label=label,path=str(path),file_sha256=sha(path),
        replay_sha256=payload['sha256'],physical_rows=len(rows),episodes=len(eps),
        successes=outcomes[1],success_rate=outcomes[1]/len(eps),
        witnessed_rows=int(reach.sum()),witness_fraction=float(reach[rows].mean()),
        position_heading_bins=len(bins),position_bins=len({b[:2] for b in bins}),
        heading_bins=len({b[2] for b in bins}),scope='u_trap_target only',
        distinct_starts=len({raw.all_observations[e['obs_ids'][0]].numpy().tobytes() for e in eps}))


def audit(root,output=None):
    root=Path(root);out=Path(output) if output else root/'audit';out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text());cfg=json.loads((root/'config.json').read_text())
    for name,digest in manifest['files'].items():assert sha(root/name)==digest,name
    audit_revisions={}
    for name,digest in manifest['collection_source_hashes'].items():
        assert sha(root/'source_snapshot'/name)==digest,name
        current=sha(ROOT/name)
        if current!=digest:
            # This loader is not imported or called by collect.py. Preserve the
            # original snapshot and explicitly record its post-collection fix.
            assert name=='experiments/curriculum_utrap/data.py',('changed collection source',name)
            audit_revisions[name]=dict(collection_snapshot_sha256=digest,current_sha256=current,
                reason='Startup cost validation now uses authoritative scalar environment cost; MPPI vector discrepancy is diagnostic only. Collection and stored costs unchanged.')
            shutil.copy2(ROOT/name,out/'training_data_source.py')
    excluded_meta=json.loads((root/'exclusions.json').read_text());excluded=np.asarray(excluded_meta['states'],np.float32)
    for name,digest in excluded_meta['source_hashes'].items():assert sha(ROOT/name)==digest
    d=load_training_data(root);w=load_training_data(root,task_mode='witnessed');verified=audit_training_data(d)
    raw,env=d.raw,d.env;eps,anchors,sources=physical_episodes(raw)
    records=[json.loads(line) for line in (root/'episodes.jsonl').read_text().splitlines()]
    collection=json.loads((root/'collection_summary.json').read_text())
    decisions=json.loads((root/'decisions.json').read_text())
    states_file=np.load(root/'physical_states.npz');states=states_file['states'];lengths=states_file['lengths']
    assert len(records)==len(eps)==len(lengths)
    assert len({tuple(r['start']) for r in records})==len(records)
    assert len({r['seed'] for r in records})==len(records)
    offset=0;min_start_distance=np.inf;rollout_proximity=0;exact_pose=0;heldout_bytes={s.tobytes() for s in excluded}
    start_rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],29]));seen=set();rejections=Counter()
    for i,(ep,rec,length) in enumerate(zip(eps,records,lengths)):
        assert ep['context']==rec['context']==i and len(ep['rows'])==rec['steps']==length-1
        expected=states[offset:offset+length];offset+=length
        equal(expected[0],np.asarray(rec['start'],np.float32),'start')
        regenerated,rejected=sample_start(env,rec['stage'],start_rng,excluded,cfg,seen)
        equal(expected[0],regenerated,'reset RNG stream');rejections.update(rejected)
        stage=cfg['stages'][rec['stage']];x,y,h=expected[0]
        accepted_box=False
        for box,center in zip(stage['boxes'],stage['headings']):
            x0,y0,x1,y1=box
            diff=abs((float(h)-center+np.pi)%(2*np.pi)-np.pi)
            if x0<=x<=x1 and y0<=y<=y1 and diff<=stage['half_width']+1e-6:accepted_box=True
        assert accepted_box and not env.is_task_feasible(expected[0])
        nearest=float(np.linalg.norm(excluded[:,:2]-expected[0,:2],axis=1).min())
        assert nearest>cfg['exclusion_radius'];min_start_distance=min(min_start_distance,nearest)
        near=np.linalg.norm(expected[:,None,:2]-excluded[None,:,:2],axis=2).min(axis=1)
        rollout_proximity+=int((near<=cfg['exclusion_radius']).sum())
        exact_pose+=sum(s.tobytes() in heldout_bytes for s in expected)
        expected_seed=int(np.random.SeedSequence([cfg['seed'],i,104729]).generate_state(1)[0])
        assert rec['seed']==expected_seed
        actions=raw.actions[ep['rows']].numpy()
        equal(actions,random_actions(rec['seed'],len(actions),cfg,env.omega_max),'random control stream')
        obs,_=env.reset(seed=rec['seed'],options={'device_id':DEVICE,'start':expected[0]})
        equal(obs,raw.all_observations[ep['obs_ids'][0]].numpy(),'initial observation')
        for j,action in enumerate(actions):
            obs,reward,done,truncated,info=env.step(action);row=int(ep['rows'][j])
            equal(env.state,expected[j+1],'physical state')
            equal(obs,raw.all_observations[ep['obs_ids'][j+1]].numpy(),'observation')
            assert np.float32(reward)==raw.rewards[row].item()
            assert bool(raw.terminals[row])==done
            expected_timeout=truncated or (j==len(actions)-1 and not done)
            assert bool(raw.timeouts[row])==expected_timeout
            if j<len(actions)-1:assert not done and not truncated
        actual=('success' if info['success'] else 'collision' if info['collision'] else
                'out_of_bounds' if info['out_of_bounds'] else 'timeout' if truncated else 'budget_cutoff')
        assert actual==rec['outcome']
        equal(env.state,np.asarray(rec['terminal_state'],np.float32),'last state')
        for field,value in [('curriculum_stage',rec['stage']),('curriculum_batch',rec['batch']),
                            ('curriculum_phase',rec['phase']),('curriculum_role',rec['role'])]:
            assert (raw.transition_infos[field][ep['rows']]==value).all()
    assert offset==len(states) and not exact_pose
    assert dict(rejections)==collection['reset_rejections']
    # Independently recompute promotion and stopping decisions from fresh batches.
    mastered=[];frontier=0;streak=0;stagnant=0;main_steps=0
    schedule=np.random.default_rng(np.random.SeedSequence([cfg['seed'],11]))
    for decision in decisions:
        batch=decision['batch'];rr=[r for r in records if r['phase']==0 and r['batch']==batch]
        fr=[r for r in rr if r['role']==0]
        recovery=stagnant>=cfg['recovery_after']
        assert all(r['stage']==frontier for r in fr)
        assert all(r['stage'] in (mastered or [frontier]) for r in rr if r['role']==1)
        for r in rr:
            expected_stage=frontier if r['role']==0 or not mastered else int(schedule.choice(mastered))
            assert r['stage']==expected_stage
        n=cfg['recovery_frontier_episodes'] if recovery else cfg['normal_frontier_episodes']
        m=batch_metrics(fr,cfg);assert m==decision['frontier_metrics']
        complete=len(fr)==n;streak=streak+1 if complete and m['passed'] else 0
        stagnant+=1;promoted=streak>=cfg['required_consecutive_passes']
        assert decision['frontier']==frontier and decision['recovery']==recovery
        if promoted:mastered.append(frontier);frontier+=1;streak=0;stagnant=0
        main_steps+=sum(r['steps'] for r in rr)
        assert decision['promoted']==promoted and decision['mastered']==mastered
        assert decision['main_steps']==main_steps and decision['consecutive_passes']==streak
    assert collection['mastered_stages']==mastered
    assert collection['curriculum_steps']==main_steps<=cfg['curriculum_step_budget']
    survey=[r for r in records if r['phase']==1]
    for stage in cfg['diagnostic_stages']:
        assert sum(r['steps'] for r in survey if r['stage']==stage)==cfg['diagnostic_steps_per_stage']
    assert collection['total_real_steps']==len(d.original)==sum(r['steps'] for r in records)
    reason=collection['stop_reason']
    if reason=='stagnation':assert stagnant>=cfg['max_stagnant_rounds']
    elif reason=='curriculum_step_budget':assert main_steps==cfg['curriculum_step_budget']
    elif reason=='all_stages_mastered':assert frontier==len(cfg['stages'])
    else:assert reason=='max_rounds' and len(decisions)==cfg['max_rounds']
    valid,_=valid_anchor_rows(raw,anchors,sources,env)
    strict=exact_reachable(raw,eps,valid,sources,scope='strict_context')
    graph=exact_reachable(raw,eps,valid,sources,scope='same_task_goal')
    equal(strict,w.raw.transition_infos['global_push_task_mask'].numpy(),'witnessed mask')
    assert d.raw.transition_infos['global_push_task_mask'][d.original].all()
    assert torch.equal(d.pair,w.pair)
    groups=[];obs=raw.all_observations.numpy()
    for stage in range(len(cfg['stages'])):
        for phase in (0,1):
            selected=[(e,r) for e,r in zip(eps,records) if r['stage']==stage and r['phase']==phase]
            if not selected:continue
            rows=np.concatenate([e['rows'] for e,r in selected]);rec=[r for e,r in selected]
            bins=state_bins(obs[sources[rows]],cfg);start_bins={coverage_key(r['start'],cfg) for r in rec}
            outcomes=Counter(r['outcome'] for r in rec);success=outcomes['success']
            groups.append(dict(stage=stage,name=cfg['stages'][stage]['name'],phase='curriculum' if phase==0 else 'diagnostic',
                episodes=len(rec),successes=success,success_rate=success/len(rec),
                descriptive_wilson95=wilson(success,len(rec)),outcomes=dict(outcomes),real_rows=len(rows),
                witnessed_rows=int(strict[rows].sum()),same_task_reachable_rows=int(graph[rows].sum()),
                source_position_heading_bins=len(bins),source_position_bins=len({b[:2] for b in bins}),
                source_heading_bins=len({b[2] for b in bins}),start_position_heading_bins=len(start_bins),
                start_heading_bins=len({b[2] for b in start_bins}),
                frontier_episodes=sum(r['role']==0 for r in rec),revisit_episodes=sum(r['role']==1 for r in rec)))
    np.savez_compressed(out/'recorded_support.npz',physical=d.physical.numpy(),strict_witness=strict,
        same_task_goal_reachable=graph,stage=raw.transition_infos['curriculum_stage'].numpy(),
        phase=raw.transition_infos['curriculum_phase'].numpy())
    write(out/'stage_metrics.json',groups)
    with (out/'stage_metrics.csv').open('w') as f:
        keys=[k for k in groups[0] if k not in ('outcomes','descriptive_wilson95')]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for g in groups:writer.writerow({k:g[k] for k in keys})
    comparisons=[descriptive(ROOT/'results/diagnostic_path_support_positive/audit_baseline/verified_replay.pt',
        'original random U subset',env,cfg),descriptive(ROOT/'results/diagnostic_path_support_only/task_path_support_audit/verified_replay.pt',
        'four paths',env,cfg),descriptive(ROOT/'results/expert_utrap_network_20261019/main/verified_replay.pt',
        'expert network',env,cfg),descriptive(root/'verified_replay.pt','curriculum + diagnostic',env,cfg)]
    write(out/'comparison.json',comparisons)
    result=dict(passed=True,collection=collection,startup_audit=verified,stage_metrics=groups,
        independent_forward_steps_checked=len(d.original),random_actions_regenerated=True,
        reset_and_revisit_rng_regenerated=True,post_collection_validation_revisions=audit_revisions,
        exact_states_observations_actions_rewards=True,all_promotion_rules_recomputed=True,
        strict_witness_rows=int(strict.sum()),same_task_reachable_rows=int(graph.sum()),
        task_all_rows=len(d.original),all_equals_witnessed=bool(strict[d.original.numpy()].all()),
        heldout_reset_min_position_distance=min_start_distance,heldout_exclusion_radius=cfg['exclusion_radius'],
        rollout_states_within_exclusion_position_radius=rollout_proximity,exact_frozen_test_pose_matches=exact_pose,
        graph_shortest_paths_used=False,training_started=False,comparison= comparisons,
        collection_effective_source_code_unchanged=True,replay_sha256=manifest['replay_sha256'])
    write(out/'summary.json',result)
    plot(out,cfg,records,states,decisions)
    report(out,cfg,result)
    shutil.copy2(__file__,out/'audit_source.py')
    write(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (root/'COMPLETE').write_text('Collection and independent audit complete; no formal QRL training.\n')
    print(json.dumps({k:result[k] for k in ('passed','strict_witness_rows','same_task_reachable_rows','all_equals_witnessed','replay_sha256')}),flush=True)


def plot(out,cfg,records,states,decisions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    xy=states[::max(1,len(states)//6000)]
    ax[0].scatter(xy[:,0],xy[:,1],s=2,c='#bbbbbb',alpha=.3,label='recorded states')
    for phase,marker in [(0,'o'),(1,'x')]:
        rr=[r for r in records if r['phase']==phase];ss=np.array([r['start'] for r in rr])
        ax[0].scatter(ss[:,0],ss[:,1],c=[r['stage'] for r in rr],cmap='viridis',vmin=0,vmax=6,
            marker=marker,s=15,label='adaptive starts' if phase==0 else 'diagnostic starts')
    for x,y,w,h in [(2.8,5.4,2.9,.5),(2.8,1.4,2.9,.5),(5.2,1.4,.5,4.5),(9.,6.,6.,1.5)]:
        ax[0].add_patch(Rectangle((x,y),w,h,color='#444444'))
    ax[0].scatter(6.7,3.65,marker='*',s=100,c='orange',label='target')
    ax[0].set(xlim=(0,18),ylim=(0,12),xlabel='x',ylabel='y',title='Reset curriculum and retained forward transitions')
    ax[0].set_aspect('equal');ax[0].legend(fontsize=8)
    batches=[d['batch'] for d in decisions]
    ax[1].plot(batches,[d['frontier_metrics']['success_rate'] for d in decisions],'-o',label='fresh frontier batch success')
    ax[1].axhline(cfg['pass_success_rate'],ls='--',c='gray',label='success-rate gate (plus coverage)')
    for d in decisions:
        ax[1].annotate(str(d['frontier']),(d['batch'],d['frontier_metrics']['success_rate']),xytext=(0,8),textcoords='offset points',fontsize=8)
    ax[1].set(xlabel='independent batch',ylabel='success fraction',ylim=(0,1.05),title='Digits = frontier stage; no forced promotion')
    ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'coverage.png',dpi=160);plt.close(fig)


def report(out,cfg,s):
    c=s['collection'];m=s['stage_metrics'];cmp=s['comparison']
    rows='\n'.join(f"| {g['stage']} {g['name']} | {g['phase']} | {g['successes']}/{g['episodes']} ({g['success_rate']:.1%}) | {g['real_rows']} | {g['witnessed_rows']} | {g['start_heading_bins']}/{g['source_heading_bins']} | {g['source_position_heading_bins']} |" for g in m)
    comp='\n'.join(f"| {r['label']} | {r['physical_rows']} | {r['successes']}/{r['episodes']} ({r['success_rate']:.1%}) | {r['witnessed_rows']} ({r['witness_fraction']:.1%}) | {r['position_heading_bins']} |" for r in cmp)
    mastered=', '.join(str(i)+' '+cfg['stages'][i]['name'] for i in c['mastered_stages']) or '无'
    text=f'''# U-trap 随机动作课程采集与审计

已完成采集和独立审计，未启动正式 QRL 训练。共 **{c['total_real_steps']:,} 条真实转移、{c['episodes']} 个 episode、{c['outcomes'].get('success',0)} 条成功轨迹**，全部失败保留。合法 anchor 只来自真实成功末尾；失败末尾无目标边。动作完全匹配按独立 episode seed 重建的持续随机控制序列，无专家/规划器动作、Oracle value、回报或最短路训练标签。

## 预先固定的课程与实际停止

最多 30,000 步自适应课程＋6,000 步固定分层诊断（旧 200k 的 18% 上限）。实际课程 {c['curriculum_steps']:,} 步，诊断 {c['diagnostic_steps']:,} 步，停止原因 **{c['stop_reason']}**，未使用的课程预算 {c['unused_curriculum_budget']:,} 步不转移、不追加。已通过连续两批门槛的阶段：**{mastered}**。停止时前沿为阶段 {c['frontier_at_stop']}；未通过的阶段不能称为已掌握。

阶段按合法接近边界→接近区→东侧转弯空间→上下东角→上下外走廊→西出口→U 内部排列，并逐渐扩大 heading 扰动至完整圆周。分区依据 U 墙、开口方向和最小转弯半径 1/3，非欧氏距离排序；这些是目标/地图信息，故本采集不宣称目标盲探索。控制本身只从 ω/ωmax={{−1,−0.5,0,0.5,1}} 采样，直行概率 0.5、保持 3–10 步；关闭局部前视，动作不读取目标、成功库或规划器。

每批正常取 24 个前沿新起点＋8 个已掌握区域新起点；无已掌握区域时回访阶段0。连续两批前沿成功率≥12.5%、至少2次成功、至少8个位置-heading格与2个heading格，才晋级。3批未晋级后改为16前沿＋16回访；6批未晋级停止，另有24批与30k步上限。回访不参与晋级判定。所有起点和动作使用分离 RNG，episode seed 唯一；独立是指新批次、新起点、新随机流，阶段选择仍依赖过去统计。

课程结束后，阶段1–6各固定采集1000步诊断数据，**不参与晋级，也不表示课程到达这些阶段**。其作用是保留上下绕行、出口和U内的成功/失败实测证据，区分覆盖与成功链支持。下表明确区分两阶段，避免用诊断采样伪装课程推进。

| 阶段 | 数据来源 | 成功/episode | 真实转移 | witnessed源 | 起点/沿途heading格 | 沿途位置-heading格 |
|---|---|---:|---:|---:|---:|---:|
{rows}

失败类型总计：`{json.dumps(c['outcomes'],ensure_ascii=False)}`。逐阶段碰撞、越界、超时、预算截断与描述性 Wilson 区间见 `stage_metrics.json`；自适应、非同分布的批次不能将汇总区间解释成固定策略泛化置信保证。

## 记录路径支持与核验

严格 context 内可用于 witnessed task push 的源为 **{s['strict_witness_rows']:,}/{c['total_real_steps']:,}**；同任务完整观测逐位连接的图可达源为 **{s['same_task_reachable_rows']:,}**。`all` 与 `witnessed` 是否等价：**{s['all_equals_witnessed']}**。未见证失败源继续参与真实边 PHR 与合法同轨迹 B1，只从 witnessed task 最大化中排除。没有近邻合并、倒放边或 reset→起点的虚构物理边；位置覆盖不等于有记录成功链。

全部 {c['total_real_steps']:,} 条记录经独立正向重放，状态、42维观测、动作、float32 reward 逐位一致。启动审计从观测重建状态后，使用环境标量成本核验，最大差 {s['startup_audit']['recomputed_cost_max_error']:.3g}。MPPI 向量近似在极近目标处的最大差为 {s['startup_audit']['mppi_vector_cost_max_error']:.3g}（目标坐标精度影响 bearing）；此项只报告，不替换环境成本，不修改 MPPI。所有 anchor 重新判定合法；阶段晋级、起点/回访/动作 RNG、预算与 B1 边界均重新验证。上述重放是额外的验证计算，不计入采集数据预算。图只做采集后的精确可达性审计，本轮没有计算最短路。

冻结测试状态仅作为排除清单读取，起点位置排除半径0.25、不论heading；实际起点最小距离 {s['heldout_reset_min_position_distance']:.6f}，精确测试pose命中 {s['exact_frozen_test_pose_matches']}。沿途有 {s['rollout_states_within_exclusion_position_radius']} 个记录状态落入该位置邻域：它们不是主动测试起点重置，且为保留完整真实轨迹而不删边。因此不能把这些结果当作与测试分布完全分离的泛化证据。

## 描述性对照（统一筛选 u_trap_target）

| 数据 | 真实转移 | 成功/episode | 严格见证源 | 位置-heading格（0.25 / 24 bins） |
|---|---:|---:|---:|---:|
{comp}

这些不是配对因果实验：旧随机数据含局部安全前视及重复起点、任务预算为50k；本轮起点安排有目标/地图信息且无安全前视；四路径/专家网络使用规划器或设计控制、只含成功路径。专家网络另有精确分支与交汇，本课程的覆盖格不能替代其连接结构。比较文件只在采集结束后读取，不用于门槛、动作或选取数据；不宣称课程已经解决远端任务连接或优化困难。

## 文件、复现与训练入口

`verified_replay.pt` 沿用 MultiEpisodeData；成功与失败正常转移、一步成本和合法成功 anchor 都在同一数据内。`goal_return_cost`全零、mask全假。原采集器为失败额外采样的独立 anchor 被丢弃，既不接到失败末尾也不作为额外训练边。

采集后仅修正了训练加载器的成本核验依据（向量近似→环境标量），采集逻辑及数据未变。原加载器快照保留于 `source_snapshot`，新旧 SHA 与原因见 `summary.json` 的 `post_collection_validation_revisions`。

数据指纹：`{s['replay_sha256']}`。`config.json`、`protocol.json` 与源码快照在首次交互前冻结，`episodes.jsonl`、`decisions.json`、`physical_states.npz` 保留全部阶段与重放信息。`audit/recorded_support.npz` 仅用于审计，训练加载器不读取它。

复现：仓库根目录运行 `python -m experiments.curriculum_utrap.collect --output results/NEW_CURRICULUM`，随后 `python -m experiments.curriculum_utrap.audit --root results/NEW_CURRICULUM`；使用仓库既有虚拟环境，详见 `experiments/curriculum_utrap/README.md`。训练器新增 `--curriculum-data`，保留共享网络、AdamW、PHR与预算；该数据含失败，推荐显式 `--task-mode witnessed`。默认 all 保持旧命令兼容，但两种 task 目标在本数据上不同。本轮只验证接入，不启动正式训练。

![覆盖和独立批次成功率](coverage.png)
'''
    (out/'report.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',help='Fresh audit directory, for rechecking an existing collection without overwriting its audit')
    a=p.parse_args();torch.set_num_threads(1);audit(a.root,a.output)
