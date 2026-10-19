"""Post-construction audit only. References here must never be training labels."""
from __future__ import annotations
import argparse
from collections import Counter
import heapq
import itertools
import json
from pathlib import Path
import numpy as np
import torch

from minimal_qrl.eval.task_path_support_audit import (physical_episodes,
    validate_physical_dynamics, valid_anchor_rows, exact_reachable)
from minimal_qrl.path_support import replay_actions
from minimal_qrl.baselines.mppi import _state_terms
from .build import DEVICE, ROOT, sha256, write_json, execute_segment
from .data import load_data


def reverse_distances(n,src,dst,cost,goals):
    reverse=[[] for _ in range(n)]
    for i,(a,b,c) in enumerate(zip(src,dst,cost)):
        # Python float accumulation: NumPy 2 scalar promotion can otherwise
        # accumulate float32 edge weights in float32 before assigning to dist.
        reverse[b].append((int(a),float(c),i))
    dist=np.full(n,np.inf);next_edge=np.full(n,-1,dtype=np.int64);queue=[]
    for g in goals:
        dist[g]=0;heapq.heappush(queue,(0,int(g)))
    while queue:
        du,u=heapq.heappop(queue)
        if du != dist[u]:continue
        for v,c,i in reverse[u]:
            candidate=du+c
            if candidate < dist[v]:
                dist[v]=candidate;next_edge[v]=i;heapq.heappush(queue,(candidate,int(v)))
    return dist,next_edge


def exact_ids(observations):
    # Full observation includes heading and task. No nearest-neighbour lookup.
    mapping={};ids=[]
    for row in observations:
        key=row.tobytes()
        if key not in mapping:mapping[key]=len(mapping)
        ids.append(mapping[key])
    return np.asarray(ids),mapping


def graph_structure(src,dst,rows,n):
    incoming=[set() for _ in range(n)];outgoing=[set() for _ in range(n)]
    for i in rows:
        incoming[dst[i]].add(int(src[i]));outgoing[src[i]].add(int(dst[i]))
    return incoming,outgoing


def require_equal(actual,expected,label):
    if not np.array_equal(np.asarray(actual),np.asarray(expected)):
        raise ValueError(f'exact replay mismatch: {label}')


def audit(root, *, recombine=True):
    root=Path(root);out=root/'audit';out.mkdir(exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha256(root/name)==digest,name
    # Verify archived source and unchanged legacy inputs, without touching them.
    for name,digest in manifest['sources'].items():
        assert sha256(root/'source_snapshot'/name)==digest,name
        assert sha256(ROOT/name)==digest,('source changed',name)
    data=load_data(root);other=load_data(root,task_mode='witnessed')
    raw,env=data.raw,data.env;infos=raw.transition_infos
    for field in ('global_push_task_mask','global_push_task_witness_mask','global_push_pair_mask',
                  'global_push_pair_source_obs_index','global_push_pair_goal_obs_index'):
        assert torch.equal(infos[field],other.raw.transition_infos[field]),field
    assert data.fingerprint==other.fingerprint
    assert infos['global_push_task_mask'][data.physical].all()
    assert not infos['global_push_task_mask'][~data.physical].any()
    assert not infos['goal_return_mask'].any() and not infos['goal_return_cost'].any()
    assert not raw.timeouts.any()
    episodes,anchors,sources=physical_episodes(raw)
    dynamics=validate_physical_dynamics(raw,episodes,env)
    valid,anchor_records=valid_anchor_rows(raw,anchors,sources,env)
    assert len(valid)==len(episodes)==64
    reach={scope:exact_reachable(raw,episodes,valid,sources,scope=scope) for scope in
           ('strict_context','same_task_goal')}
    assert all(mask[data.physical.numpy()].all() for mask in reach.values())
    segments=json.loads((root/'segments.json').read_text())
    records=json.loads((root/'trajectories.json').read_text())
    # Replay every saved segment in the actual environment, including costs and observations.
    for name,segment in segments.items():
        actual=execute_segment(env,segment['start'],segment['actions'],success=name.startswith('tail'))
        for field in ('states','observations','actions','rewards'):
            require_equal(np.asarray(actual[field],np.float32),np.asarray(segment[field],np.float32),name+'/'+field)
    obs=raw.all_observations.numpy();cost=data.cost.numpy();physical=data.physical.numpy()
    independent_cost_error=[];suffix=np.full(len(cost),np.nan);b1_upper=np.full(len(cost),np.nan)
    lengths=[]
    for ep,record in zip(episodes,records):
        rows=ep['rows'];actions=raw.actions[rows].numpy()
        names=record['segments']
        recorded_states=np.concatenate([np.asarray(segments[n]['states'],np.float32)[:-1] for n in names]
                                       +[np.asarray(segments[names[-1]]['states'],np.float32)[-1:]])
        actual=replay_actions(env,recorded_states[0],actions,device_id=DEVICE,seed=record['seed'])
        require_equal(np.asarray(actual['states'],np.float32),recorded_states,'full trajectory states')
        require_equal(np.asarray(actual['observations'],np.float32),obs[ep['obs_ids']],'replay observations')
        require_equal(np.asarray(actual['rewards'],np.float32),raw.rewards[rows].numpy(),'one-step rewards')
        states=recorded_states[1:];zero=np.zeros(len(rows),bool)
        independent,_=_state_terms(env,states,zero,zero)
        independent_cost_error.extend(abs(independent-cost[rows]).tolist())
        suffix[rows]=np.cumsum(cost[rows][::-1],dtype=np.float64)[::-1]
        prefix=np.r_[0,np.cumsum(cost[rows],dtype=np.float64)]
        pairs=infos['global_push_pair_goal_obs_index'][rows].numpy()
        assert (pairs>sources[rows]).all() and (pairs<=ep['obs_ids'][-1]).all()
        b1_upper[rows]=prefix[pairs-ep['obs_ids'][0]]-prefix[np.arange(len(rows))]
        lengths.append(len(rows))
    assert max(independent_cost_error)<5e-5
    ids,mapping=exact_ids(obs);s=ids[sources];t=ids[sources+1]
    inc,outgoing=graph_structure(s,t,np.flatnonzero(physical),len(mapping))
    branch=[i for i,v in enumerate(outgoing) if len(v)>1]
    merge=[i for i,v in enumerate(inc) if len(v)>1]
    unique_edges={(int(s[i]),int(t[i]),raw.actions[i].numpy().tobytes(),float(cost[i])) for i in np.flatnonzero(physical)}
    assert len(branch)>=2 and len(merge)>=2
    junctions={}
    for name,state in manifest['shared_states'].items():
        o=env.state_to_observation(np.asarray(state,np.float32)).astype(np.float32)
        node=mapping[o.tobytes()]
        incoming_rows=np.flatnonzero(physical & (t==node))
        outgoing_rows=np.flatnonzero(physical & (s==node))
        junctions[name]=dict(state=state,node=node,in_degree=len(inc[node]),out_degree=len(outgoing[node]),
            incoming_rows=incoming_rows.tolist(),outgoing_rows=outgoing_rows.tolist(),
            exact_full_observation=True)
    # The inner/outer alternatives share the first waypoint on each side:
    # EXIT splits upper/lower; each of those splits again further along.
    assert junctions['exit']['in_degree']==64 and junctions['exit']['out_degree']==2
    assert junctions['east_join']['in_degree']==4 and junctions['east_join']['out_degree']==2
    # Separate evaluation artifact, built only after physical checks passed.
    src,dst=data.src.numpy(),data.dst.numpy();goals=data.goal.unique().numpy()
    dist,next_edge=reverse_distances(len(data.nodes),src,dst,cost,goals)
    assert np.isfinite(dist).all()
    assert np.all(dist[src]<=cost+dist[dst]+1e-9)
    assert np.all(dist[goals]==0)
    branch_records=[]
    for node in branch:
        oi=int(np.flatnonzero(ids==node)[0])
        representatives={}
        for row in np.flatnonzero(physical & (s==node)):
            representatives.setdefault(int(t[row]),int(row))
        alternatives=[dict(row=row,next_raw_node=target,action=raw.actions[row].tolist(),
            one_step_cost=float(cost[row]),recorded_graph_cost_to_go=float(cost[row]+dist[dst[row]]))
            for target,row in representatives.items()]
        branch_records.append(dict(raw_node=node,source_observation=obs[oi].tolist(),
            physical_state=env.observation_to_state(obs[oi]).tolist(),
            in_degree=len(inc[node]),out_degree=len(outgoing[node]),alternatives=alternatives,
            branch_cost_gap=max(a['recorded_graph_cost_to_go'] for a in alternatives)
                            -min(a['recorded_graph_cost_to_go'] for a in alternatives)))
    assert all(b['branch_cost_gap']>1e-5 for b in branch_records)
    # Check shortest witness chain for every canonical node; no virtual physical edge.
    for start in range(len(dist)):
        node=start;total=0.;seen=set()
        while node not in goals:
            assert node not in seen;seen.add(node)
            edge=int(next_edge[node]);assert edge>=0 and src[edge]==node
            total+=float(cost[edge]);node=int(dst[edge])
        assert abs(total-dist[start])<1e-8
    saving=suffix[physical]-dist[src[physical]]
    assert saving.min()>-1e-8
    np.savez_compressed(out/'graph_reference_audit_only.npz',distance=dist,next_edge=next_edge,
        source=src,target=dst,physical=physical,recorded_suffix_cost=suffix,b1_chain_upper=b1_upper)
    combinations=[]
    if recombine:
        # Exhaustively execute all 512 combinations, including 448 never stored as episodes.
        for pre,route,tail in itertools.product([f'prefix_{i:02d}' for i in range(64)],
                ['upper_inner','upper_outer','lower_inner','lower_outer'],['tail_left','tail_right']):
            names=[pre,route,tail];actions=sum([segments[n]['actions'] for n in names],[])
            actual=replay_actions(env,segments[pre]['start'],actions,device_id=DEVICE,seed=20261019)
            expected=np.concatenate([np.asarray(segments[n]['states'],np.float32)[:-1] for n in names]
                                     +[np.asarray(segments[tail]['states'],np.float32)[-1:]])
            require_equal(np.asarray(actual['states'],np.float32),expected,'recombination')
            combinations.append(dict(segments=names,steps=len(actions),success=True))
        write_json(out/'recombination_checks.json',combinations)
    # Full normal transition payload only, with a strict metadata schema guard.
    forbidden=[k for k in infos if any(word in k.lower() for word in ('oracle','shortest','value_label','nstep'))]
    assert not forbidden
    route_costs={n:dict(steps=len(seg['actions']),cost=float(-np.sum(seg['rewards'],dtype=np.float64)))
                 for n,seg in segments.items() if not n.startswith('prefix')}
    # Raw source-state coverage; repeated rows and distinct states both reported.
    positions=obs[sources[physical],:2]
    coverage={}
    for name,box in [('U interior',[2.8,1.9,5.2,5.4]),('west exit',[0,1.9,2.8,5.4]),
                     ('upper detour',[0,5.9,9,12]),('lower detour',[0,0,9,1.4]),
                     ('east approach',[5.7,1.4,10,5.9])]:
        x0,y0,x1,y1=box;mask=(positions[:,0]>=x0)&(positions[:,0]<=x1)&(positions[:,1]>=y0)&(positions[:,1]<=y1)
        coverage[name]=dict(rows=int(mask.sum()),unique_full_source_states=len(set(s[physical][mask].tolist())))
    result=dict(passed=True,trajectories=64,physical_rows=int(physical.sum()),anchors=len(valid),
        unique_physical_edges=len(unique_edges),duplicate_physical_rows=int(physical.sum())-len(unique_edges),
        exact_raw_nodes=len(mapping),canonical_nodes=len(data.nodes),
        nonterminal_branch_nodes=len(branch),nonterminal_merge_nodes=len(merge),junctions=junctions,
        branch_choices=branch_records,
        segment_count=len(segments),trajectory_steps=dict(min=min(lengths),max=max(lengths)),
        dynamics=dynamics,bit_exact_full_episode_replays=64,bit_exact_segment_replays=len(segments),
        independent_cost_max_error=max(independent_cost_error),legal_anchors=anchor_records,
        all_equals_witnessed=True,task_witness_rows=int(infos['global_push_task_witness_mask'].sum()),
        reachable_rows={k:int(v.sum()) for k,v in reach.items()},valid_B1_rows=int(infos['global_push_pair_mask'].sum()),
        graph_reference_is_audit_only=True,graph_shortest_witnesses_checked=len(dist),
        sources_with_shorter_recombined_route=int((saving>1e-5).sum()),
        max_recorded_suffix_minus_graph=float(saving.max()),
        recombinations_checked=len(combinations),novel_recombinations_checked=max(0,len(combinations)-64),
        recombination_max_steps=max((r['steps'] for r in combinations),default=None),
        routes=route_costs,coverage=coverage,fingerprints=data.fingerprint,
        no_oracle_or_return_labels=True,unchanged_legacy_and_core_source_hashes=True,training_started=False)
    write_json(out/'summary.json',result)
    plot(root,segments,result)
    report(root,result)
    write_json(out/'artifact_hashes.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:result[k] for k in ('passed','physical_rows','unique_physical_edges',
        'nonterminal_branch_nodes','nonterminal_merge_nodes','all_equals_witnessed','recombinations_checked')},indent=2))
    return result


def plot(root,segments,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    scenario=json.loads((root/'scenario.json').read_text())
    fig,ax=plt.subplots(figsize=(10,7))
    for obstacle in scenario['obstacles'][:3]:
        ax.add_patch(Rectangle((obstacle['x_min'],obstacle['y_min']),obstacle['x_max']-obstacle['x_min'],
            obstacle['y_max']-obstacle['y_min'],color='#444444'))
    colors=dict(upper_inner='#2979b9',upper_outer='#49a5db',lower_inner='#bd5120',lower_outer='#e78e42',
                tail_left='#22814b',tail_right='#7e4aaf')
    for name,segment in segments.items():
        xy=np.asarray(segment['states'])[:,:2]
        if name.startswith('prefix'):
            ax.plot(*xy.T,color='#999999',alpha=.25,lw=.6)
        else:
            ax.plot(*xy.T,color=colors[name],lw=1.7,label=f"{name}: {result['routes'][name]['cost']:.2f}")
            j=len(xy)//2;d=xy[j+1]-xy[j]
            ax.annotate('',xy=xy[j]+3*d,xytext=xy[j],arrowprops=dict(arrowstyle='->',color=colors[name],lw=2))
    for name,j in result['junctions'].items():
        x,y,h=j['state'];ax.scatter(x,y,c='black',zorder=5)
        ax.annotate(f"{name}\nin={j['in_degree']}, out={j['out_degree']}",(x,y),xytext=(x-.7,y+.6))
        ax.arrow(x,y,.35*np.cos(h),.35*np.sin(h),head_width=.1,color='black')
    ax.scatter(6.7,3.65,marker='*',s=120,color='#d7a000',label='task target')
    ax.set(xlim=(.5,10),ylim=(0,7.6),xlabel='x',ylabel='y',
           title='Expert-assisted U-trap network: exact state joins; unchanged dynamics')
    ax.set_aspect('equal');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(root/'audit/network.png',dpi=180);plt.close(fig)


def report(root,r):
    route_rows='\n'.join(f"| {n} | {v['steps']} | {v['cost']:.6f} |" for n,v in r['routes'].items())
    coverage='\n'.join(f"| {n} | {v['rows']} | {v['unique_full_source_states']} |" for n,v in r['coverage'].items())
    text=f'''# 专家辅助 U-trap 转移网络：构造与审计

本轮仅构造数据并审计，未启动训练或保存模型；兼容性测试仅实例化生产网络并做无梯度前向。64 条成功轨迹包含 **{r['physical_rows']:,} 条真实转移记录、{r['unique_physical_edges']:,} 条去重物理边、64 条合法零成本 anchor**。{r['duplicate_physical_rows']:,} 条重复记录来自共享片段；不是同样数量的独立样本。每条轨迹 {r['trajectory_steps']['min']}–{r['trajectory_steps']['max']} 步，环境上限保持 300。

64 个 U 内前缀精确汇合到出口 `(2, 3.65, 3)`；出口先选上/下，两侧各自再选内/外，形成 4 条绕行路线，它们精确汇合到东侧非终态 `(8.4, 3.65, 3)`；随后有 2 条不同接近路线到合法任务终态。两个交汇节点的 heading 均为 3 rad。原始完整 42 维观测中共有 {r['nonterminal_branch_nodes']} 个分支节点、{r['nonterminal_merge_nodes']} 个交汇节点；出口入/出度为 64/2，东侧为 4/2。度数按不同前驱/后继计算，不把重复行算成分支。四处分支的记录图成本间隔均非零，逐分支动作和成本见 `summary.json` 的 `branch_choices`。终态 quotient 合并另行处理，不充作物理交汇。

专家辅助来源是本轮助手在脚本中显式设计的起点、绕行 waypoint 和接近控制，加仅优化端点几何的 float32 控制序列 shooting；没有真人示教或外部专家标注。提议动力学只用于求控制；每一段、每一条最终轨迹都由原环境 `step` 重新执行。连接要求 `(x,y,heading)` 和完整观测逐位一致；没有状态吸附、空间近邻边或段内 reset。方向受限、障碍物、任务条件、一步成本均沿用原环境。

| 片段 | 步数 | 累计环境成本（仅报告） |
|---|---:|---:|
{route_rows}

共享出口和东侧交汇使 64×4×2=512 种完整组合有记录边支持；训练 replay 仅存 64 种。审计实际重新执行了 {r['recombinations_checked']} 种，其中 {r['novel_recombinations_checked']} 种未作为完整训练轨迹保存，全部逐位匹配组成片段并合法成功，最长 {r['recombination_max_steps']} 步。它们未追加到训练数据。记录图审计发现 {r['sources_with_shorter_recombined_route']:,} 个源记录可通过改选分支取得更低成本，最大较本条记录后缀低 {r['max_recorded_suffix_minus_graph']:.6f}。这使路线排序、交汇后的片段拼接成为实质诊断；图最短距离只保存在 `audit/graph_reference_audit_only.npz`。

全部 {r['task_witness_rows']:,} 个 task push 源，在自己的 context 内就有连续成功后缀及合法 anchor，跨轨迹拼接不是见证的前提。现有 `configure_global_push_sampling(state_mode="trajectory", seed=42)` 重建 B1 后，all 与 witnessed 的 task mask、B1 索引和指纹完全相等。{r['valid_B1_rows']:,} 个 B1 配对均严格位于本轨迹后续物理状态内。anchor 在既有生产 canonicalization 下映为恒等边；入终态的物理边仍保留原始正成本。

审计重放全部 64 条轨迹和 70 个片段，观测、动作、奖励逐位一致；另用现有向量成本实现重算，最大成本差 {r['independent_cost_max_error']:.3g}。解析一步动力学最大位置/heading 误差为 {r['dynamics']['max_position_error']:.3g}/{r['dynamics']['max_heading_error']:.3g}（观测编码浮点误差），无碰撞、越界、超时或中途终止。严格 context 可达和同任务精确图可达均覆盖所有物理源。图中的全部 {r['graph_shortest_witnesses_checked']} 个 canonical 节点都有逐边核验的合法最短路见证。

| 覆盖区域 | 源记录数 | 去重完整源状态数 |
|---|---:|---:|
{coverage}

数据指纹 `replay_sha256`：`{r['fingerprints']['replay']}`。文件 SHA256、环境签名、旧输入/源码哈希和源码快照见 `manifest.json`；节点、成本、固定 B1 指纹见 `audit/summary.json`。`construction_attempts.json` 保留未成功的 shooting 长度候选；它们不进入 replay。旧实验产物及环境/MPPI/encoder 源码哈希均核验未变。

沿用四路径的 `MultiEpisodeData`、一步 reward/cost 和生产终态映射；完整 replay、witnessed 与 PHR 校准提示需同时诊断可达支持、长程紧致性和约束可行性，不能以残差下降单独判定距离恢复。训练 payload 没有 Oracle value 或最短路回归标签，既有 `goal_return_cost` 为零且 mask 全关。加载器不会读取 audit 参考。后续入口与复现方式见仓库 `experiments/expert_utrap_network/README.md`。

这是有专家辅助、单任务、有限精确状态网络上的机制诊断数据。起点与路线是设计样本，可能靠近既有测试位置；不继承旧四路径的 held-out 排除协议，不能拿旧探针宣称独立泛化。不宣称无专家学习、连续环境最优性或闭环 MPPI 改善。共享片段会改变按行采样权重，未来若去重或改权重，应另立协议并重建 B1/见证，不能静默替换当前数据。

![网络与路线成本](network.png)
'''
    (root/'audit/report.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    a=p.parse_args();torch.set_num_threads(1);audit(a.root)
