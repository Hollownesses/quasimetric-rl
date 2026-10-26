"""Post-completion paired analysis, real dynamics replay and support diagnostics."""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from .common import *
from .run import verify
from experiments.four_path_execution.statistics import aggregate,paired


def describe(x):
    a=np.asarray(x,dtype=float)
    if not a.size:return dict(n=0,mean=None,median=None,p95=None,max=None,min=None)
    assert np.isfinite(a).all()
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()),min=float(a.min()))

class Coverage:
    """Posthoc geometry only. Periodic joint metric with fixed 0.5 length/radian."""
    def __init__(self,support):
        self.support=np.asarray(support,dtype=float);self.xy=cKDTree(self.support[:,:2])
        expanded=np.concatenate([self.support+np.array([0,0,offset]) for offset in (-2*np.pi,0,2*np.pi)])
        scaled=expanded.copy();scaled[:,2]*=.5;self.joint=cKDTree(scaled)
    def query(self,states):
        s=np.asarray(states,dtype=float).reshape(-1,3)
        if not len(s):return np.zeros((0,5))
        nearest_xy,xyindex=self.xy.query(s[:,:2],workers=1)
        scaled=s.copy();scaled[:,2]*=.5;distance,index=self.joint.query(scaled,workers=1)
        chosen=self.support[index%len(self.support)]
        delta=s-chosen;theta=abs((delta[:,2]+np.pi)%(2*np.pi)-np.pi)
        xytheta=abs((s[:,2]-self.support[xyindex,2]+np.pi)%(2*np.pi)-np.pi)
        return np.column_stack([nearest_xy,np.linalg.norm(delta[:,:2],axis=1),theta,distance,xytheta])

COVER_KEYS=('nearest_xy','joint_neighbor_xy','joint_neighbor_heading_rad','joint_metric','nearest_xy_heading_rad')

def detailed_group(rows):
    s=aggregate(rows)
    s['cost_by_outcome']={name:describe([r['cost'] for r in rows if r[name]]) for name in ('success','collision','out_of_bounds','timeout')}
    s['steps_by_outcome']={name:describe([r['steps'] for r in rows if r[name]]) for name in ('success','collision','out_of_bounds','timeout')}
    return s


def pair_details(a,b):
    result=paired(a,b);aa={(r['state_id'],r['planner_seed']):r for r in a};bb={(r['state_id'],r['planner_seed']):r for r in b}
    result['new_success']=[list(k) for k in aa if not aa[k]['success'] and bb[k]['success']]
    result['lost_success']=[list(k) for k in aa if aa[k]['success'] and not bb[k]['success']]
    both=[k for k in aa if aa[k]['success'] and bb[k]['success']]
    result['common_success_differences']=[dict(state_id=k[0],planner_seed=k[1],cost_delta=bb[k]['cost']-aa[k]['cost'],steps_delta=bb[k]['steps']-aa[k]['steps']) for k in both]
    result['common_success_cost_delta_distribution']=describe([bb[k]['cost']-aa[k]['cost'] for k in both])
    if both and all('cost_components' in aa[k] and 'cost_components' in bb[k] for k in both):
        result['common_success_cost_component_delta']={name:float(np.mean([bb[k]['cost_components'][name]-aa[k]['cost_components'][name] for k in both])) for name in COST_KEYS}
        assert abs(sum(result['common_success_cost_component_delta'].values())-result['joint_success_cost_delta'])<1e-8
    return result


def analyze(root):
    p=verify(root);out=root/'comparison';out.mkdir(exist_ok=False);data=load();env=data.env
    records=read(root/'states.json')['records'];expected={(r['id'],seed) for r in records for seed in SEEDS}
    # No aggregate report or model comparison is made until ALL four budgets finish.
    rows={}
    for arm in ARMS:
        directory=root/'arms'/arm;done=read(directory/'COMPLETE');assert done['episodes']==64
        assert done['episodes_sha256']==file_digest(directory/'episodes.jsonl')
        rows[arm]=[json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
        assert len(rows[arm])==64 and {(r['state_id'],r['planner_seed']) for r in rows[arm]}==expected
    # Support uses physical raw source AND next observations, not canonical goal poses.
    idx=torch.cat((data.src_obs[data.original],data.src_obs[data.original]+1)).unique()
    support=np.unique(poses(data.raw.all_observations[idx].numpy()),axis=0);coverage=Coverage(support)
    np.savez_compressed(out/'training_pose_support.npz',support=support)
    audits={};timings={};queries={};covers={};artifact_hashes={};paired_banks={};started=time.perf_counter()
    for arm in ARMS:
        directory=root/'arms'/arm;all_times=[];all_values=[];cov={k:[] for k in ('executed','candidate_all','candidate_valid_unfinished','queried')};counts=defaultdict(int)
        maxstate=maxreward=0.;step_count=0
        for row in rows[arm]:
            path=directory/f"{row['state_id']}_{row['planner_seed']}.npz";z=np.load(path);artifact_hashes[str(path)]=file_digest(path)
            env.reset(seed=row['planner_seed'],options={'device_id':'u_trap_target','start':np.array(row['pose'],np.float32)})
            np.testing.assert_array_equal(env.state,z['states'][0]);n=len(z['actions']);assert n==row['steps']
            savedinfo=read(path.with_suffix('.json'));artifact_hashes[str(path.with_suffix('.json'))]=file_digest(path.with_suffix('.json'))
            for i,action in enumerate(z['actions']):
                _,reward,done,truncated,info=env.step(action);step_count+=1
                maxstate=max(maxstate,float(abs(env.state-z['states'][i+1]).max()));maxreward=max(maxreward,abs(reward-z['rewards'][i]))
                np.testing.assert_array_equal([info[k] for k in COST_KEYS],z['cost_terms'][i])
                for k in COST_KEYS:assert info[k]==savedinfo['step_infos'][i][k]
                assert not(done or truncated) or i==n-1
            assert done or truncated
            assert info['success']==row['success'] and info['collision']==row['collision'] and info['out_of_bounds']==row['out_of_bounds']
            assert bool(truncated and not info['success'])==row['timeout']
            assert not row['success'] or env.is_task_feasible(env.state)
            assert abs(-z['rewards'].sum()-row['cost'])<1e-9
            for k,v in stage_metrics(z['states'],next(r for r in records if r['id']==row['state_id'])).items():assert row[k]==v
            bankkey=(row['state_id'],row['planner_seed']);bank=str(z['candidate_sha256'][0])
            assert bank==row['first_candidate_sha256']
            if bankkey in paired_banks:assert paired_banks[bankkey]==bank
            else:paired_banks[bankkey]=bank
            assert z['candidate_final_states'].shape==(n,128,3)
            mask=~(z['candidate_invalid']|z['candidate_success'])
            assert z['query_counts'].sum()==row['terminal_queries']==len(z['query_values'])
            if arm=='none':assert row['terminal_queries']==0
            else:
                np.testing.assert_array_equal(z['query_states'],z['candidate_final_states'][mask])
                np.testing.assert_array_equal(z['query_counts'],mask.sum(axis=1))
            actual_invalid=np.array([not env.is_valid_state(s) for s in z['query_states']],bool)
            np.testing.assert_array_equal(actual_invalid,z['query_invalid'])
            assert np.isfinite(z['query_values']).all() and (z['query_values']>=0).all()
            if arm=='oracle':
                cd=row['controller_diagnostics'];assert cd['oracle_value_queries']==row['terminal_queries']
                assert cd['oracle_value_unreachable_queries']==row['terminal_unreachable_queries']
                counts['cache_hit_episodes']+=int(cd['oracle_value_cache_hit'])
                counts['cache_miss_episodes']+=int(not cd['oracle_value_cache_hit'])
            counts['queries']+=row['terminal_queries'];counts['invalid_queries']+=row['terminal_invalid_queries'];counts['unreachable_queries']+=row['terminal_unreachable_queries']
            counts['candidate_count']+=n*128;counts['invalid_candidates']+=int(z['candidate_invalid'].sum())
            counts['all_invalid_decisions']+=int(z['candidate_invalid'].all(axis=1).sum())
            all_times.extend(z['planning_seconds']);all_values.extend(z['query_values'])
            sets=dict(executed=z['states'],candidate_all=z['candidate_final_states'].reshape(-1,3),candidate_valid_unfinished=z['candidate_final_states'][mask],queried=z['query_states'])
            coverage_arrays={name:coverage.query(states) for name,states in sets.items()}
            for name,arr in coverage_arrays.items():cov[name].append(arr)
            np.savez_compressed(out/f"{arm}_{row['state_id']}_{row['planner_seed']}_coverage.npz",**coverage_arrays)
        assert maxstate==maxreward==0.
        audits[arm]=dict(episodes=64,replayed_steps=step_count,state_error_max=maxstate,reward_error_max=maxreward,step_cost_components_exact=True,termination_exact=True)
        timings[arm]=dict(online_seconds=describe(all_times),online_total_seconds=float(sum(all_times)),
            initialization_seconds=describe([r['initialization_seconds'] for r in rows[arm]]),
            oracle_build_seconds=sum(r['controller_diagnostics'].get('oracle_value_build_time_sec',0) for r in rows[arm]))
        queries[arm]=dict(counts,values=describe(all_values),invalid_ratio=counts['invalid_queries']/max(1,counts['queries']),
            unreachable_ratio=counts['unreachable_queries']/max(1,counts['queries']),invalid_candidate_ratio=counts['invalid_candidates']/counts['candidate_count'])
        covers[arm]={name:{k:describe(np.concatenate(arrays)[:,i]) for i,k in enumerate(COVER_KEYS)} for name,arrays in cov.items()}
        print(json.dumps(dict(replayed=arm,episodes=64,steps=step_count,exact=True)),flush=True)
    groups={'all':lambda r:True,'inside':lambda r:r['stratum']=='inside','west':lambda r:r['stratum']=='west'}
    for h in range(8):groups[f'heading_{h}']=lambda r,h=h:r['heading_bin']==h
    for region in ('inside','west'):
        for h in range(8):groups[f'{region}_heading_{h}']=lambda r,region=region,h=h:r['stratum']==region and r['heading_bin']==h
    for seed in SEEDS:groups[f'seed_{seed}']=lambda r,seed=seed:r['planner_seed']==seed
    grouped={name:{arm:detailed_group([r for r in rows[arm] if pred(r)]) for arm in ARMS} for name,pred in groups.items()}
    comparisons={}
    for name,pred in groups.items():
        comparisons[name]={f'{a}_to_{b}':pair_details([r for r in rows[a] if pred(r)],[r for r in rows[b] if pred(r)]) for a,b in [('R0','R1'),('oracle','R0'),('oracle','R1'),('none','R0'),('none','R1')]}
    summary=dict(grouped=grouped,paired=comparisons,timings=timings,queries=queries,coverage=covers,
        oracle_validation=read(root/'oracle_validation.json'),start_overlap=read(root/'start_overlap.json'),adapter_validation=read(root/'adapter_validation.json'),
        audits=audits,pairing=dict(episodes=256,common_state_seed_keys=64,first_candidates_exact_across_arms=True,
            reset_and_noise_contract_tested=True,checkpoints_unchanged=True,formal_only=True),
        support=dict(count=len(support),definition='unique raw physical source/next poses; no abstract nodes',
            metric='nearest XY; joint nearest under sqrt(dx^2+dy^2+(0.5*wrapped_dtheta)^2); only coverage, not reachability'),
        manifest_sha256=file_digest(root/'manifest.json'),analysis_seconds=time.perf_counter()-started)
    write(out/'replay_audit.json',audits)
    finalize(root,summary,rows)


def finalize(root,summary,rows):
    out=root/'comparison'
    background=TRAIN/'comparison/summary.json'
    summary['graph_background_sha256']=file_digest(background)
    summary['assessment']=dict(success_gain_episodes=summary['paired']['all']['R0_to_R1']['improved'],
        success_loss_episodes=summary['paired']['all']['R0_to_R1']['worsened'],
        gained_start_ids=sorted({p[0] for p in summary['paired']['all']['R0_to_R1']['new_success']}),
        joint_success_cost_delta=summary['paired']['all']['R0_to_R1']['joint_success_cost_delta'],
        scope='Descriptive paired results; single model seed and 32 shared start poses, not 64 independent geographic draws')
    write(out/'summary.json',summary)
    from .report import report,plots
    report(root,summary,rows);plots(root,summary,rows)
    verify(root)
    for name in ('analyze.py','report.py','test_analysis.py'):
        src=Path(__file__).parent/name
        if src.exists():shutil.copy2(src,out/('source_'+name))
    artifact_hashes={str(f):file_digest(f) for f in root.rglob('*') if f.is_file() and 'source_snapshot' not in str(f) and f.name not in ('final_integrity.json','COMPLETE','analysis.log')}
    artifact_hashes[str(background)]=file_digest(background)
    write(root/'final_integrity.json',dict(passed=True,old_inputs_unchanged=True,all_actions_replayed_exact=True,
        pairing=summary['pairing'],artifacts=artifact_hashes))
    write(out/'COMPLETE',dict(passed=True,summary_sha256=file_digest(out/'summary.json')))
    write(root/'COMPLETE',dict(passed=True,episodes=256,final_integrity_sha256=file_digest(root/'final_integrity.json')))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);analyze(a.parse_args().root)
