"""Forward-only reset curriculum; random controls; no expert/Oracle/graph policy."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import shutil
import numpy as np
import torch
from quasimetric_rl.data import MultiEpisodeData
from minimal_qrl.dataset import QRLExploreConfig,collect_qrl_explore_episode_pair
from minimal_qrl.global_push_sampling import replay_fingerprint
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from .common import ROOT,SCENARIO,DEVICE,FIELDS,sha,write,make_env

# Geometry bands use the U opening/walls and minimum turn radius v/omega_max,
# not a Euclidean-distance ranking. Centers describe reset headings, NOT actions.
STAGES=[
 dict(name='near_terminal',boxes=[[7.52,3.35,7.92,3.95]],headings=[math.pi],half_width=.25,
      rationale='east legal approach boundary, nearly west-facing; nonterminal filter'),
 dict(name='approach',boxes=[[7.45,3.10,8.60,4.20]],headings=[math.pi],half_width=.70,
      rationale='larger legal approach neighborhood and heading error'),
 dict(name='east_turning',boxes=[[7.60,2.10,9.00,5.20]],headings=[math.pi],half_width=1.40,
      rationale='east-side room for turns; broad vertical offset and heading error'),
 dict(name='east_corners',boxes=[[5.95,6.05,8.60,6.90],[5.95,.40,8.60,1.25]],
      headings=[-math.pi/2,math.pi/2],half_width=2.0,
      rationale='separate upper/lower corners require turning around wall envelope'),
 dict(name='outer_corridors',boxes=[[1.30,6.05,5.10,7.00],[1.30,.30,5.10,1.25]],
      headings=[0.,0.],half_width=math.pi,
      rationale='outside upper/lower U arms; eastward progress requires long travel'),
 dict(name='west_exit',boxes=[[1.,2.,2.70,5.20]],headings=[0.],half_width=math.pi,
      rationale='west opening: direct eastward line hits closed wall; must select a detour'),
 dict(name='u_interior',boxes=[[3.,2.10,5.,5.20]],headings=[0.],half_width=math.pi,
      rationale='U interior with full heading support: escape west before detouring')]


def default_config():
    return dict(schema_version=1,seed=2026102007,curriculum_step_budget=30000,
        diagnostic_steps_per_stage=1000,diagnostic_stages=list(range(1,7)),
        max_rounds=24,max_stagnant_rounds=6,recovery_after=3,
        normal_frontier_episodes=24,normal_revisit_episodes=8,
        recovery_frontier_episodes=16,recovery_revisit_episodes=16,
        pass_success_rate=.125,pass_min_successes=2,pass_min_heading_bins=2,
        pass_position_heading_bins=8,required_consecutive_passes=2,
        coverage_position_resolution=.25,coverage_heading_bins=24,
        exclusion_radius=.25,start_attempt_limit=10000,stages=STAGES,
        policy=dict(action_hold_min_steps=3,action_hold_max_steps=10,
                    straight_action_probability=.5,local_safety_lookahead_steps=0,
                    safety_lookahead_mode='full_horizon'),
        policy_information='goal-blind persistent random actions; no lookahead or action ranking',
        reset_information='goal region, obstacle geometry, turn radius, previous collection success/coverage',
        diagnostic_policy='after adaptive phase, fixed 1000 real steps per stage 1..6; no promotion from these data',
        rationale='36k upper bound is 18% of old 200k; 30k adaptation plus fixed 6k failure/coverage survey; no budget extension')


def exclusions(scenario):
    files=[ROOT/'results/four_path_execution_20261017/main/states.json',
           ROOT/'results/full_replay_four_path_20261017/main/independent_states.json',
           ROOT/'results/diagnostic_u_shadow_corridors_topology_v2/config/diagnostic_task_bank.json']
    poses=[]
    # Read only frozen pose fields for exclusion; never actions, values, outcomes.
    for path in files[:2]:
        poses.extend(r['pose'] for r in json.loads(path.read_text())['records'])
    xmin,ymin,xmax,ymax=scenario['bounds']
    for r in json.loads(files[2].read_text())['records']:
        if r['split']=='test' and r['device_id']==DEVICE:
            x,y,h=r['start_normalized'];poses.append([xmin+x*(xmax-xmin),ymin+y*(ymax-ymin),h])
    probes=scenario['metadata']['u_trap_local_navigability_probes']
    poses.extend([p['x'],probes['centerline_y'],h['theta']] for p in probes['positions'] for h in probes['headings'])
    return np.asarray(poses,np.float32),{str(p.relative_to(ROOT)):sha(p) for p in files}


def coverage_key(state,cfg):
    x,y,h=map(float,state)
    return (int(np.floor(x/cfg['coverage_position_resolution'])),
            int(np.floor(y/cfg['coverage_position_resolution'])),
            int(np.floor(((h+np.pi)%(2*np.pi))/(2*np.pi)*cfg['coverage_heading_bins'])))


def sample_start(env,stage,rng,excluded,cfg,seen):
    spec=cfg['stages'][stage];reject=Counter()
    for attempt in range(cfg['start_attempt_limit']):
        box=int(rng.integers(len(spec['boxes'])));x0,y0,x1,y1=spec['boxes'][box]
        heading=spec['headings'][box]+rng.uniform(-spec['half_width'],spec['half_width'])
        state=np.array([rng.uniform(x0,x1),rng.uniform(y0,y1),(heading+np.pi)%(2*np.pi)-np.pi],np.float32)
        if not env.is_valid_state(state):reject['invalid']+=1;continue
        if env.is_task_feasible(state):reject['already_terminal']+=1;continue
        if np.any(np.linalg.norm(excluded[:,:2]-state[:2],axis=1)<=cfg['exclusion_radius']):
            reject['heldout_proximity']+=1;continue
        if state.tobytes() in seen:reject['duplicate']+=1;continue
        seen.add(state.tobytes())
        return state,dict(reject)
    raise RuntimeError(f'No admissible fresh start in stage {stage}; do not relax exclusions')


def batch_metrics(records,cfg):
    n=len(records);success=sum(r['outcome']=='success' for r in records)
    cells={coverage_key(r['start'],cfg) for r in records}
    return dict(episodes=n,successes=success,success_rate=success/max(1,n),
        position_heading_bins=len(cells),heading_bins=len({c[2] for c in cells}),
        passed=bool(n>=16 and success>=cfg['pass_min_successes'] and
            success/n>=cfg['pass_success_rate'] and
            len(cells)>=min(cfg['pass_position_heading_bins'],n//2) and
            len({c[2] for c in cells})>=cfg['pass_min_heading_bins']))


def collect_one(env,start,seed,context,limit,cfg,stage,batch,phase,role):
    policy=QRLExploreConfig(**cfg['policy'])
    episode,anchor,diag=collect_qrl_explore_episode_pair(env,start_state=start,start_index=context,
        max_steps=limit,seed=seed,context_id=context,device_id=DEVICE,config=policy,
        budget_cutoff=limit<env.max_episode_steps,include_global_push_pairs=False)
    # The historical collector computes optional suffix labels and produces an
    # unrelated sampled anchor on failures. Neither belongs in this experiment.
    episode.transition_infos['goal_return_cost'].zero_()
    episode.transition_infos['goal_return_mask'].zero_()
    success=diag['outcome']=='success'
    if not success:anchor=None
    result=[episode]+([anchor] if anchor is not None else [])
    for ep in result:
        for name,value in [('curriculum_stage',stage),('curriculum_batch',batch),
                           ('curriculum_phase',phase),('curriculum_role',role)]:
            ep.transition_infos[name]=torch.full((ep.num_transitions,),value,dtype=torch.long)
    record=dict(context=context,seed=seed,start=np.asarray(start,np.float32).tolist(),
        steps=episode.num_transitions,outcome=diag['outcome'],stage=stage,batch=batch,phase=phase,role=role,
        loop_detected=diag['loop_detected'],action_segments=diag['action_segments'],
        safety_resampled_segments=diag['safety_resampled_segments'],
        terminal_state=np.asarray(diag['raw_states'][-1],np.float32).tolist())
    assert success==(anchor is not None)
    return result,record,np.asarray(diag['raw_states'],np.float32)


def collect(output,config):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scenario=json.loads(SCENARIO.read_text());env=make_env(scenario)
    excluded,exclusion_hashes=exclusions(scenario)
    write(output/'config.json',config);write(output/'scenario.json',scenario)
    write(output/'exclusions.json',dict(states=excluded.tolist(),source_hashes=exclusion_hashes,
        position_radius=config['exclusion_radius'],applies_to='resets only, independent of heading; rollout proximity audited separately'))
    # Freeze collection inputs BEFORE any env.step. No results-based protocol edits.
    dependencies=[SCENARIO,ROOT/'minimal_qrl/dataset.py',
        ROOT/'minimal_qrl/envs/comm_inspection_dubins_uav_2d.py',ROOT/'minimal_qrl/envs/dubins_uav_2d.py',
        ROOT/'minimal_qrl/envs/base.py',ROOT/'minimal_qrl/global_push_sampling.py',
        ROOT/'quasimetric_rl/modules/quasimetric_critic/models/encoder.py',ROOT/'minimal_qrl/baselines/mppi.py',
        ROOT/'minimal_qrl/industry_exp/scalability_scenarios.py']+list(Path(__file__).parent.glob('*.py'))
    hashes={}
    for source in dependencies:
        if not source.exists():continue
        relative=source.relative_to(ROOT);target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);hashes[str(relative)]=sha(source)
    write(output/'protocol.json',dict(config_sha256=sha(output/'config.json'),sources=hashes,
        exclusion_sources=exclusion_hashes,environment_signature=environment_signature(env),
        min_turn_radius=env.v/env.omega_max,python=platform.python_version(),numpy=np.__version__,
        torch=str(torch.__version__),expert_actions=False,oracle_values=False,graph_guidance=False))
    episodes=[];records=[];states=[];decisions=[];seen=set();rejections=Counter()
    schedule=np.random.default_rng(np.random.SeedSequence([config['seed'],11]))
    starts=np.random.default_rng(np.random.SeedSequence([config['seed'],29]))
    frontier=0;mastered=[];streak=0;stagnant=0;main_steps=0;stop_reason='max_rounds'
    def take(stage,batch,phase,role,remaining):
        start,reject=sample_start(env,stage,starts,excluded,config,seen);rejections.update(reject)
        context=len(records)
        seed=int(np.random.SeedSequence([config['seed'],context,104729]).generate_state(1)[0])
        eps,record,ss=collect_one(env,start,seed,context,min(env.max_episode_steps,remaining),
                                 config,stage,batch,phase,role)
        episodes.extend(eps);records.append(record);states.append(ss)
        with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return record
    for batch in range(config['max_rounds']):
        recovery=stagnant>=config['recovery_after']
        nf=config['recovery_frontier_episodes'] if recovery else config['normal_frontier_episodes']
        nr=config['recovery_revisit_episodes'] if recovery else config['normal_revisit_episodes']
        # Promotion evaluated from fresh FRONTIER episodes only, never revisits.
        frontier_records=[]
        for role,count in [(0,nf),(1,nr)]:
            for _ in range(count):
                remaining=config['curriculum_step_budget']-main_steps
                if remaining<=0:break
                stage=frontier if role==0 or not mastered else int(schedule.choice(mastered))
                record=take(stage,batch,0,role,remaining);main_steps+=record['steps']
                if role==0:frontier_records.append(record)
        metrics=batch_metrics(frontier_records,config)
        complete=len(frontier_records)==nf
        streak=streak+1 if complete and metrics['passed'] else 0
        stagnant+=1;old=frontier;promoted=streak>=config['required_consecutive_passes']
        if promoted:
            mastered.append(frontier);frontier+=1;streak=0;stagnant=0
        decision=dict(batch=batch,frontier=old,recovery=recovery,frontier_metrics=metrics,
            complete_frontier_batch=complete,promoted=promoted,mastered=mastered.copy(),
            consecutive_passes=streak,rounds_since_promotion=stagnant,main_steps=main_steps)
        decisions.append(decision);write(output/'decisions.json',decisions)
        print(json.dumps(decision),flush=True)
        if main_steps>=config['curriculum_step_budget']:stop_reason='curriculum_step_budget';break
        if frontier==len(config['stages']):stop_reason='all_stages_mastered';break
        if stagnant>=config['max_stagnant_rounds']:stop_reason='stagnation';break
    # Fixed coverage survey is explicitly NOT an expansion/mastery declaration.
    survey_steps=0
    for stage in config['diagnostic_stages']:
        consumed=0
        while consumed<config['diagnostic_steps_per_stage']:
            record=take(stage,len(decisions)+stage,1,2,config['diagnostic_steps_per_stage']-consumed)
            consumed+=record['steps'];survey_steps+=record['steps']
    raw=MultiEpisodeData.cat(episodes)
    torch.save(dict(schema_version=1,sha256=replay_fingerprint(raw),
        raw={k:getattr(raw,k) for k in FIELDS}),output/'verified_replay.pt')
    np.savez_compressed(output/'physical_states.npz',states=np.concatenate(states),
        lengths=np.array([len(s) for s in states],np.int64))
    summary=dict(stop_reason=stop_reason,mastered_stages=mastered,frontier_at_stop=frontier,
        curriculum_steps=main_steps,diagnostic_steps=survey_steps,total_real_steps=main_steps+survey_steps,
        total_step_cap=config['curriculum_step_budget']+len(config['diagnostic_stages'])*config['diagnostic_steps_per_stage'],
        unused_curriculum_budget=config['curriculum_step_budget']-main_steps,
        episodes=len(records),outcomes=dict(Counter(r['outcome'] for r in records)),
        reset_rejections=dict(rejections),reset_states_unique=len(seen),
        all_failures_retained=True,anchors_only_from_actual_success=True,training_started=False)
    write(output/'collection_summary.json',summary)
    write(output/'manifest.json',dict(schema_version=1,dataset_kind='curriculum_network',
        replay_sha256=replay_fingerprint(raw),files={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        environment_signature=environment_signature(env),collection_source_hashes=hashes,
        expert_actions=False,oracle_or_shortest_path_labels=False))
    (output/'COLLECTION_COMPLETE').write_text('Collection complete; awaiting independent audit.\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',help='frozen JSON config; defaults to the declared 36k-cap protocol')
    a=p.parse_args();torch.set_num_threads(1)
    collect(a.output,json.loads(Path(a.config).read_text()) if a.config else default_config())
