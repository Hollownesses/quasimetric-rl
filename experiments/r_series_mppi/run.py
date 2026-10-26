"""Prepare, smoke, freeze, and execute the new 4 x 32 x 2 evaluation matrix."""
import argparse
import platform
import shutil
import subprocess
import sys
import time
from .common import *
from minimal_qrl.baselines.hybrid_astar_value import _signature_payload, _digest
from minimal_qrl.baselines.base import rollout_controller_episode
from minimal_qrl.eval.task_path_support_audit import physical_episodes

CORE=('minimal_qrl/baselines/mppi.py','minimal_qrl/baselines/hybrid_astar_value.py',
      'minimal_qrl/baselines/hybrid_astar.py','minimal_qrl/baselines/base.py',
      'minimal_qrl/envs/comm_inspection_dubins_uav_2d.py','minimal_qrl/envs/dubins_uav_2d.py',
      'minimal_qrl/gc_agents.py','experiments/four_path_oracle_execution/protocol.py')

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);data=load();env=data.env
    core={}
    for f in CORE:
        old=subprocess.check_output(['git','show',HISTORICAL+':'+f])
        assert Path(f).read_bytes()==old,('historical source differs',f)
        core[f]=file_digest(f)
    previous=read(OLD/'manifest.json')
    assert json.loads(json.dumps(asdict(CONFIG)))==previous['config']
    assert json.loads(json.dumps(asdict(ORACLE_CONFIG)))==previous['oracle_config']
    checkpoints={};inputs={}
    for arm in ('R0','R1'):
        directory=TRAIN/arm;ck=torch.load(model_path(arm),weights_only=True);cfg=read(directory/'config.json')
        assert ck['step']==50000 and ck['initialization_seed']==606 and ck['fingerprints']==data.fingerprint
        assert ck['task_mode']=='all' and ck['task_weight_mode']=={'R0':'record','R1':'source'}[arm]
        done=read(directory/'COMPLETE');assert done['final_sha256']==file_digest(model_path(arm))
        checkpoints[arm]=dict(path=str(model_path(arm)),sha256=file_digest(model_path(arm)),step=50000,seed=606)
        for name in ('initial.pt','final.pt','config.json','COMPLETE'):inputs[str(directory/name)]=file_digest(directory/name)
    from experiments.task_source_balance.train import equal
    a=torch.load(TRAIN/'R0/initial.pt',weights_only=True);b=torch.load(TRAIN/'R1/initial.pt',weights_only=True)
    for field in ('model','optimizer','dual','sampler','rng','visits'):assert equal(a[field],b[field]),field
    for f in ('verified_replay.pt','scenario.json','manifest.json'):inputs[str(DATA/f)]=file_digest(DATA/f)
    states=build_states(env);write(root/'states.json',states)
    episodes,_,_=physical_episodes(data.raw)
    trainposes=poses(data.raw.all_observations[[e['obs_ids'][0] for e in episodes]].numpy())
    oldfiles=[OLD/'states.json',Path('results/full_replay_four_path_20261017/main/independent_states.json')]
    oldposes=[]
    for p in oldfiles:
        item=read(p);oldposes.extend(r['pose'] for r in item['records']);inputs[str(p)]=file_digest(p)
    probes=read(DATA/'scenario.json')['metadata']['u_trap_local_navigability_probes']
    oldposes.extend([x['x'],probes['centerline_y'],h['theta']] for x in probes['positions'] for h in probes['headings'])
    new=np.array([r['pose'] for r in states['records']]);overlap={}
    for label,support in [('training_starts',trainposes),('old_test_and_local_probes',np.array(oldposes))]:
        delta=new[:,None,:]-support[None];angle=abs((delta[:,:,2]+np.pi)%(2*np.pi)-np.pi)
        pos=np.linalg.norm(delta[:,:,:2],axis=2);exact=(pos<1e-6)&(angle<1e-6)
        overlap[label]=dict(count=len(support),exact_pose_overlaps=int(exact.any(axis=1).sum()),
            min_position_distance=float(pos.min()),min_joint_pose_distance=float(np.sqrt(pos**2+angle**2).min()))
    assert all(x['exact_pose_overlaps']==0 for x in overlap.values())
    write(root/'start_overlap.json',overlap)
    # Check every environment/task/cost/grid signature field, not merely a filename.
    env.reset(seed=0,options={'device_id':'u_trap_target','start':[4.,3.65,0.]})
    payload=_signature_payload(env,ORACLE_CONFIG);digest=_digest(payload)
    with np.load(OLD_CACHE,allow_pickle=False) as cached:
        cached_digest=str(cached['digest'].item());cached_meta=json.loads(str(cached['diagnostics'].item()))
    cache_dir=root/'oracle_cache';cache_dir.mkdir()
    reused=digest==cached_digest
    if reused:shutil.copy2(OLD_CACHE,cache_dir/OLD_CACHE.name)
    t=time.perf_counter();oracle=HybridAStarValueOracle(ORACLE_CONFIG,cache_dir=cache_dir);diag=oracle.begin_episode(env,seed=0);elapsed=time.perf_counter()-t
    cache=oracle._cache_path(digest)
    assert diag['oracle_value_table_digest']==digest and (not reused or diag['oracle_value_cache_hit'])
    write(root/'oracle_validation.json',dict(reused=reused,old_cache=str(OLD_CACHE),old_cache_sha256=file_digest(OLD_CACHE),
        old_digest=cached_digest,current_digest=digest,signature_payload=payload,cache_path=str(cache),cache_sha256=file_digest(cache),
        current_setup_seconds=elapsed,current_build_seconds=diag['oracle_value_build_time_sec'],historical_build_metadata=cached_meta,
        diagnostics=diag,source_exact_to=HISTORICAL))
    inputs[str(OLD_CACHE)]=file_digest(OLD_CACHE)
    write(root/'adapter_validation.json',{arm:validate_model(data,arm) for arm in ('R0','R1')})
    write(root/'draft.json',dict(historical_commit=HISTORICAL,historical_core=core,config=asdict(CONFIG),oracle_config=asdict(ORACLE_CONFIG),
        arms=ARMS,planner_seeds=SEEDS,max_steps=300,episodes_per_arm=64,total_episodes=256,training_data=str(DATA),fingerprints=data.fingerprint,
        environment_signature=read(DATA/'manifest.json')['environment_signature'],oracle_signature=payload,
        checkpoints=checkpoints,immutable_inputs=inputs,states_sha256=file_digest(root/'states.json'),
        task_id='u_trap_target',torch_threads_per_worker=1,workers=4,python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        sampling='seed+7001 Gaussian; zero nominal reset; candidate zero nominal; shifted weighted-mean warm start; clipped [-3,3]',
        support_metric='posthoc only; nearest XY and heading at nearest XY; joint pose radius 0.5 environment-unit/radian',
        evaluation_only=True,model_selection='seed606 fixed50k; no retraining/calibration',oracle_used_for_start_selection=False))
    print(json.dumps(dict(prepared=True,cache_reused=reused,cache_digest=digest,overlap=overlap)),flush=True)


def store_rollout(out,arm,record,seed,rollout,control):
    states=np.asarray(rollout['states']);actions=np.asarray(rollout['actions']);rewards=np.asarray(rollout['rewards'])
    terms=np.array([[info[k] for k in COST_KEYS] for info in rollout['step_infos']])
    np.testing.assert_allclose(terms.sum(axis=1),-rewards,rtol=0,atol=1e-12)
    diag=rollout['action_diagnostics'];cd=rollout['controller_diagnostics'];n=len(actions)
    assert n<=300 and sum(d['model_rollouts'] for d in diag)==n*128
    arr=control.arrays();assert len(arr['query_counts'])==n
    assert arr['query_counts'].sum()==len(arr['query_values'])
    row=dict(arm=arm,state_id=record['id'],stratum=record['stratum'],heading_bin=record['heading_bin'],planner_seed=seed,pose=record['pose'],
        success=rollout['success'],collision=rollout['collision'],out_of_bounds=rollout['out_of_bounds'],timeout=rollout['truncated'] and not rollout['success'],
        steps=n,cost=float(-rewards.sum()),cost_components=dict(zip(COST_KEYS,terms.sum(axis=0).tolist())),
        **stage_metrics(states,record),planning_seconds=sum(d['planning_time_sec'] for d in diag),
        initialization_seconds=cd['initial_planning_time_sec'],controller_diagnostics=cd,
        model_rollouts=sum(d['model_rollouts'] for d in diag),model_transitions=sum(d['model_transitions'] for d in diag),
        terminal_queries=len(arr['query_values']),terminal_invalid_queries=int(arr['query_invalid'].sum()),
        terminal_unreachable_queries=int(arr['query_unreachable'].sum()),first_candidate_sha256=control.banks[0])
    assert sum(row[k] for k in ('success','collision','out_of_bounds','timeout'))==1
    np.savez_compressed(out/f"{record['id']}_{seed}.npz",states=states,actions=actions,rewards=rewards,cost_terms=terms,
        planning_seconds=[d['planning_time_sec'] for d in diag],invalid_candidate_fraction=[d['invalid_rollout_ratio'] for d in diag],**arr)
    # Full actual step information and terminal-query diagnostics retained per episode.
    write(out/f"{record['id']}_{seed}.json",dict(summary=row,step_infos=rollout['step_infos'],action_diagnostics=diag))
    return row


def smoke(root):
    data=load();env=data.env;out=root/'smoke';out.mkdir(exist_ok=False);records=[]
    # Separate development poses, never part of the frozen formal 32-state bank.
    probes=[dict(id='dev_inside',stratum='inside',heading_bin=4,pose=[4.0,3.65,.31],prefix_stages={}),
            dict(id='dev_terminal_near',stratum='development',heading_bin=7,pose=[7.8,3.8,2.95],prefix_stages={})]
    for arm in ARMS:
        for probe in probes:
            first=None
            for repeat in range(2):
                c=make(data,arm,root/'oracle_cache');r=rollout_controller_episode(c,env,episode_seed=SEEDS[0],device_id='u_trap_target',start_state=np.array(probe['pose'],np.float32))
                if r['success']:assert env.is_task_feasible(env.state)
                if first is None:
                    first=r;row=store_rollout(out,arm,dict(probe,id=arm+'_'+probe['id']),SEEDS[0],r,c);records.append(row)
                else:
                    for key in ('states','actions','rewards'):np.testing.assert_array_equal(np.asarray(first[key]),np.asarray(r[key]))
            print(json.dumps(dict(smoke=arm,pose=probe['id'],repeat_exact=True)),flush=True)
    write(root/'smoke.json',dict(passed=True,repeat_exact=True,episodes_executed=16,unique_smoke_cases=8,formal_starts_used=False,rows=records))


def freeze(root,test_log):
    assert read(root/'smoke.json')['passed'];p=read(root/'draft.json')
    assert 'passed' in Path(test_log).read_text() and 'failed' not in Path(test_log).read_text()
    for f,h in p['immutable_inputs'].items():assert file_digest(f)==h,f
    assert file_digest(root/'states.json')==p['states_sha256']
    paths=set()
    for folder in ('minimal_qrl','quasimetric_rl','experiments/r_series_mppi','experiments/four_path_execution',
                   'experiments/four_path_oracle_execution','experiments/four_path_factorial','experiments/expert_utrap_network',
                   'experiments/connected_expert_utrap','experiments/full_replay_four_path','experiments/task_source_balance'):
        paths.update(Path(folder).rglob('*.py'))
    paths.add(Path('experiments/r_series_mppi/PROTOCOL.md'))
    source={}
    for f in sorted(paths):
        dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);source[str(f)]=file_digest(f)
    # External IQE package is also content-addressed without modifying it.
    import torchqmet
    external={str(f):file_digest(f) for f in Path(torchqmet.__file__).parent.rglob('*.py')}
    p.update(sources=source,external_sources=external,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tests_sha256=file_digest(test_log),test_log=str(test_log),smoke_sha256=file_digest(root/'smoke.json'),
        oracle_validation_sha256=file_digest(root/'oracle_validation.json'),adapter_validation_sha256=file_digest(root/'adapter_validation.json'))
    write(root/'manifest.json',p);write(root/'FROZEN',dict(manifest_sha256=file_digest(root/'manifest.json'),passed=True))
    print('Protocol frozen; ready for 256 episodes',flush=True)


def verify(root):
    p=read(root/'manifest.json');assert read(root/'FROZEN')['manifest_sha256']==file_digest(root/'manifest.json')
    assert asdict(CONFIG)==p['config'] and file_digest(root/'states.json')==p['states_sha256']
    for group in ('sources','external_sources','immutable_inputs'):
        for f,h in p[group].items():assert file_digest(f)==h,f
    ov=read(root/'oracle_validation.json');assert file_digest(ov['cache_path'])==ov['cache_sha256']
    return p


def arm(root,name):
    p=verify(root);assert name in ARMS;data=load();assert data.fingerprint==p['fingerprints']
    out=root/'arms'/name;out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    with (out/'episodes.jsonl').open('w') as stream:
        for record in read(root/'states.json')['records']:
            for seed in SEEDS:
                # Fresh planner object and zero nominal every episode, shared model checkpoint only.
                c=make(data,name,root/'oracle_cache')
                before=tensor_digest(*c.qrl_agent.state_dict().values()) if c.qrl_agent else None
                r=rollout_controller_episode(c,data.env,episode_seed=seed,device_id='u_trap_target',start_state=np.array(record['pose'],np.float32))
                np.testing.assert_array_equal(r['states'][0],np.array(record['pose'],np.float32))
                if r['success']:assert data.env.is_task_feasible(data.env.state) and data.env.is_valid_state(data.env.state)
                if before:assert before==tensor_digest(*c.qrl_agent.state_dict().values())
                row=store_rollout(out,name,record,seed,r,c);stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(arm=name,state=record['id'],seed=seed,success=row['success'],steps=row['steps'],elapsed=time.perf_counter()-started)),flush=True)
    verify(root)
    write(out/'COMPLETE',dict(episodes=64,elapsed_seconds=time.perf_counter()-started,parameters_unchanged=True,
        episodes_sha256=file_digest(out/'episodes.jsonl')))


def suite(root):
    verify(root);processes=[]
    for name in ARMS:
        log=(root/(name+'.log')).open('x');process=subprocess.Popen([sys.executable,'-m','experiments.r_series_mppi.run','arm','--root',str(root),'--arm',name],stdout=log,stderr=subprocess.STDOUT)
        processes.append((name,process,log))
    statuses={}
    for name,process,log in processes:statuses[name]=process.wait();log.close()
    write(root/'process_results.json',statuses)
    assert all(v==0 for v in statuses.values()),statuses

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','smoke','freeze','arm','suite']);a.add_argument('--root',type=Path,required=True);a.add_argument('--arm',choices=ARMS);a.add_argument('--test-log',type=Path)
    x=a.parse_args()
    if x.command=='prepare':prepare(x.root)
    elif x.command=='smoke':smoke(x.root)
    elif x.command=='freeze':freeze(x.root,x.test_log)
    elif x.command=='arm':arm(x.root,x.arm)
    else:suite(x.root)
