"""Generate evaluation poses from frozen geometry, without any model scores."""
import argparse
from dataclasses import asdict
import shutil
import subprocess

import numpy as np
import torch

from experiments.four_path_execution.common import *
from minimal_qrl.path_support import load_bank,digest


def build_states(data):
    bank=load_bank(BANK);env=data.env;env.set_task_by_device_id('u_trap_target')
    records=[]
    for index,path in enumerate(bank['records']):
        states=np.asarray(path['states'],dtype=np.float32)
        west=np.flatnonzero((states[:,0]<=2.8)&(states[:,1]>=1.9)&(states[:,1]<=5.4))
        east=np.flatnonzero(states[:,0]>5.7)
        assert len(west) and len(east)
        for label,j in [('start',0),('west_exit',int(west[0])),('east_wall',int(east[0]))]:
            prefix=stage_metrics(states[:j+1],{})
            state=states[j]
            assert env.is_valid_state(state) and not env.is_task_feasible(state)
            records.append(dict(id=f'path{index}_{label}',stratum='support',pose=state.tolist(),
                path=index,path_index=j,prefix_stages=dict(exit=prefix['west_exit_visited'],wrap=prefix['observed_wrap'])))
    rng=np.random.default_rng(2026101701)
    for record in list(records):
        for _ in range(10000):
            offset=rng.uniform([-.15,-.15,-.35],[.15,.15,.35]);pose=np.asarray(record['pose'])+offset
            pose[2]=(pose[2]+np.pi)%(2*np.pi)-np.pi;pose=pose.astype(np.float32)
            if env.is_valid_state(pose) and not env.is_task_feasible(pose):break
        else:raise RuntimeError('perturbation sampling exhausted')
        records.append(dict(id=record['id']+'_perturbed',stratum='perturbed',pose=pose.tolist(),
            source_id=record['id'],offset=offset.tolist(),prefix_stages={}))
    exclusions=np.concatenate([np.asarray(p['states']) for p in bank['records']]+
        [np.asarray(bank['excluded_starts']),np.asarray(bank['candidate_starts'])],axis=0)
    rng=np.random.default_rng(2026101702)
    for i in range(8):
        for attempt in range(100000):
            pose=rng.uniform([3.1,2.2,-np.pi],[4.9,5.1,np.pi]).astype(np.float32)
            distance=float(np.linalg.norm(exclusions[:,:2]-pose[:2],axis=1).min())
            previous=[r['pose'] for r in records if r['stratum']=='holdout']
            separated=not previous or np.linalg.norm(np.asarray(previous)[:,:2]-pose[:2],axis=1).min()>.25
            if distance>.25 and separated and env.is_valid_state(pose) and not env.is_task_feasible(pose):break
        else:raise RuntimeError('holdout rejection sampling exhausted')
        records.append(dict(id=f'holdout_{i:02d}',stratum='holdout',pose=pose.tolist(),
            exclusion_distance=distance,attempts=attempt+1,prefix_stages={}))
    assert len(records)==32
    return dict(records=records,digest=digest(records),scenario_sha256=file_digest(SCENARIO),
        training_bank_digest=bank['content_digest'],planner_seeds=PLANNER_SEEDS)


def prepare(output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);data=load_data()
    validations={arm:validate_adapter(data,arm) for arm in ARMS if arm not in ('none','geometry')}
    write(output/'adapter_validation.json',validations)
    bank=build_states(data);write(output/'states.json',bank)
    sources={}
    paths=list((ROOT/'experiments/four_path_execution').glob('*'))+[ROOT/'minimal_qrl/baselines/mppi.py',ROOT/'minimal_qrl/baselines/base.py',ROOT/'minimal_qrl/gc_agents.py',ROOT/'minimal_qrl/envs/comm_inspection_dubins_uav_2d.py',ROOT/'minimal_qrl/envs/dubins_uav_2d.py']
    for path in paths:
        if not path.is_file():continue
        relative=path.relative_to(ROOT);target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);sources[str(relative)]=file_digest(path)
    shutil.copy2(SCENARIO,output/'scenario.json')
    write(output/'manifest.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=sources,config=asdict(CONFIG),states_sha256=file_digest(output/'states.json'),
        checkpoints={arm:dict(path=v['checkpoint'],sha256=v['sha256']) for arm,v in validations.items()},
        arms=ARMS,planner_seeds=PLANNER_SEEDS,episodes_per_arm=64,total_episodes=768,max_steps=300,
        data_fingerprints={k:v for k,v in data.fingerprint.items() if k!='initial_checkpoint'},
        device='cpu',torch_threads=1))
    print(json.dumps(dict(output=str(output),poses=len(bank['records']),max_adapter_error=max(v['max_absolute_error'] for v in validations.values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    prepare(Path(p.parse_args().output))
