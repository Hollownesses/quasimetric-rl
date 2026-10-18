"""Frozen H10/N128 evaluation. Historical baselines reused only on identical poses/RNG."""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.four_path_execution.common import (ROOT,SCENARIO,PLANNER_SEEDS,read,write,
    load_data,adapter,checkpoint,file_digest,TimedMPPIController)
from experiments.four_path_oracle_execution.protocol import CONFIG,ORACLE_CONFIG,controller_factory

OLD=ROOT/'results/four_path_oracle_execution_20261017/main'
ARMS=('none','oracle','merged_606','qp_101','qp_202','qp_303','qp_404','qp_505')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);data=load_data();env=data.env
    records=read(OLD/'states.json')['records'];rng=np.random.default_rng(2026101703);new=[]
    # Independent draws, no rejection using model scores or proximity to replay.
    # Stratify headings to retain east/north/south cases rather than mainly westward starts.
    for i in range(24):
        theta=-np.pi+(i%8+.5)*2*np.pi/8+rng.uniform(-.18,.18)
        pose=np.array([rng.uniform(3.05,4.95),rng.uniform(2.15,5.15),theta],np.float32)
        env.reset(seed=0,options={'device_id':'u_trap_target','start':pose})
        assert env.is_valid_state(pose) and not env.is_task_feasible(pose)
        assert all(np.linalg.norm(pose-np.array(r['pose']))>1e-6 for r in records+new)
        new.append(dict(id=f'independent_{i:02d}',stratum='independent',pose=pose.tolist(),prefix_stages={}))
    write(root/'independent_states.json',dict(seed=2026101703,records=new,rule='24 random U interior poses; eight heading strata; no score-based rejection'))
    paths=[*Path('experiments/full_replay_four_path').glob('*.py'),
        *Path('experiments/four_path_execution').glob('*.py'),*Path('experiments/four_path_oracle_execution').glob('*.py'),
        Path('experiments/four_path_factorial/run.py'),Path('minimal_qrl/global_push_sampling.py'),
        Path('quasimetric_rl/modules/quasimetric_critic/losses/phr.py'),
        *Path('quasimetric_rl/modules/quasimetric_critic/models').rglob('*.py'),
        *Path('minimal_qrl/baselines').glob('*.py'),Path('minimal_qrl/gc_agents.py'),
        Path('minimal_qrl/envs/comm_inspection_dubins_uav_2d.py'),Path('minimal_qrl/envs/dubins_uav_2d.py')]
    sources={}
    for path in paths:
        dest=root/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        sources[str(path)]=file_digest(path)
    write(root/'protocol.json',dict(config=asdict(CONFIG),oracle_config=asdict(ORACLE_CONFIG),sources=sources,
        historical_manifest_sha256=file_digest(OLD/'manifest.json'),historical_states_sha256=file_digest(OLD/'states.json'),
        original_states=records,independent_states=new,planner_seeds=PLANNER_SEEDS,
        training_seed=606,training_budget=dict(phr=20000,decay=5000,batch=256),
        selection='final fixed-budget checkpoint, no oracle/graph-based stopping',
        reuse='all seven historical baselines only for identical 32 original states and two planner seeds'))


def model_path(root,arm):
    return root/'train/final.pt' if arm=='merged_606' else checkpoint(arm)


def make_controller(root):
    old=controller_factory(root)
    def make(data,arm):
        if arm!='merged_606':return old(data,arm)
        adapted,_=adapter(data,model_path(root,arm))
        return TimedMPPIController(CONFIG,terminal_mode='qrl',qrl_agent=adapted)
    return make


def bind(root):
    protocol=read(root/'protocol.json')
    for path,sha in protocol['sources'].items():assert file_digest(path)==sha,path
    assert asdict(CONFIG)==protocol['config']==read(OLD/'manifest.json')['config']
    assert file_digest(OLD/'states.json')==protocol['historical_states_sha256']
    assert (root/'train/COMPLETE').exists()
    for group,records,arms in [('original',protocol['original_states'],['merged_606']),('independent',protocol['independent_states'],ARMS)]:
        target=root/group;target.mkdir(exist_ok=False)
        write(target/'states.json',dict(records=records))
        write(target/'manifest.json',dict(config=asdict(CONFIG),sources=protocol['sources'],
            states_sha256=file_digest(target/'states.json'),arms=arms,planner_seeds=PLANNER_SEEDS,
            checkpoints={a:dict(path=str(model_path(root,a)),sha256=file_digest(model_path(root,a))) for a in arms if a not in ('none','oracle')}))
    write(root/'baseline_reuse.json',{a:dict(episodes_sha256=file_digest(OLD/'arms'/a/'episodes.jsonl'),
        complete_sha256=file_digest(OLD/'arms'/a/'COMPLETE'),checkpoint=read(OLD/'manifest.json')['checkpoints'].get(a)) for a in ARMS if a!='merged_606'})


def evaluate(root,group,arm):
    import experiments.four_path_execution.run as runner
    runner.checkpoint=lambda a:model_path(root,a)
    runner.evaluate_arm(root/group,arm,config=CONFIG,controller_factory=make_controller(root))


def candidates(root):
    from experiments.four_path_execution.candidates import diagnostic
    # Same 16 development candidate banks as prior experiment, all models including the new one.
    diagnostic(root/'independent',config=CONFIG,arms=ARMS,controller_factory=make_controller(root),
        comparison_pairs=[('oracle',a) for a in ARMS if a!='oracle'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','bind','arm','candidates'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--group',default='independent');p.add_argument('--arm')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='bind':bind(a.root)
    elif a.command=='candidates':candidates(a.root)
    else:evaluate(a.root,a.group,a.arm)
