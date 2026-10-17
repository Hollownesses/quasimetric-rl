"""Prepare a fresh frozen tree, execute all arms, or score common candidates."""
import argparse
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from experiments.four_path_execution.common import (
    ROOT, SCENARIO, PLANNER_SEEDS, Path, torch, read, write, load_data,
    file_digest, validate_adapter,
)
from experiments.four_path_oracle_execution.protocol import (
    ARMS, CONFIG, ORACLE_CONFIG, HISTORICAL_COMMIT, PREVIOUS_ROOT, controller_factory,
)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    previous=read(PREVIOUS_ROOT/'manifest.json');data=load_data()
    fingerprints={k:v for k,v in data.fingerprint.items() if k!='initial_checkpoint'}
    assert fingerprints==previous['data_fingerprints']
    assert file_digest(PREVIOUS_ROOT/'states.json')==previous['states_sha256']
    validations={a:validate_adapter(data,a) for a in ARMS if a.startswith('qp_')}
    for arm,v in validations.items():assert v['sha256']==previous['checkpoints'][arm]['sha256']
    write(root/'adapter_validation.json',validations)
    shutil.copy2(PREVIOUS_ROOT/'states.json',root/'states.json');shutil.copy2(SCENARIO,root/'scenario.json')
    paths=[*Path('experiments/four_path_execution').glob('*.py'),
           *Path('experiments/four_path_oracle_execution').glob('*.py'),
           Path('experiments/four_path_oracle_execution/PROTOCOL.md'),
           Path('experiments/four_path_factorial/run.py'),
           *Path('minimal_qrl/baselines').glob('*.py'),Path('minimal_qrl/gc_agents.py'),
           Path('minimal_qrl/envs/comm_inspection_dubins_uav_2d.py'),Path('minimal_qrl/envs/dubins_uav_2d.py')]
    sources={}
    for path in paths:
        target=root/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/path,target);sources[str(path)]=file_digest(ROOT/path)
    write(root/'manifest.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        historical_oracle_commit=HISTORICAL_COMMIT,previous_manifest_sha256=file_digest(PREVIOUS_ROOT/'manifest.json'),
        config=asdict(CONFIG),oracle_config=asdict(ORACLE_CONFIG),oracle_unreachable_cost=1e6,
        states_sha256=file_digest(root/'states.json'),sources=sources,data_fingerprints=fingerprints,
        checkpoints={a:dict(path=v['checkpoint'],sha256=v['sha256']) for a,v in validations.items()},
        arms=ARMS,planner_seeds=PLANNER_SEEDS,episodes_per_arm=64,total_episodes=448,max_steps=300,
        device='cpu',torch_threads=1,python=sys.version,torch=torch.__version__,platform=platform.platform()))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','suite','arm','candidates'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='suite':
        from experiments.four_path_execution.suite import run
        # Separate tiny entry point retains the common subprocess runner's CLI.
        raise SystemExit(run(args.root,args.workers,module='experiments.four_path_oracle_execution.arm'))
    elif args.command=='arm':evaluate(args.root,args.arm)
    else:
        from experiments.four_path_execution.candidates import diagnostic
        diagnostic(args.root,config=CONFIG,arms=ARMS,controller_factory=controller_factory(args.root),
                   comparison_pairs=[('oracle',a) for a in ARMS if a!='oracle'])


def evaluate(root,arm):
    from experiments.four_path_execution.run import evaluate_arm
    evaluate_arm(root,arm,config=CONFIG,controller_factory=controller_factory(root))


if __name__=='__main__':main()
