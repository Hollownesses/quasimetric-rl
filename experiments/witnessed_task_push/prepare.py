"""Freeze the paired objective-only intervention, without value references."""
import argparse
import platform
import shutil
import subprocess
from pathlib import Path
import torch
from experiments.four_path_execution.common import read, write, file_digest

BASELINE=Path('results/full_replay_four_path_20261017/main')
SOURCES=[
    'experiments/full_replay_four_path/data.py',
    'experiments/full_replay_four_path/train.py',
    'experiments/four_path_factorial/run.py',
    'minimal_qrl/global_push_sampling.py',
    'quasimetric_rl/modules/quasimetric_critic/losses/global_push.py',
    'quasimetric_rl/modules/quasimetric_critic/losses/phr.py',
]

def freeze_dependencies(root,baseline):
    """Also verify the model/environment dependency snapshot against the old run."""
    target=root/'dependency_audit.json';assert not target.exists()
    original=baseline/'source_snapshot';hashes={}
    for group in ('quasimetric_rl','minimal_qrl','torchqmet'):
        for previous in (original/group).rglob('*.py'):
            relative=previous.relative_to(original)
            current=Path(relative)
            assert current.exists(),str(current)
            assert file_digest(current)==file_digest(previous),str(current)
            dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(current,dest);hashes[str(relative)]=file_digest(current)
    write(target,dict(unchanged_from_baseline=True,source_hashes=hashes))

def run(root, baseline=BASELINE):
    root.mkdir(parents=True,exist_ok=True)
    target=root/'protocol.json'
    assert not target.exists(), 'Do not overwrite a frozen protocol'
    config=read(baseline/'train/config.json')
    hashes={}
    for name in SOURCES:
        p=Path(name);dest=root/'source_snapshot'/p
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes[name]=file_digest(p)
    old_files=['train/config.json','train/initial.pt','train/phr.pt','train/final.pt',
               'train/training.jsonl','train/final_constraints.npz',
               'value_analysis/recorded_values.npz','value_analysis/four_path.json',
               'value_analysis/four_path_values.json']
    protocol=dict(
        baseline=str(baseline),source_hashes=hashes,
        baseline_hashes={p:file_digest(baseline/p) for p in old_files},
        task_mode='witnessed',seed=606,
        intervention='Only task push selects witnessed successful-suffix sources; all real edge constraints and valid B1 pairs remain.',
        task_normalization='Mean over retained task entries in each minibatch and source stratum. Empty task batch contributes differentiable zero; B1 retains 0.4 weight.',
        unchanged={k:config[k] for k in ('seed','phr_steps','decay_steps','batch','threads','rho','dual_step','fingerprints','initial_model_sha256')},
        questions=['corridor value growth','four-path long-distance recovery','all-recorded-edge feasibility'],
        stopping='Fixed 20000 PHR + 5000 decay steps, final checkpoint only. No reference-based stopping or selection.',
        scope='Paired single-seed objective diagnosis; no new collection, model, QP, MPPI or cost changes. No new navigation claims.',
        diagnostics='Fixed 512 rows per original task and all path sources; model values only during training. Graph references reused only after COMPLETE.',
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform()),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if (root/'train/config.json').exists():
        actual=read(root/'train/config.json')
        for k,v in protocol['unchanged'].items():assert actual[k]==v,k
        assert actual['task_mode']=='witnessed'
        protocol['initial_config_verified']=True
    write(target,protocol)
    freeze_dependencies(root,baseline)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=BASELINE);a=p.parse_args();run(a.root,a.baseline)
