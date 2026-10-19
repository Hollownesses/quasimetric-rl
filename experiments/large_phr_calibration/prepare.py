"""Freeze the dual calibration protocol and its immutable witnessed parent."""
import argparse
import platform
import shutil
import subprocess
from pathlib import Path
import torch
from experiments.four_path_execution.common import read,write,file_digest
from experiments.large_phr_calibration.train import SOURCE,EXPECTED_SOURCE


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    parent=SOURCE.parent.parent
    assert file_digest(SOURCE)==EXPECTED_SOURCE
    config=read(SOURCE.parent/'config.json');assert config['task_mode']=='witnessed'
    dependency_sources=read(parent/'protocol.json')['source_hashes'].copy()
    dependency_sources.update(read(parent/'dependency_audit.json')['source_hashes'])
    for path,digest in dependency_sources.items():assert file_digest(path)==digest,path
    for path in Path('experiments/large_phr_calibration').glob('*.py'):
        dependency_sources[str(path)]=file_digest(path)
    for name in dependency_sources:
        target=root/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(name,target)
    old_files=['train/config.json','train/final.pt','train/final_constraints.npz','train/training.jsonl',
               'train/data_audit.json','train/diagnostic_probes.json','comparison/summary.json',
               'comparison/four_path.json','comparison/four_path_values.json','comparison/recorded_values.npz']
    write(root/'protocol.json',dict(parent=str(parent),parent_sha256={p:file_digest(parent/p) for p in old_files},
        source_hashes=dependency_sources,source_checkpoint=str(SOURCE),source_checkpoint_sha256=EXPECTED_SOURCE,
        source_step=25000,extra_steps=5000,source_initialization_seed=606,
        data_fingerprints=config['fingerprints'],task_mask_sha256=config['task_mask_sha256'],
        arm_A='Original dual += pre-primal h on each sampled visit, eta=1; path dual same every step.',
        arm_B='Original dual held fixed between outer iterations. Every 250 primal steps, scan latest model and apply relu(lambda+100*h) simultaneously to all 200000 original physical rows. No per-visit original update; path dual unchanged.',
        outer_period=250,outer_eta=100.,outer_updates=20,
        boundary_semantics='Updates occur after primal steps 250,...,5000. The final update is saved for recovery but cannot change the already-completed final primal model; 19 outer updates influence subsequent primal steps within this 5k window.',
        outer_rationale='eta=rho=100 is the standard PHR outer ascent coefficient, rather than the prior eta=1 visited-row heuristic. 250 primal steps allow adaptation between refreshes and yield 20 full updates within the fixed 5k budget. No initial update; first outer step follows primal step 250.',
        attribution_limit='B changes the freshness, frequency, and cumulative strength of original dual updates. This is a calibration-policy contrast, not a frequency-only or matched-dual-budget ablation. Nominal eta sum per edge B=2000, A=about 6.4; projections and changing residuals preclude converting this to a fixed actual multiplier ratio.',
        common=dict(batch=256,threads=4,rho=100.,path_dual_eta=1.,task_mode='witnessed',
            push=config['push'],constraints=config['constraints'],task_normalization=config['task_normalization'],
            optimizer='inherit complete AdamW moments/steps, weight_decay=0',
            lr='restart low-rate decay at 2e-6, exponentially reach 2e-7 on continuation step 5000; both arms identical',
            sampler='inherit source order/cursor/epochs/RNG and retain paired physical batches',
            audit='both arms scan all 201087 rows every 250 steps; B applies outer update after scan, checkpoints saved after update'),
        stopping='Fixed 5000 extra updates in each arm; evaluate final only; no graph/Oracle stopping or selection.',
        scope='No task-push, architecture, replay, cost, MPPI, success-definition, QP or collection changes. No closed-loop evaluation.',
        numeric_limit='Four CPU threads exhibit small nondeterministic update differences as documented in parent; this remains a single paired checkpoint continuation.',
        runtime=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform()),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
