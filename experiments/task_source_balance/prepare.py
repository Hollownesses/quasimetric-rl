"""Freeze training-only protocol and full source provenance before launching either arm."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from experiments.connected_expert_utrap.data import load_training_data,audit_training_data
from experiments.four_path_factorial.run import file_digest
from .train import task_weights,initial_state,equal,write

DATA='results/connected_expert_utrap_20261021/main'
HISTORY='results/connected_expert_qrl_20261023/main/train'


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=load_training_data(DATA);weights,stats=task_weights(data)
    history=json.loads(Path(HISTORY,'config.json').read_text())
    expected=dict(seed=606,phr_steps=45000,decay_steps=5000,batch=256,threads=4,
                  audit_interval=1000,task_mode='all',rho=100.,dual_step=1.,b1_seed=42)
    for key,value in expected.items(): assert history[key]==value,(key,history[key],value)
    assert data.fingerprint==history['fingerprints']
    initial=torch.load(Path(HISTORY,'initial.pt'),weights_only=True)
    model,opt,dual,visits,sampler=initial_state(data,606)
    fresh=dict(model=model.state_dict(),optimizer=opt.state_dict(),dual=dual,visits=visits,
               sampler=sampler.state_dict(),rng=torch.get_rng_state())
    for key,value in fresh.items():assert equal(value,initial[key]),key
    assert initial['step']==0 and not dual.any()
    # Counts are measured above. These are explicit expectation checks, never weight constants.
    assert stats['N']==42536 and stats['M']==6766,stats
    write(out/'data_audit.json',audit_training_data(data));torch.save(weights,out/'task_weights.pt')
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    files=[p for p in tracked if p.endswith('.py') and Path(p).is_file()]
    files+= [str(p) for p in Path('experiments/task_source_balance').glob('*.py')]
    source={}
    for file in sorted(set(files)):
        source[file]=file_digest(file);target=out/'source_snapshot'/file
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    evidence={}
    for rev in ('e06e2ad','50452ef','eb2967d','e162dd3','b7c940f'):
        evidence[rev]=subprocess.check_output(['git','rev-parse',rev],text=True).strip()
    # Verify the original trainer still equals 50452ef; no diagnostic trainer is inherited.
    old=subprocess.check_output(['git','show','50452ef:experiments/full_replay_four_path/train.py'])
    assert old==Path('experiments/full_replay_four_path/train.py').read_bytes()
    protocol=dict(**expected,decay_start_lr=2e-6,decay_end_lr=2e-7,main_lr=2e-5,
        sampler_seed=2307,optimizer='AdamW',weight_decay=0.,task_coefficient=.6,b1_coefficient=.4,
        data=DATA,fingerprints=data.fingerprint,weights=stats,
        historical_initial=str(Path(HISTORY,'initial.pt')),
        historical_initial_sha256=file_digest(Path(HISTORY,'initial.pt')),
        historical_config_sha256=file_digest(Path(HISTORY,'config.json')),
        initial_exact_fields=list(fresh),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        historical_revisions=evidence,source_files=source,
        sampling='physical records without replacement; retain partial epoch-ending batch (40 records)',
        task_measure='R0 physical-record mean; R1 equal canonical source mean; changes empirical task objective distribution',
        dual_clock='per sampled record, pre-primal residual, eta=1; distinct multiplier for every record',
        stopping='50000 fixed updates; final.pt official; no reference or oracle used by training/weights/stopping/selection',
        historical_protocol_conflicts=[],four_thread_bitwise_reproduction_promised=False)
    write(out/'protocol.json',protocol)
    write(out/'PREFLIGHT_COMPLETE',dict(protocol_sha256=file_digest(out/'protocol.json'),historical_initial_exact=True,weights=stats))
    print(json.dumps(dict(output=str(out),weights=stats,initial_exact=True),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
