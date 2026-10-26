"""Reference-free verification and freeze for the sole seed606 R2 run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import torch
from experiments.task_partial_balance.prepare import verify_existing,environment,OLD,FIELDS
from experiments.connected_expert_utrap.data import load_training_data,audit_training_data
from experiments.task_source_balance.train import task_weights,initial_state,initial_fields,equal,write
from experiments.four_path_factorial.run import file_digest,tensor_digest
from .train import edge_weights

HALF=Path('results/task_partial_balance_20261025/main')


def verify_baselines(data):
    previous,verification=verify_existing(data)
    half_p=json.loads((HALF/'protocol.json').read_text());manifest=json.loads((HALF/'final_integrity.json').read_text())
    assert json.loads((HALF/'COMPLETE').read_text())['final_integrity_sha256']==file_digest(HALF/'final_integrity.json')
    for name,h in manifest['artifacts'].items():assert file_digest(name)==h,name
    for key in ('seed','sampler_seed','b1_seed','batch','threads','phr_steps','decay_steps','audit_interval',
        'task_mode','rho','dual_step','optimizer','weight_decay','main_lr','decay_start_lr','decay_end_lr','fingerprints'):
        assert half_p[key]==previous[key],key
    assert half_p['alpha']==.5
    folder=HALF/'Rhalf';cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'COMPLETE').read_text())
    sig=dict(protocol_sha256=file_digest(HALF/'protocol.json'),task_weight_mode='partial-source',alpha=.5,
        weights_sha256=half_p['mixed_weights']['weights_sha256'],fingerprints=data.fingerprint)
    assert cfg['training_signature']==done['training_signature']==sig and done['step']==50000 and not done['reference_loaded']
    common=torch.load(OLD/'R1/initial.pt',weights_only=True);r1=torch.load(OLD/'R1/final.pt',weights_only=True)
    for name,step in [('initial.pt',0),('phr.pt',45000),('final.pt',50000)]:
        ck=torch.load(folder/name,weights_only=True)
        assert ck['step']==step and ck['training_signature']==sig and ck['fingerprints']==data.fingerprint
        if step==0:
            for k in FIELDS:assert equal(ck[k],common[k]),k
        if step==50000:
            for k in ('sampler','visits','rng','push_counters','sampling_chain_sha256'):assert equal(ck[k],r1[k]),k
    assert done['final_sha256']==file_digest(folder/'final.pt')
    scans=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines() if 'connected_expert"' in s]
    assert [r['step'] for r in scans]==list(range(0,50001,1000))
    for name in ('initial.pt','phr.pt','final.pt','config.json','COMPLETE','training.jsonl'):
        verification['artifacts'][str(folder/name)]=file_digest(folder/name)
    verification['half_config_initial_and_budget_comparable']=True
    return previous,verification


def run(output):
    torch.set_num_threads(4);runtime=environment()
    previous=json.loads((OLD/'protocol.json').read_text());data=load_training_data(previous['data'])
    old,reuse=verify_baselines(data);w,wstats=task_weights(data);assert wstats==old['weights']
    v,ids,reps,counts,estats=edge_weights(data)
    # Expectations are checks on measured values, never constants in weighting.
    assert (estats['N'],estats['K'])==(42536,6958)
    common=torch.load(OLD/'R1/initial.pt',weights_only=True);fresh=initial_fields(*initial_state(data,606,2307))
    for k in FIELDS:assert equal(fresh[k],common[k]),k
    assert not common['dual'].any() and common['step']==0
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'runtime.json',runtime);write(out/'reuse_verification.json',reuse)
    write(out/'data_audit.json',audit_training_data(data))
    torch.save(w,out/'task_weights.pt');torch.save(dict(weights=v,ids=ids,representatives=reps,counts=counts),out/'edge_weights.pt')
    names=[n for n in subprocess.check_output(['git','ls-files'],text=True).splitlines() if n.endswith('.py') and Path(n).is_file()]
    names += [str(x) for x in Path('experiments/phr_edge_balance').glob('*.py')]
    sources={}
    for name in sorted(set(names)):
        sources[name]=file_digest(name);target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    p={k:v for k,v in old.items() if k not in ('source_files','git_head','task_measure','constraints')}
    p.update(protocol_version='phr-unique-edge-v1',task_weight_mode='source',phr_weight_mode='unique-edge',
        edge_weights=estats,source_files=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initial_checkpoint=str(OLD/'R1/initial.pt'),initial_checkpoint_sha256=file_digest(OLD/'R1/initial.pt'),initial_exact_to_history=True,
        baseline_roots=dict(R0=str(OLD/'R0'),R1=str(OLD/'R1'),Rhalf=str(HALF/'Rhalf')),
        prior_protocol_sha256=file_digest(OLD/'protocol.json'),reused_artifacts=reuse['artifacts'],
        runtime_sha256=file_digest(out/'runtime.json'),
        task_measure='same R1 global canonical source balance; coefficient 0.6',
        constraints='global edge weight N/(K*m) times EACH record PHR numerator BEFORE mean; rho100; all record multipliers retained',
        dual_clock='UNCHANGED: per record visit lambda_i=relu(lambda_i+h_i), pre-primal h, eta1; no edge weighting or merging',
        edge_copy_bins=[dict(label='1',min=1,max=1),dict(label='2-4',min=2,max=4),dict(label='5-16',min=5,max=16),
            dict(label='17-64',min=17,max=64),dict(label='65+',min=65,max=None)],
        tail_quantiles=[.9,.95,.99],fixed_path_policy='Freeze seed606 R0 final paths; 165 unique starts primary, 307 auxiliary; evaluate only post-budget',
        additional_seeds=False,hyperparameter_search=False)
    write(out/'protocol.json',p)
    write(out/'PREFLIGHT_COMPLETE',dict(passed=True,protocol_sha256=file_digest(out/'protocol.json'),
        edge_weights=estats,initial_exact=True,baselines_verified=True,runtime_matches_prior=True))
    print(json.dumps(dict(output=str(out),edge_weights=estats,task_weights=wstats,reuse_verified=True),indent=2));return p

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
