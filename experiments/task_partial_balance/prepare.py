"""Verify existing R0/R1 without reference labels; freeze only alpha=0.5."""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import numpy as np
import torch
import torchqmet
from experiments.connected_expert_utrap.data import load_training_data,audit_training_data
from experiments.task_source_balance.train import initial_state,initial_fields,equal
from experiments.four_path_factorial.run import file_digest
from .train import mixed_weights,write

OLD=Path('results/task_source_balance_20261025/main')
FIELDS=('model','optimizer','dual','visits','sampler','rng','push_counters')


def verify_existing(data):
    p=json.loads((OLD/'protocol.json').read_text())
    assert p['fingerprints']==data.fingerprint
    expected=dict(seed=606,batch=256,threads=4,phr_steps=45000,decay_steps=5000,
        audit_interval=1000,sampler_seed=2307,b1_seed=42,task_mode='all',rho=100.,dual_step=1.,
        optimizer='AdamW',weight_decay=0.,task_coefficient=.6,b1_coefficient=.4,
        main_lr=2e-5,decay_start_lr=2e-6,decay_end_lr=2e-7)
    for k,v in expected.items():assert p[k]==v,(k,p[k],v)
    integrity=json.loads((OLD/'final_integrity.json').read_text())
    assert json.loads((OLD/'COMPLETE').read_text())['final_integrity_sha256']==file_digest(OLD/'final_integrity.json')
    for name,digest in integrity['artifacts'].items():assert file_digest(OLD.parent/name)==digest,name
    initial={};final={};artifacts={};audits={}
    for arm,mode in [('R0','record'),('R1','source')]:
        folder=OLD/arm;cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'COMPLETE').read_text())
        sig=dict(protocol_sha256=file_digest(OLD/'protocol.json'),task_weight_mode=mode,
            weights_sha256=p['weights']['weights_sha256'],fingerprints=data.fingerprint)
        assert cfg['training_signature']==done['training_signature']==sig
        for k in expected:assert cfg[k]==p[k],(arm,k)
        assert done['step']==50000 and not done['reference_loaded'] and done['final_sha256']==file_digest(folder/'final.pt')
        initial[arm]=torch.load(folder/'initial.pt',weights_only=True)
        final[arm]=torch.load(folder/'final.pt',weights_only=True)
        for name,step in [('initial.pt',0),('phr.pt',45000),('final.pt',50000)]:
            ck=torch.load(folder/name,weights_only=True)
            assert ck['step']==step and ck['training_signature']==sig and ck['fingerprints']==data.fingerprint
        records=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
        audits[arm]=[r for r in records if 'connected_expert' in r]
        assert [r['step'] for r in audits[arm]]==list(range(0,50001,1000))
        for name in ('initial.pt','phr.pt','final.pt','config.json','COMPLETE','training.jsonl'):
            artifacts[str(folder/name)]=file_digest(folder/name)
    for k in FIELDS:assert equal(initial['R0'][k],initial['R1'][k]),k
    for k in ('sampler','visits','rng','push_counters','sampling_chain_sha256'):
        assert equal(final['R0'][k],final['R1'][k]),k
    assert initial['R0']['step']==0 and not initial['R0']['dual'].any()
    assert int(final['R0']['visits'].sum())==12735416
    for x,y in zip(audits['R0'],audits['R1']):
        for k in ('sampling_chain_sha256','push_counters','connected_expert_visits'):assert x[k]==y[k]
    # Stored old sources must match their old frozen hashes. Current changes are
    # limited to the subsequent seed infrastructure; objective/update ASTs agree.
    changed=[]
    for name,digest in p['source_files'].items():
        assert file_digest(OLD/'source_snapshot'/name)==digest,name
        if file_digest(name)!=digest:changed.append(name)
    assert set(changed)<=set('experiments/task_source_balance/'+x+'.py' for x in ('prepare','train','test_balance'))
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
    old=functions(OLD/'source_snapshot/experiments/task_source_balance/train.py');now=functions('experiments/task_source_balance/train.py')
    core=('task_weights','weighted_push','advance','learning_rate','restore')
    for name in core:assert old[name]==now[name],name
    return p,dict(passed=True,artifacts=artifacts,existing_core_functions_identical=list(core),
        changed_since_original_freeze=changed,existing_paired_initial_exact=True,
        existing_paired_budget_exact=True,visits_sum=int(final['R0']['visits'].sum()),
        sampling_chain_sha256=final['R0']['sampling_chain_sha256'])


def environment():
    old=json.loads((OLD/'runtime.json').read_text())
    current=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=str(torch.__version__),
        numpy=np.__version__,torch_build=torch.__config__.show(),formal_threads_per_process=4,
        packages={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata['Name']})
    for k in ('python','torch','numpy','torch_build','platform'):assert old[k]==current[k],('runtime changed',k)
    ext=json.loads((OLD/'external_source_manifest.json').read_text())
    for name,h in ext['sha256'].items():assert file_digest(Path(torchqmet.__file__).parent/name)==h
    current['matches_prior_runtime']=True;current['torchqmet_matches_prior_source']=True
    return current


def run(output):
    torch.set_num_threads(4)
    runtime=environment();previous=json.loads((OLD/'protocol.json').read_text())
    data=load_training_data(previous['data']);old,verification=verify_existing(data)
    weights,mixed,base=mixed_weights(data,.5);assert base==old['weights']
    fresh=initial_fields(*initial_state(data,606,2307))
    common=torch.load(OLD/'R0/initial.pt',weights_only=True)
    for k in FIELDS:assert equal(fresh[k],common[k]),('fresh paired initial',k)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'runtime.json',runtime);write(out/'reuse_verification.json',verification)
    write(out/'data_audit.json',audit_training_data(data));torch.save(weights,out/'task_weights.pt')
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    names=[n for n in tracked if n.endswith('.py') and Path(n).is_file()]
    names += [str(p) for p in Path('experiments/task_partial_balance').glob('*.py')]
    sources={}
    for name in sorted(set(names)):
        sources[name]=file_digest(name);target=out/'source_snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    p={k:v for k,v in old.items() if k not in ('source_files','git_head','task_measure')}
    p.update(protocol_version='partial-source-v1',alpha=.5,task_weight_mode='partial-source',mixed_weights=mixed,
        source_files=sources,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initial_checkpoint=str(OLD/'R0/initial.pt'),initial_checkpoint_sha256=file_digest(OLD/'R0/initial.pt'),
        initial_exact_to_history=True,prior_root=str(OLD),prior_protocol_sha256=file_digest(OLD/'protocol.json'),
        reused_artifacts=verification['artifacts'],runtime_sha256=file_digest(out/'runtime.json'),
        task_measure='0.5 original record measure + 0.5 equal canonical source measure; only task changes',
        multiplicity_bins=[dict(label='1',min=1,max=1),dict(label='2-4',min=2,max=4),
            dict(label='5-16',min=5,max=16),dict(label='17-64',min=17,max=64),dict(label='65+',min=65,max=None)],
        grouping_policy='Prespecified bins based only on task-source multiplicity; post-budget evaluation; no reference used for bins',
        alpha_search=False,additional_seeds=False)
    write(out/'protocol.json',p)
    write(out/'PREFLIGHT_COMPLETE',dict(passed=True,protocol_sha256=file_digest(out/'protocol.json'),
        initial_exact=True,old_runs_reusable=True,runtime_matches_prior=True,mixed_weights=mixed))
    print(json.dumps(dict(output=str(out),weights=mixed,reuse_verified=True,runtime_matches=True),indent=2))
    return p

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
