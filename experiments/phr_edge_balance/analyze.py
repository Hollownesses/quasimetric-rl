"""Only after R2 finishes: common corrected-reference and fixed-path evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import evaluate,graph_inputs,graph_structure,decisions
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.task_source_balance.analyze import decomposition
from experiments.connected_phr_solver.objective import all_distances
from experiments.four_path_factorial.run import IndexedIQE,file_digest,tensor_digest
from experiments.full_replay_four_path.train import EpochSampler
from experiments.constraint_exposure_diagnostic.core import rollout_exposure
from .prepare import verify_baselines,OLD,HALF,FIELDS
from .train import load_training_data,task_weights,edge_weights,equal,write,learning_rate
from .metrics import residual_stats,copy_groups,path_positive,successor_diagnostics,common_pair_changes

ARMS=('R0','Rhalf','R1','R2')


def verify_r2(root,p,data):
    folder=root/'R2';cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'COMPLETE').read_text())
    w,ws=task_weights(data);v,ids,reps,counts,es=edge_weights(data)
    assert es==p['edge_weights'] and ws==p['weights']
    sig=dict(protocol_sha256=file_digest(root/'protocol.json'),task_weight_mode='source',phr_weight_mode='unique-edge',
        edge_weights_sha256=es['weights_sha256'],edge_partition_sha256=es['partition_sha256'],
        weights_sha256=ws['weights_sha256'],fingerprints=data.fingerprint)
    assert cfg['training_signature']==done['training_signature']==sig
    assert done['step']==50000 and not done['reference_loaded'] and done['final_sha256']==file_digest(folder/'final.pt')
    for k in p:assert cfg[k]==p[k],('configuration changed',k)
    checkpoints={}
    for name,step in [('initial.pt',0),('phr.pt',45000),('final.pt',50000)]:
        ck=torch.load(folder/name,weights_only=True);checkpoints[step]=ck
        assert ck['step']==step and ck['training_signature']==sig and ck['fingerprints']==data.fingerprint
        assert ck['task_weight_mode']=='source' and ck['phr_weight_mode']=='unique-edge'
        assert torch.equal(ck['task_weights'],w) and torch.equal(ck['edge_weights'],v)
        assert torch.equal(ck['edge_ids'],ids) and torch.equal(ck['edge_counts'],counts) and torch.equal(ck['edge_representatives'],reps)
    initial=checkpoints[0];final=checkpoints[50000]
    for arm,base in p['baseline_roots'].items():
        i=torch.load(Path(base)/'initial.pt',weights_only=True);f=torch.load(Path(base)/'final.pt',weights_only=True)
        for k in FIELDS:assert equal(initial[k],i[k]),(arm,k)
        for k in ('sampler','visits','rng','push_counters','sampling_chain_sha256'):assert equal(final[k],f[k]),(arm,k)
    lines=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
    scans=[r for r in lines if 'connected_expert' in r];updates=[r for r in lines if 'lr' in r]
    assert [r['step'] for r in scans]==list(range(0,50001,1000))
    assert [r['step'] for r in updates]==list(range(100,50001,100))
    for r in updates:
        assert r['lr']==learning_rate(r['step'],45000,5000)
        c=r['connected_expert_components'];assert c['task_rows']==c['pair_rows']==c['optimized_pair_rows']
    sampler=EpochSampler(len(data.original),2307);visits=torch.zeros_like(final['visits']);chain='0'*64
    for _ in range(50000):
        ii=sampler.next(256);visits[ii]+=1;chain=hashlib.sha256(bytes.fromhex(chain)+data.original[ii].numpy().tobytes()).hexdigest()
    assert chain==final['sampling_chain_sha256']==done['sampling_chain_sha256']
    assert torch.equal(visits,final['visits']) and equal(sampler.state_dict(),final['sampler'])
    return dict(initial_complete_state_exact_to_all_baselines=True,zero_initial_dual=not bool(initial['dual'].any()),
        ordered_sequence_independently_reconstructed=True,final_sampler_visits_rng_counters_exact=True,
        sampling_chain_sha256=chain,steps=50000,visits_sum=int(visits.sum()),visits_min=int(visits.min()),visits_max=int(visits.max()),
        full_audits=len(scans),task_weights_exact_r1=True,edge_weights_validated=True),scans


@torch.no_grad()
def model_edges(data,path):
    ck=torch.load(path,weights_only=True);model=IndexedIQE(data.small,'encoder',0,initialization_seed=ck['initialization_seed'])
    model.load_state_dict(ck['model']);model.eval();pred,edge=all_distances(model,data)
    return pred,edge-data.cost.double().numpy()


def extra_metrics(data,graph,ref,pred,h,decision,fixed,reps,ids,counts,bins):
    rows=data.original.numpy();reps=reps.numpy();ids=ids.numpy();counts=counts.numpy()
    duplicate_error=float(abs(h[rows]-h[reps[ids[rows]]]).max());assert duplicate_error<1e-6
    local=dict(records=residual_stats(h[rows]),unique_edges=residual_stats(h[reps]))
    copies=copy_groups(h,rows,reps,ids,counts,bins)
    exposure,_,regrets=rollout_exposure(fixed['episodes'],data.src.numpy(),data.dst.numpy(),data.cost.double().numpy(),ids,ref,graph.goal,unique=True)
    fixed165=path_positive(h,fixed['episodes'],True);fixed307=path_positive(h,fixed['episodes'],False)
    assert fixed165['count']==165 and fixed307['count']==307
    assert abs(np.dot(exposure,np.maximum(h[reps],0))/165-fixed165['mean'])<1e-8
    own165=path_positive(h,decision['episodes'],True)
    rollout_exposure(decision['episodes'],data.src.numpy(),data.dst.numpy(),data.cost.double().numpy(),ids,ref,graph.goal,unique=True)
    branches=successor_diagnostics(data,graph,ref,pred,decision)
    return dict(local=local,edge_copy_groups=copies,fixed_R0_paths165=fixed165,fixed_R0_paths307=fixed307,
        own_paths165=own165,successor_error=branches['summary'],duplicate_residual_max_difference=duplicate_error),branches


def finalize(root,summary):
    out=root/'comparison'
    sources={}
    for name in ('analyze.py','metrics.py','report.py'):
        source=Path(__file__).parent/name;sources[str(source)]=file_digest(source)
        shutil.copy2(source,out/('source_'+name))
    summary['analysis_sources']=sources
    from .report import assessment
    summary['assessment']=assessment(summary)
    write(out/'summary.json',summary)
    from .report import report,plot
    report(out,summary);plot(out,summary)
    write(out/'COMPLETE',dict(passed=True,summary_sha256=file_digest(out/'summary.json'),report_sha256=file_digest(out/'report.md'),pairing=summary['pairing']))
    p=json.loads((root/'protocol.json').read_text())
    for file,digest in p['reused_artifacts'].items():assert file_digest(file)==digest,file
    files=[root/'protocol.json',root/'reuse_verification.json',root/'runtime.json',root.parent/'tests.log',
        out/'summary.json',out/'report.md',out/'arrays.npz',out/'comparison.png',out/'branch_differences.json']
    if (root.parent/'analysis_tests.log').exists():files.append(root.parent/'analysis_tests.log')
    files += [root/'R2'/n for n in ('initial.pt','phr.pt','final.pt','COMPLETE','config.json','training.jsonl','edge_weights.pt')]
    files += [out/('source_'+n) for n in ('analyze.py','metrics.py','report.py')]
    write(root/'final_integrity.json',dict(passed=True,old_artifacts_unchanged=True,pairing=summary['pairing'],
        validation=summary['validation'],artifacts={str(f):file_digest(f) for f in files}))
    write(root/'COMPLETE',dict(passed=True,official_endpoint=50000,final_integrity_sha256=file_digest(root/'final_integrity.json')))


def run(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());torch.set_num_threads(1)
    assert json.loads((root/'R2/COMPLETE').read_text())['step']==50000
    for file,h in p['source_files'].items():assert file_digest(file)==h,file
    for file,h in p['reused_artifacts'].items():assert file_digest(file)==h,file
    data=load_training_data(p['data']);verify_baselines(data);pairing,logs=verify_r2(root,p,data)
    _,ids,reps,counts,es=edge_weights(data)
    # All reference and decision data are accessed only after completed-budget gate.
    refroot=Path('results/connected_reference_fix_20261024/main');info=json.loads((refroot/'summary.json').read_text())
    assert info['passed'] and info['fingerprints']==data.fingerprint
    assert file_digest(refroot/'graph_reference_audit_only.npz')==info['reference_sha256']
    with np.load(refroot/'graph_reference_audit_only.npz') as f:
        ref=f['distance'].copy();next_edge=f['next_edge'].copy()
        for k,t in [('source',data.src),('target',data.dst),('physical',data.physical)]:assert np.array_equal(f[k],t.numpy())
    validate_reference(data,ref);graph,starts=graph_inputs(data);structure=graph_structure(data,graph,starts)
    assert structure['acyclic'] and structure['longest_possible_recorded_rollout']==275
    assert structure['branches']==58 and structure['unique_starts']==165 and len(starts)==307
    assert sorted(r for rr in graph.outgoing for r in rr)==sorted(reps.tolist())
    oracle,_=decisions(data,graph,ref,ref,starts,300)
    assert oracle['summary']['optimal_branch_choices']==58 and oracle['summary']['mean_successful_regret']<1e-8
    fixed_path=OLD/'comparison/R0_decisions.json';fixed=json.loads(fixed_path.read_text())
    prior=json.loads((HALF/'comparison/summary.json').read_text())
    out=root/'comparison';out.mkdir(exist_ok=False)
    results={};stages={};extras={};stage_extras={};branch_details={};details={};arrays={};all_logs={}
    checks={}
    for arm in ARMS:
        folder=root/'R2' if arm=='R2' else Path(p['baseline_roots'][arm])
        for step,name in [(50000,'final.pt'),(45000,'phr.pt')]:
            path=folder/name;before=file_digest(path)
            result,pred,decision=evaluate(data,graph,ref,starts,path)
            result['decisions']['strict_reference_pairs']=sum(b['strict_reference_pairs'] for b in decision['branches'])
            result['decisions']['correctly_ordered_pairs']=sum(b['correctly_ordered_pairs'] for b in decision['branches'])
            diag,pieces=decomposition(data,ref,next_edge,path,pred,starts);result['decomposition']=diag
            actual,h=model_edges(data,path);assert np.array_equal(actual,pred)
            extra,branches=extra_metrics(data,graph,ref,pred,h,decision,fixed,reps,ids,counts,p['edge_copy_bins'])
            if arm!='R2':assert result==prior['results' if step==50000 else 'stage45000'][arm],(arm,step,'prior evaluator drift')
            if arm=='R0' and step==50000:assert decision==fixed,'fixed R0 decisions changed'
            assert file_digest(path)==before
            if step==50000:
                results[arm]=result;extras[arm]=extra;branch_details[arm]=branches;details[arm]=decision
                arrays.update({arm+'_'+k:v for k,v in pieces.items()});arrays[arm+'_prediction']=pred;arrays[arm+'_residual']=h
                write(out/(arm+'_decisions.json'),decision);write(out/(arm+'_branch_errors.json'),branches)
                with np.load(folder/'final_constraints.npz') as f:scan_error=float(abs(h-f['residual']).max())
                assert scan_error<1e-3,(arm,scan_error)
                checks[arm]=dict(saved_scan_residual_max_difference=scan_error,
                    decomposition_identity_max_error=diag['identity_max_error'],duplicate_residual_max_difference=extra['duplicate_residual_max_difference'])
            else:stages[arm]=result;stage_extras[arm]=extra
            print(json.dumps(dict(arm=arm,step=step,distance=result['distance'],decisions=result['decisions'],local=extra['local'],fixed_R0_path_positive=extra['fixed_R0_paths165']['mean'])),flush=True)
        all_logs[arm]=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines() if 'connected_expert"' in s]
    pairs=common_pair_changes(branch_details['R1'],branch_details['R2'])
    left={b['source']:b for b in branch_details['R1']['branches']};right={b['source']:b for b in branch_details['R2']['branches']}
    changes=[dict(source=s,R1=left[s],R2=right[s]) for s in left if left[s]['chosen_row']!=right[s]['chosen_row']]
    differences=dict(R1_to_R2_strict_pairs=pairs,changed_branch_count=len(changes),changed_branches=changes)
    write(out/'branch_differences.json',differences)
    np.savez_compressed(out/'arrays.npz',reference=ref,starts=starts,edge_ids=ids.numpy(),edge_representatives=reps.numpy(),edge_counts=counts.numpy(),**arrays)
    summary=dict(results=results,stage45000=stages,extra=extras,stage45000_extra=stage_extras,pairing=pairing,
        validation=dict(passed=True,baseline_evaluations_exact=True,branch_score_identities_checked=True,fixed_path_connectivity_costs_and_regrets_checked=True,
            fixed_path_direct_sums_match_exposure_weighting=True,checks=checks),
        branch_changes=differences,edge_weights=es,fingerprints=data.fingerprint,graph_structure=structure,
        fixed_R0_paths_sha256=file_digest(fixed_path),reference_sha256=info['reference_sha256'],protocol_sha256=file_digest(root/'protocol.json'),
        official_endpoint=50000,training=all_logs,scope='Only seed606 R2 trained; four threads; fixed recorded graph; no checkpoint selection')
    finalize(root,summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
