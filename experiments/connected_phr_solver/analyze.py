"""Post-budget evaluation and exact pairing audit; never imported by training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.connected_expert_utrap.compare import evaluate,graph_inputs,equality,write_json
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.four_path_factorial.run import file_digest
from .diagnostic import audit_checkpoint
from .train import SOURCE,SOURCE_SHA,DATA


def verify_pair(parent,initials,finals):
    for arm in ('A','B'):
        for key in ('model','optimizer','dual','visits','sampler','rng','push_counters'):
            assert equality(parent[key],initials[arm][key]),(arm,'initial',key)
        final=finals[arm]
        assert final['continuation_step']==5000 and final['step']==parent['step']+5000
        assert torch.equal(final['nominal_dual_steps'],final['visits']-parent['visits'])
    for key in ('visits','nominal_dual_steps','sampler','rng','push_counters'):
        assert equality(finals['A'][key],finals['B'][key]),('final mismatch',key)
    return dict(initial_state_exact=True,final_sampling_and_nominal_budgets_exact=True,
                extra_visits_min=int(finals['A']['nominal_dual_steps'].min()),
                extra_visits_max=int(finals['A']['nominal_dual_steps'].max()),
                extra_visits_sum=int(finals['A']['nominal_dual_steps'].sum()))


def run(args):
    torch.set_num_threads(1);root=Path(args.root);out=root/'comparison';out.mkdir(exist_ok=False)
    data=load_training_data(DATA);graph,starts=graph_inputs(data)
    refroot=Path('results/connected_reference_fix_20261024/main')
    refinfo=json.loads((refroot/'summary.json').read_text())
    assert refinfo['fingerprints']==data.fingerprint
    assert file_digest(refroot/'graph_reference_audit_only.npz')==refinfo['reference_sha256']
    with np.load(refroot/'graph_reference_audit_only.npz') as a:
        ref=a['distance'].copy();next_edge=a['next_edge'].copy()
        assert np.array_equal(a['source'],data.src.numpy()) and np.array_equal(a['target'],data.dst.numpy())
    validate_reference(data,ref)
    assert file_digest(SOURCE)==SOURCE_SHA
    parent=torch.load(SOURCE,weights_only=True);initials={};finals={};configs={};logs={};complete={}
    for arm in ('A','B'):
        folder=root/arm;complete[arm]=json.loads((folder/'COMPLETE').read_text())
        assert complete[arm]['steps']==5000 and not complete[arm]['reference_loaded']
        configs[arm]=json.loads((folder/'config.json').read_text())
        assert configs[arm]['source_sha256']==SOURCE_SHA
        for path,digest in configs[arm]['source_files'].items():assert file_digest(path)==digest
        initials[arm]=torch.load(folder/'initial.pt',weights_only=True)
        finals[arm]=torch.load(folder/'final.pt',weights_only=True)
        records=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        logs[arm]=[r for r in records if r['kind']=='boundary']
        assert [r['continuation_step'] for r in logs[arm]]==list(range(0,5001,500))
        for step in (0,5000):assert json.loads((folder/f'supervision_{step:04d}.json').read_text())['passed']
    for key in configs['A']:
        if key!='arm':assert configs['A'][key]==configs['B'][key],key
    pairing=verify_pair(parent,initials,finals)
    before=torch.load(root/'B/final_before_outer.pt',weights_only=True)
    start=torch.load(root/'B/last_block_start.pt',weights_only=True)
    assert torch.equal(start['dual'],before['dual']),'last inner block did not freeze multipliers'
    for key in ('model','optimizer','visits','sampler','rng','push_counters'):
        assert equality(before[key],finals['B'][key]),('final outer changed primal state',key)
    pairing.update(last_B_inner_dual_exactly_frozen=True,final_B_outer_changes_no_primal_state=True)
    results={};decision_details={};arrays={}
    prior=json.loads(Path('results/connected_phr_diagnostic_20261024/main/summary.json').read_text())
    assert prior['results']['50k']['checkpoint_sha256']==SOURCE_SHA
    for arm,path in [('parent',SOURCE),('A',root/'A/final.pt'),('B',root/'B/final.pt')]:
        before=file_digest(path)
        result,pred,decision=evaluate(data,graph,ref,starts,path)
        if arm=='parent':diag=prior['results']['50k']
        else:
            diag,pieces=audit_checkpoint(data,ref,next_edge,path)
            arrays.update({arm+'_'+k:v for k,v in pieces.items()})
        result['diagnostic']=diag;results[arm]=result;arrays[arm+'_prediction']=pred
        decision_details[arm]=decision
        write_json(out/(arm+'_decisions.json'),decision)
        assert file_digest(path)==before
        print(json.dumps(dict(arm=arm,distance=result['distance']['unique_sources'],
            local=result['local']['unique_edges'],decisions=result['decisions']),ensure_ascii=False),flush=True)
    np.savez_compressed(out/'arrays.npz',reference=ref,**arrays)
    changes={}
    for left,right in [('parent','A'),('parent','B'),('A','B')]:
        lb={b['source']:b for b in decision_details[left]['branches']}
        rb={b['source']:b for b in decision_details[right]['branches']}
        changes[left+'_'+right]=[dict(source=s,left_choice=lb[s]['chosen_row'],right_choice=rb[s]['chosen_row'],
            left_regret=lb[s]['regret'],right_regret=rb[s]['regret'])
            for s in lb if lb[s]['chosen_row']!=rb[s]['chosen_row']]
    summary=dict(pairing=pairing,results=results,training=logs,complete=complete,branch_changes=changes,
        fingerprints=data.fingerprint,reference_sha256=refinfo['reference_sha256'],
        analysis_source_sha256=file_digest(__file__))
    write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm in ('A','B'):
        history=logs[arm];steps=[r['continuation_step'] for r in history]
        axes[0].plot(steps,[r['max_violation'] for r in history],'-o',label=arm)
        axes[1].plot(steps,[r['mean_positive'] for r in history],'-o',label=arm)
    axes[0].set(xlabel='Continuation step',ylabel='Maximum edge violation');axes[0].legend()
    axes[1].set(xlabel='Continuation step',ylabel='Mean positive edge violation');axes[1].legend()
    labels=['parent','A','B'];x=np.arange(3)
    for j,key in enumerate(('unique_sources','longest_quartile','unique_starts')):
        axes[2].bar(x+(j-1)*.24,[results[a]['distance'][key]['nrmse_percent'] for a in labels],width=.24,label=key)
    axes[2].set(xticks=x,xticklabels=labels,ylabel='Task NRMSE (%)');axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    write_json(out/'COMPLETE',dict(passed=True,pairing=pairing))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/connected_phr_solver_20261024/main')
    run(p.parse_args())
