"""Independent arithmetic/artifact checks of the saved, post-budget comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.compare import statistics
from experiments.four_path_factorial.run import file_digest
from .train import write


def run(root):
    root=Path(root);out=root/'comparison';summary=json.loads((out/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text());torch.set_num_threads(1)
    for p,h in protocol['source_files'].items():assert file_digest(p)==h,p
    data=load_training_data(protocol['data']);physical=data.original.numpy();src=data.src.numpy();dst=data.dst.numpy();cost=data.cost.double().numpy()
    unique=np.unique(src[physical]);max_identity=0.;max_regret_identity=0.
    with np.load(out/'arrays.npz') as f:
        ref=f['reference'];starts=f['starts']
        groups=dict(records=src[physical],unique_sources=unique,starts=starts,unique_starts=np.unique(starts),
            longest_quartile=unique[ref[unique]>=np.quantile(ref[unique],.75)])
        for arm in ('historical','R0','R1'):
            result=summary['results'][arm];pred=f[arm+'_prediction']
            gap=f[arm+'_gap'];edge=f[arm+'_edge_slack'];triangle=f[arm+'_triangle_slack']
            assert np.allclose(gap,ref-pred,atol=1e-10,rtol=0)
            err=float(abs(gap-edge-triangle).max());assert err<1e-7;max_identity=max(max_identity,err)
            for name,ids in groups.items():
                expected=statistics(pred[ids],ref[ids])
                for key,value in expected.items():assert abs(value-result['distance'][name][key])<1e-10
                for key,values in [('gap',gap),('edge_slack',edge),('triangle_slack',triangle)]:
                    assert abs(values[ids].mean()-result['decomposition']['groups'][name][key])<1e-10
            details=json.loads((out/(arm+'_decisions.json')).read_text())
            for b in details['branches']:
                alternatives=b['alternatives'];scores=[cost[a['row']]+pred[dst[a['row']]] for a in alternatives]
                assert alternatives[int(np.argmin(scores))]['row']==b['chosen_row']
                truth=cost[b['chosen_row']]+ref[dst[b['chosen_row']]]-ref[b['source']]
                assert abs(b['regret']-max(0.,truth))<1e-9
            assert sum(b['optimal_choice'] for b in details['branches'])==result['decisions']['optimal_branch_choices']
            unique_episodes={}
            for e in details['episodes']:
                rows=np.array(e['rows'],dtype=int);assert e['status']=='success'
                assert src[rows[0]]==e['start'] and np.array_equal(dst[rows[:-1]],src[rows[1:]])
                assert ref[dst[rows[-1]]]==0
                assert abs(float(cost[rows].sum())-e['cost'])<1e-9
                assert abs(e['regret']-(e['cost']-ref[e['start']]))<1e-9
                delta=float((cost[rows]+ref[dst[rows]]-ref[src[rows]]).sum())
                err=abs(delta-e['regret']);assert err<1e-8;max_regret_identity=max(max_regret_identity,err)
                if e['start'] in unique_episodes:assert unique_episodes[e['start']]['rows']==e['rows']
                unique_episodes[e['start']]=e
            assert len(unique_episodes)==165 and len(details['episodes'])==307
            assert abs(np.mean([e['regret'] for e in unique_episodes.values()])-result['decisions']['unique_start_rollouts']['mean_successful_regret'])<1e-9
    files=[root/'protocol.json',root/'validation.json',root/'runtime.json',root.parent/'tests.log',root.parent/'integration_smoke.json',
           out/'summary.json',out/'report.md',out/'arrays.npz',out/'comparison.png',out/'analysis_source.py']
    for arm in ('R0','R1'):
        files += [root/arm/name for name in ('initial.pt','phr.pt','final.pt','COMPLETE','config.json','training.jsonl')]
    manifest=dict(passed=True,official_endpoint=50000,no_extra_training=True,
        frozen_training_sources_unchanged=True,independent_task_statistics=True,
        independent_greedy_choices=True,independent_path_costs_and_regret=True,
        decomposition_identity_max_error=max_identity,path_advantage_regret_identity_max_error=max_regret_identity,
        artifacts={str(p.relative_to(root.parent)):file_digest(p) for p in files},
        validator_sha256=file_digest(__file__),pairing=summary['pairing'])
    write(root/'final_integrity.json',manifest)
    write(root/'COMPLETE',dict(passed=True,final_integrity_sha256=file_digest(root/'final_integrity.json'),official_endpoint=50000))
    print(json.dumps({k:v for k,v in manifest.items() if k!='artifacts'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
