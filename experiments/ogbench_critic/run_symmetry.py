"""Read-only direction decomposition and fixed symmetric-critic execution."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT,BASE,BASE_SHA,load_base,load_data,distances,tree_digest,sha256
from symmetry import SymmetricScore,components,paired_statistics
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import jax
import numpy as np
from scipy.spatial import cKDTree


def read(path):return json.loads(Path(path).read_text())


def run(root,output):
    started=time.monotonic();deadline=started+240;phase='setup';env=engine=None
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__).with_name(n) for n in ['SYMMETRY_PROTOCOL.md','symmetry.py','test_symmetry.py','run_symmetry.py','common.py','exact_score_cache.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('240-second symmetry diagnostic exhausted; no extension')
    try:
        fast=read(root/'cached_pointmaze_verification/verification.json')
        assert fast['passed'] and len(fast['cases'])==25 and len(fast['candidates'])==35
        for name,digest in fast['source_sha256'].items():
            directory=Path(__file__).parent if name in ['exact_score_cache.py','test_exact_score_cache.py'] else ROOT/'experiments/ogbench_diagnostics'
            assert sha256(directory/name)==digest
        agent,config=load_base();initial=tree_digest(agent);env,_,val=load_data(config)
        assert config['value_p_randomgoal']==1 and int(agent.network.opt_state[0].count)==100000
        bank=root/'evaluation_bank.npz';cases=core.read_bank(bank);assert len(cases)==25
        protocol=dict(training_updates=0,checkpoint_sha256=BASE_SHA,checkpoint=str(BASE/'params_100000.pkl'),
            epoch=100000,coefficient=.5,primary='symmetric',auxiliary_fields=['forward','reverse','antisymmetric'],
            bank_sha256=sha256(bank),baseline_summary_sha256=sha256(root/'baseline_greedy/summary.json'),
            config=config,time_limit_seconds=240,devices=[str(d) for d in jax.devices()],
            fast_verification_sha256=sha256(root/'cached_pointmaze_verification/verification.json'),
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()})
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        phase='validation';np.random.seed(9001);batch=val.sample(2048)
        assert tree_digest(batch)==read(root/'tail_1000/protocol.json')['validation_sha256']
        f=np.asarray(distances(agent,batch['observations'],batch['value_goals']))
        r=np.asarray(distances(agent,batch['value_goals'],batch['observations']))
        pf=np.asarray(distances(agent,batch['observations'],batch['next_observations']))
        pr=np.asarray(distances(agent,batch['next_observations'],batch['observations']))
        validation=paired_statistics(f,r,pf,pr)
        assert validation['symmetric_push']<=validation['paired_push']+1e-10
        assert validation['symmetric_pair_risk']<=validation['paired_pair_risk']+1e-10
        assert all(np.isfinite(v) for v in validation.values())
        np.savez_compressed(output/'validation.npz',forward=f,reverse=r,pair_forward=pf,pair_reverse=pr)
        print('VALIDATION',validation,flush=True)
        phase='fields';budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as old:
            np.testing.assert_array_equal(nodes,old['nodes']);old_path=old['path'].copy();old_values=old['values'].copy()
        targets=[('canonical',core.GOAL,core.ANCHOR)]
        targets += [(name,case['goal'],case['observation']) for name,case in cases.items() if name.startswith('task3_')]
        fields={};decomposition={};field_values={}
        for name,goal,anchor in targets:
            budget();goals=np.broadcast_to(goal,nodes.shape)
            values=components(np.asarray(distances(agent,nodes,goals)),np.asarray(distances(agent,goals,nodes)))
            np.testing.assert_allclose(values['forward'],values['symmetric']+values['antisymmetric'],rtol=1e-6,atol=1e-5)
            np.savez_compressed(output/f'{name}_field.npz',nodes=nodes,walls=walls,goal=goal,**values)
            stats={}
            for kind in ['forward','reverse','symmetric']:
                stats[kind]=core.graph_metrics(nodes,graph,values[kind],goal=goal,anchor=anchor)[0]
            fields[name]=stats;field_values[name]=values['symmetric']
            if name=='canonical':
                np.testing.assert_allclose(values['forward'],old_values,rtol=2e-6,atol=2e-5)
                idx=cKDTree(nodes).query(old_path)[1];np.savez_compressed(output/'canonical_route.npz',path=old_path,**{k:v[idx] for k,v in values.items()})
                decomposition=dict(probe_nodes=old_path[[0,15]].tolist(),
                    delta={k:float(v[idx[15]])-float(v[idx[0]]) for k,v in values.items()})
            print('FIELD',name,{k:dict(descent=v['strict_descent_from_anchor'],peak_over_iqr=v['minimum_peak_gap_over_iqr']) for k,v in stats.items()},flush=True)
        (output/'field_statistics.json').write_text(json.dumps(fields,indent=2)+'\n')
        phase='evaluation';engine=FastPointmaze(env,nthread=4);scorer=SymmetricScore(agent)
        greedy=output/'greedy';greedy.mkdir();results={};failure_endpoints={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return scorer(observations,goals)
            info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
            np.savez_compressed(greedy/f'{name}.npz',**trace);results[name]=info
            (greedy/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,info,flush=True)
            if name.startswith('task3_') and not info['success']:
                budget();failure_endpoints[name]=core.graph_metrics(nodes,graph,field_values[name],goal=case['goal'],anchor=trace['observations'][-1])[0]
        (greedy/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        budget();assert tree_digest(agent)==initial and sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,protocol=protocol,validation=validation,fields=fields,decomposition=decomposition,
            cases=results,failure_endpoints=failure_endpoints,cache=scorer.cache.statistics(),elapsed_seconds=time.monotonic()-started)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),phase=phase,elapsed_seconds=time.monotonic()-started,
            traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
