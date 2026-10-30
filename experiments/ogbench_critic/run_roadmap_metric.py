"""Frozen data-support metric pilot; no neural training or evaluation geometry input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import ROOT,BASE,BASE_SHA,load_data,sha256
from data_graph import valid_transitions
from roadmap_metric import RoadmapMetric
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def run(root,output):
    started=time.monotonic();deadline=started+240;phase='setup';env=engine=None
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot';snapshot.mkdir()
    files=[Path(__file__).with_name(n) for n in ['ROADMAP_METRIC_PROTOCOL.md','roadmap_metric.py','test_roadmap_metric.py','run_roadmap_metric.py','common.py','data_graph.py']]
    files += [ROOT/'experiments/ogbench_diagnostics'/n for n in ['fast_pointmaze.py','core.py']]
    for path in files:shutil.copy2(path,snapshot/path.name)
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('240-second roadmap metric pilot exhausted; no extension')
    try:
        assert sha256(BASE/'params_100000.pkl')==BASE_SHA
        verified=read(root/'cached_pointmaze_verification/verification.json')
        assert verified['passed'] and len(verified['cases'])==25
        for name in ['fast_pointmaze.py','core.py']:
            assert sha256(ROOT/'experiments/ogbench_diagnostics'/name)==verified['source_sha256'][name]
        config=read(BASE/'flags.json')['agent'];env,train,_=load_data(config)
        phase='graph';source,target=valid_transitions(train.dataset)
        model=RoadmapMetric.build(source,target,connections=8);model.save(output/'graph.npz')
        catalog=np.concatenate([source,target]);np.testing.assert_array_equal(model.nodes,catalog[model.prototype_indices])
        sample_hash=hashlib.sha256(source.tobytes()+target.tobytes()).hexdigest()
        protocol=dict(training_updates=0,time_limit_seconds=240,spacing=.2,connections=8,
            source_transitions=len(source),nodes=len(model.nodes),undirected_edges=len(model.edges),
            source_endpoint_sha256=sample_hash,graph_sha256=sha256(output/'graph.npz'),
            edge_cost='prototype L_inf distance / 0.2; symmetric local-reversibility prior',
            query_correction='half maximum pair consistency deficit; no tuning',
            source_sha256={p.name:sha256(p) for p in snapshot.iterdir()},baseline_checkpoint_sha256=BASE_SHA)
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        def progress(rows):
            (output/'distance_progress.json').write_text(json.dumps(dict(completed_rows=rows,total_rows=len(model.nodes),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
        model.precompute(output/'distances.npy',budget=budget,progress=progress)
        protocol['distance_sha256']=sha256(output/'distances.npy');protocol['build_seconds']=time.monotonic()-started
        probe=np.linspace(0,len(model.nodes)-1,32,dtype=int);distance=np.asarray(model.distances[np.ix_(probe,probe)])
        for k in range(32):assert np.all(distance<=distance[:,k,None]+distance[None,k,:]+1e-8)
        restored=RoadmapMetric.load(output/'graph.npz',output/'distances.npy')
        query=source[np.linspace(0,len(source)-1,32,dtype=int)].astype(np.float64)
        np.testing.assert_array_equal(model(query,query[::-1]),restored(query,query[::-1]))
        print('GRAPH_READY',protocol,flush=True)
        # Graph and all distances are complete before reading any evaluation case.
        phase='evaluation';cases=core.read_bank(root/'evaluation_bank.npz');assert len(cases)==25
        protocol['bank_sha256']=sha256(root/'evaluation_bank.npz')
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        engine=FastPointmaze(env,nthread=4);greedy=output/'greedy';greedy.mkdir();results={};query_stats={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return model(observations,goals)
            info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
            trace['state_distances']=model(trace['observations'],np.broadcast_to(case['goal'],trace['observations'].shape))
            changes=np.diff(trace['state_distances'])
            info['strict_distance_decrease_fraction']=float((changes<-1e-8).mean())
            info['distance_increase_fraction']=float((changes>1e-8).mean())
            for label,ob in [('start',case['observation']),('goal',case['goal'])]:
                ids,costs,correction=model.attachments(ob[None]);query_stats[f'{name}_{label}']=dict(correction=float(correction[0]),
                    prototype_indices=ids[0].tolist(),attachment_costs=costs[0].tolist())
            np.savez_compressed(greedy/f'{name}.npz',**trace);results[name]=info
            (greedy/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,info,flush=True)
        (greedy/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        phase='fields';budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as old:np.testing.assert_array_equal(nodes,old['nodes'])
        fields={};failure_endpoints={}
        targets=[('canonical',core.GOAL,core.ANCHOR)]+[(name,c['goal'],c['observation']) for name,c in cases.items() if name.startswith('task3_')]
        for name,goal,anchor in targets:
            budget();values=model(nodes,np.broadcast_to(goal,nodes.shape))
            stats,field=core.graph_metrics(nodes,graph,values,goal=goal,anchor=anchor);fields[name]=stats
            np.savez_compressed(output/f'{name}_field.npz',**{**field,'walls':walls,'goal':goal})
            if name!='canonical' and not results[name]['success']:
                failure_endpoints[name]=core.graph_metrics(nodes,graph,values,goal=goal,anchor=np.array(results[name]['final_xy']))[0]
        base=read(root/'baseline_greedy/summary.json')['cases']
        teacher_gate=all(results[n]['success'] for n in results if n.startswith('task3_')) and all(results[n]['success'] for n in base if base[n]['success'])
        budget();assert sha256(BASE/'params_100000.pkl')==BASE_SHA
        result=dict(complete=True,protocol=protocol,cases=results,fields=fields,failure_endpoints=failure_endpoints,
            query_statistics=query_stats,eligible_for_neural_study=teacher_gate,elapsed_seconds=time.monotonic()-started,
            queries=model.calls)
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',result['elapsed_seconds'],'TEACHER_GATE',teacher_gate,flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),phase=phase,elapsed_seconds=time.monotonic()-started,
            traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.root,args.output)
