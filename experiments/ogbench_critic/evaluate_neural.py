"""Independent-process original-QRL neural scoring only; no teacher imports."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from common import ROOT,load_base,load_data,distances,tree_digest,sha256
from exact_score_cache import ExactScoreCache
sys.path.insert(0,str(ROOT))
from experiments.ogbench_diagnostics import core
from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
import numpy as np
from utils.flax_utils import restore_agent


def run(root,checkpoint,epoch,output,limit):
    started=time.monotonic();deadline=started+limit;root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);env=engine=None
    def budget():
        if time.monotonic()>deadline:raise TimeoutError('Neural-only evaluation deadline exhausted')
    try:
        assert epoch==101000 and 'roadmap_metric' not in sys.modules and 'data_graph' not in sys.modules
        verified=json.loads((root/'cached_pointmaze_verification/verification.json').read_text())
        assert verified['passed'] and len(verified['cases'])==25
        for name in ['core.py','fast_pointmaze.py']:
            assert sha256(ROOT/'experiments/ogbench_diagnostics'/name)==verified['source_sha256'][name]
        assert sha256(Path(__file__).with_name('exact_score_cache.py'))==verified['source_sha256']['exact_score_cache.py']
        base,config=load_base();agent=restore_agent(base,str(checkpoint),epoch)
        assert int(agent.network.opt_state[0].count)==epoch and int(agent.network.step)==epoch+1
        frozen=tree_digest(agent);env,_,_=load_data(config)
        cases=core.read_bank(root/'evaluation_bank.npz');assert len(cases)==25
        cache=ExactScoreCache(agent,audit=True);engine=FastPointmaze(env,nthread=4);results={}
        for name,case in cases.items():
            budget()
            def score(observations,goals):budget();return cache(observations,goals)
            info,trace=engine.rollout(case,score);trace['final_integration']=core.save_integration(env)
            trace['state_values']=np.asarray(distances(agent,trace['observations'],np.broadcast_to(case['goal'],trace['observations'].shape)))
            change=np.diff(trace['state_values'])
            info['strict_decrease_fraction']=float((change<-1e-4).mean());info['increase_fraction']=float((change>1e-4).mean())
            np.savez_compressed(output/f'{name}.npz',**trace);results[name]=info
            (output/'progress.json').write_text(json.dumps(results,indent=2)+'\n');print(name,info,flush=True)
        budget();walls,radius=core.geometry(env);nodes,graph=core.build_graph(walls,radius,.2)
        with np.load(root/'baseline_field/field.npz') as old:np.testing.assert_array_equal(nodes,old['nodes'])
        values=np.asarray(distances(agent,nodes,np.broadcast_to(core.GOAL,nodes.shape)))
        field_stats,field=core.graph_metrics(nodes,graph,values)
        np.savez_compressed(output/'field.npz',**{**field,'walls':walls})
        budget();assert tree_digest(agent)==frozen
        assert 'roadmap_metric' not in sys.modules and 'data_graph' not in sys.modules
        result=dict(complete=True,epoch=epoch,checkpoint_sha256=sha256(Path(checkpoint)/f'params_{epoch}.pkl'),
            only_neural_scores=True,roadmap_module_imported=False,data_graph_module_imported=False,candidate_evaluator_source_verified=True,
            controller='original forward QRLAgent value; no teacher and no symmetry wrapper',cases=results,field=field_stats,
            cache=cache.statistics(),bank_sha256=sha256(root/'evaluation_bank.npz'),elapsed_seconds=time.monotonic()-started,
            source_sha256={Path(__file__).name:sha256(__file__),'common.py':sha256(Path(__file__).with_name('common.py')),
                'exact_score_cache.py':sha256(Path(__file__).with_name('exact_score_cache.py'))})
        (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('NEURAL_COMPLETE',result['elapsed_seconds'],flush=True)
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(error),elapsed_seconds=time.monotonic()-started,
            traceback=traceback.format_exc(),complete=False),indent=2)+'\n');raise
    finally:
        if engine is not None:engine.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--epoch',type=int,required=True);parser.add_argument('--output',required=True);parser.add_argument('--limit-seconds',type=float,required=True)
    args=parser.parse_args();run(args.root,args.checkpoint,args.epoch,args.output,args.limit_seconds)
