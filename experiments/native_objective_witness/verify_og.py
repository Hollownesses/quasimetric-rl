"""Check custom population terms against official losses and all actual witness traces."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

def run(raw,out):
    repo=Path(__file__).resolve().parents[2];sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw),str(repo)]
    from common import load_base,load_data,tree_equal
    from utils.flax_utils import restore_agent
    from experiments.ogbench_diagnostics import core
    from experiments.native_margin.shared import sha,write
    from experiments.native_objective_witness.core import coordinate_scale,risk,hoeffding_halfwidth
    import jax
    base,cfg=load_base();env,train,_=load_data(cfg);env.close()
    g=raw/'results/ogbench_critic_20261029';audit=json.loads((out/'audit.json').read_text());points=audit['points']
    pop=np.load(out/'populations.npz');models={};ext={};proof={}
    for name in ['original','E16']:
        source=base if name=='original' else restore_agent(base,str(g/'hard_goal_1000/model'),101000)
        ex=restore_agent(base,str(g/('extraction_1000/original/actor' if name=='original' else 'extraction_1000/improved/actor')),102000 if name=='original' else 103000)
        boundary=restore_agent(source,str(out/(name+'_boundary')),'witness')
        alpha=points[name+'_boundary']['alpha']
        assert tree_equal(boundary.network.params,coordinate_scale(source.network.params,alpha,cfg['latent_dim']))
        assert tree_equal(boundary.network.opt_state,source.network.opt_state) and tree_equal(boundary.rng,source.rng)
        models[name+'_raw']=source;models[name+'_boundary']=boundary
        ext[name+'_raw']=ex.network.params['modules_dynamics']
        ext[name+'_boundary']=coordinate_scale({**source.network.params,'modules_dynamics':ext[name+'_raw']},alpha,cfg['latent_dim'])['modules_dynamics']
        assert sha(out/(name+'_boundary')/'params_witness.pkl')==points[name+'_boundary']['checkpoint_sha256']
    @jax.jit
    def official(agent,batch):
        _,info=agent.value_loss(batch,agent.network.params)
        d,_=agent.dynamics_loss(batch,agent.network.params)
        return info['d_neg_loss'],info['d_pos_loss'],d
    ii=np.flatnonzero(train.dataset['valids']);assert len(ii)==1000000
    for point,a in models.items():
        d=pop[point+'_distance'];assert len(d)==1000000
        np.testing.assert_allclose(risk(d),points[point]['risk'],atol=1e-14,rtol=0)
        if point.endswith('boundary'):assert risk(d)<=cfg['eps']+1e-6
        for kind in ['raw','extracted']:
            expected=pop[point+'_D_'+kind]
            np.testing.assert_allclose(expected.mean(),points[point]['D_'+kind],atol=1e-14,rtol=0)
            agent=a if kind=='raw' else a.replace(network=a.network.replace(params={**a.network.params,'modules_dynamics':ext[point]}))
            max_error=0.
            for k in [0,127,511,975]:
                begin=k*1024;end=min(begin+1024,len(ii));batch=train.dataset.get_subset(ii[begin:end])
                batch['value_goals']=batch['next_observations']
                _,r,dy=official(agent,batch)
                error=abs(float(dy)-float(expected[begin:end].mean()));max_error=max(max_error,error)
                np.testing.assert_allclose(float(dy),expected[begin:end].mean(),atol=1e-4,rtol=2e-4)
                np.testing.assert_allclose(float(r),risk(d[begin:end]),atol=1e-5,rtol=2e-4)
            proof[point+'_'+kind]=dict(official_loss_max_difference=max_error)
    with (out/'queries.csv').open() as f:queries=list(csv.DictReader(f))
    assert len(queries)==4096 and len({(r['batch'],r['point']) for r in queries})==4096
    for point in models:
        vv=np.array([float(r['P']) for r in queries if r['point']==point])
        assert len(vv)==1024
        np.testing.assert_allclose(vv[512:].mean(),points[point]['P'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(vv.mean(),points[point]['P_all1024'],atol=1e-12,rtol=0)
    np.random.seed(4400)
    for b in range(4):
        batch=train.sample(1024)
        for point,a in models.items():
            p,_,_=official(a,batch)
            cached=next(float(r['P']) for r in queries if r['point']==point and int(r['batch'])==b)
            np.testing.assert_allclose(float(p),cached,atol=1e-4,rtol=1e-6)
    comparisons=audit['comparisons'];half=hoeffding_halfwidth(512*1024)
    for r in comparisons:
        a,b=points['original_'+r['scale']],points['E16_'+r['scale']]
        expected=b['P']-a['P']+b['D_'+r['dynamics']]-a['D_'+r['dynamics']]
        np.testing.assert_allclose(r['delta_primal'],expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(r['hoeffding_upper']-r['delta_primal'],half,atol=1e-12,rtol=0)
    bank=core.read_bank(g/'official_counterfactual/official_task3_bank.npz')
    with (out/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
    assert len(episodes)==100 and len({(r['critic'],r['case']) for r in episodes})==100
    traces={}
    for r in episodes:
        p=out/(r['critic']+'_boundary')/'greedy'/(r['case']+'.npz');tr=np.load(p)
        assert len(tr['actions'])==int(r['steps']) and int(r['steps'])<=1000
        assert len(tr['observations'])==len(tr['actions'])+1 and 'final_integration' in tr
        np.testing.assert_array_equal(tr['observations'][0],bank[r['case']]['observation'])
        np.testing.assert_array_equal(tr['goal'],bank[r['case']]['goal'])
        assert bool(np.linalg.norm(tr['observations'][-1]-tr['goal'])<=1)==bool(int(r['success']))
        traces[str(p.relative_to(out))]=sha(p)
    for p,digest in audit['inputs'].items():assert sha(raw/p)==digest
    write(out/'verification.json',dict(official_terms=proof,points=4,dynamics_contexts=2,full_population_rows=1000000,
        total_queries=1048576,new_queries=524288,independent_official_loss_checks=True,full_traces=100,
        traces=traces,coordinate_witnesses_reloaded=True,training_updates=0))
    print('VERIFIED OG original losses, four candidate points, and 100 complete new episodes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
