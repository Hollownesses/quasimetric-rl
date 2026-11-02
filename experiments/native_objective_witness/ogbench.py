"""Feasible native value objectives and exact full-population dynamics costs."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np

def run(raw,out):
    out.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw),str(repo)]
    from common import load_base,load_data,distances,tree_digest,tree_equal
    from utils.flax_utils import restore_agent,save_agent
    from exact_score_cache import ExactScoreCache
    from experiments.ogbench_diagnostics import core
    from experiments.ogbench_diagnostics.fast_pointmaze import FastPointmaze
    from experiments.native_margin.shared import rows,write,sha
    from experiments.native_objective_witness.core import mean_boundary,risk,push_loss,hoeffding_halfwidth,coordinate_scale
    import jax
    import jax.numpy as jnp
    base,cfg=load_base();base_digest=tree_digest(base);env,train,_=load_data(cfg)
    g=raw/'results/ogbench_critic_20261029'
    manifest=json.loads((repo/'results/unified_mechanism_20261102/audit.json').read_text())['verified_sha256']
    sources={'original':(None,g/'extraction_1000/original/actor',102000),
             'E16':(g/'hard_goal_1000/model',g/'extraction_1000/improved/actor',103000)}
    models={};extracted={};input_hashes={}
    for name,(path,ex,step) in sources.items():
        a=base if path is None else restore_agent(base,str(path),101000)
        b=restore_agent(base,str(ex),step);assert tree_equal(a.network.params['modules_value'],b.network.params['modules_value'])
        models[name]=a;extracted[name]=b.network.params['modules_dynamics']
        candidates=[ex/f'params_{step}.pkl']+([] if path is None else [path/'params_101000.pkl'])
        for p in candidates:
            digest=sha(p);assert digest==manifest[str(p.relative_to(raw))];input_hashes[str(p.relative_to(raw))]=digest
    frozen={k:tree_digest(v) for k,v in models.items()}
    @jax.jit
    def transition_terms(agent,other_dynamics,s,sn,actions):
        value=agent.network.select('value');d,z,zn=value(s,sn,info=True)
        zs=jnp.concatenate([z,actions],axis=-1)
        pred=z+agent.network.select('dynamics')(zs)
        pred2=z+agent.network.select('dynamics')(zs,params={**agent.network.params,'modules_dynamics':other_dynamics})
        loss=lambda p:(value(zn,p,is_phi=True)+value(p,zn,is_phi=True))/2
        return d,loss(pred),loss(pred2)
    np.random.seed(4400);smoke=train.sample(1024)
    # The scalar primal below is the one generating native value/dynamics gradients.
    def surrogate(p):
        _,info=base.value_loss(smoke,p);d,_=base.dynamics_loss(smoke,p)
        return info['d_neg_loss']+jax.lax.stop_gradient(info['lam'])*info['d_pos_loss']+d
    gp=jax.jit(jax.grad(surrogate))(base.network.params)
    gn=jax.grad(lambda p:base.total_loss(smoke,p)[0])(base.network.params)
    ga=jax.grad(lambda p:base.actor_loss(smoke,p,base.rng)[0])(base.network.params)
    grad_checks={}
    for key in ['modules_value','modules_dynamics']:
        aa=np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(gp[key])]).astype(float)
        bb=np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(gn[key])]).astype(float)
        rel=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))
        print('gradient smoke',key,'relative_l2',rel,'max_error',float(abs(aa-bb).max()),flush=True)
        assert rel<2e-5
        assert all(np.max(abs(np.asarray(x)))==0 for x in jax.tree_util.tree_leaves(ga[key]))
        grad_checks[key]=dict(relative_l2=rel,max_error=float(abs(aa-bb).max()),actor_gradient_zero=True)
    dd,dy,_=transition_terms(base,extracted['original'],smoke['observations'],smoke['next_observations'],smoke['actions'])
    np.testing.assert_allclose(np.asarray(dy).mean(),float(base.dynamics_loss(smoke,base.network.params)[0]),atol=1e-5,rtol=1e-5)
    coordinate_smoke={}
    for name,a in models.items():
        for kind,dyn in [('raw',a.network.params['modules_dynamics']),('extracted',extracted[name])]:
            source=a.replace(network=a.network.replace(params={**a.network.params,'modules_dynamics':dyn}))
            scaled=source.replace(network=source.network.replace(params=coordinate_scale(source.network.params,1.25,cfg['latent_dim'])))
            def latent_prediction(aa):
                _,z,zn=aa.network.select('value')(smoke['observations'],smoke['next_observations'],info=True)
                pred=z+aa.network.select('dynamics')(jnp.concatenate([z,smoke['actions']],axis=-1))
                return z,pred
            z,p=latent_prediction(source);zz,pp=latent_prediction(scaled)
            z,p,zz,pp=(np.asarray(v).astype(float) for v in (z,p,zz,pp))
            rz=float(np.linalg.norm(zz-1.25*z)/np.linalg.norm(1.25*z))
            rp=float(np.linalg.norm(pp-1.25*p)/np.linalg.norm(1.25*p))
            assert rz<1e-4 and rp<1e-4
            oldd=float(source.dynamics_loss(smoke,source.network.params)[0]);newd=float(scaled.dynamics_loss(smoke,scaled.network.params)[0])
            coordinate_smoke[name+'_'+kind]=dict(latent_relative_l2=rz,prediction_relative_l2=rp,D_old=oldd,D_new=newd,D_expected=1.25*oldd,D_absolute_error=abs(newd-1.25*oldd))
    print('coordinate smoke',json.dumps(coordinate_smoke),flush=True)
    write(out/'smoke.json',dict(gradient=grad_checks,coordinate=coordinate_smoke))
    ii=np.flatnonzero(train.dataset['valids']);assert len(ii)==1000000
    obs=train.dataset['observations'];actions=train.dataset['actions']
    x,y,act=(np.asarray(v,np.float32) for v in (obs[ii],obs[ii+1],actions[ii]))
    data_digest=hashlib.sha256(x.tobytes()+y.tobytes()+act.tobytes()).hexdigest()
    kernel_digest=hashlib.sha256(inspect.getsource(transition_terms.__wrapped__).encode()).hexdigest()
    def scan(agent,other,label):
        xx,yy,ac=(np.pad(z,((0,(-len(x))%1024),(0,0)),mode='edge') for z in (x,y,act))
        cache_dir=out/'scan_cache'/label;cache_dir.mkdir(parents=True,exist_ok=True)
        signature=dict(params=tree_digest(agent.network.params),other_dynamics=tree_digest(other),data=data_digest,kernel=kernel_digest,
                       rows=len(x),batch=1024,jax=jax.__version__,x64=bool(jax.config.x64_enabled))
        result=[[],[],[]]
        for start in range(0,len(x),128*1024):
            stop=min(len(x),start+128*1024);file=cache_dir/f'{start}.npz';meta=cache_dir/f'{start}.json'
            if file.exists() or meta.exists():
                saved=json.loads(meta.read_text());assert saved['signature']==signature and saved['sha256']==sha(file)
                with np.load(file) as f:values=[f[k] for k in ['distance','D_raw','D_extracted']]
            else:
                block=[[],[],[]]
                for k in range(start,stop,1024):
                    values=transition_terms(agent,other,xx[k:k+1024],yy[k:k+1024],ac[k:k+1024])
                    for target,value in zip(block,values):target.append(np.asarray(value).astype(float))
                values=[np.concatenate(v)[:stop-start] for v in block]
                np.savez_compressed(file,**dict(zip(['distance','D_raw','D_extracted'],values)))
                write(meta,dict(signature=signature,sha256=sha(file)))
            assert all(len(v)==stop-start for v in values)
            for target,value in zip(result,values):target.append(value)
            print('witness scan',label,stop,flush=True)
        return [np.concatenate(z)[:len(x)] for z in result]
    points={};cache_arrays={};boundary_models={};boundary_ex={};alphas={}
    for name,a in models.items():
        d0,D0,De=scan(a,extracted[name],name+'_raw')
        np.savez_compressed(out/(name+'_raw_population.npz'),distance=d0,D_raw=D0,D_extracted=De)
        lo,hi,alpha=mean_boundary(d0,cfg['eps']);alphas[name]=alpha
        scaled_params=coordinate_scale(a.network.params,alpha,cfg['latent_dim'])
        other_params=coordinate_scale({**a.network.params,'modules_dynamics':extracted[name]},alpha,cfg['latent_dim'])
        assert tree_equal(scaled_params['modules_value'],other_params['modules_value'])
        b=a.replace(network=a.network.replace(params=scaled_params));other=other_params['modules_dynamics']
        assert tree_equal(b.network.opt_state,a.network.opt_state) and tree_equal(b.rng,a.rng)
        for key in ['modules_actor','modules_lam']:assert tree_equal(b.network.params[key],a.network.params[key])
        assert all(xx.shape==yy.shape for xx,yy in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
        d1,D1,De1=scan(b,other,name+'_boundary');assert risk(d1)<=cfg['eps']+1e-6
        np.savez_compressed(out/(name+'_boundary_population.npz'),distance=d1,D_raw=D1,D_extracted=De1)
        homo={key:dict(max_error=float(abs(new-alpha*old).max()),relative_l2=float(np.linalg.norm(new-alpha*old)/max(np.linalg.norm(alpha*old),1e-30))) for key,new,old in [('d',d1,d0),('D_raw',D1,D0),('D_extracted',De1,De)]}
        # Near-zero reconstruction losses are ill-conditioned relative checks.
        # The actual risk and every actual D are evaluated above, never replaced by a*D.
        assert all(np.isfinite(v).all() for v in [d1,D1,De1])
        print('population homogeneity',name,json.dumps(homo),flush=True)
        for scale,d,dr,de in [('raw',d0,D0,De),('boundary',d1,D1,De1)]:
            key=name+'_'+scale
            points[key]=dict(critic=name,scale=scale,alpha=1. if scale=='raw' else alpha,alpha_lower=lo,alpha_upper=hi,
                risk=risk(d),feasible=int(risk(d)<=cfg['eps']+1e-6),D_raw=float(dr.mean()),D_extracted=float(de.mean()),
                homogeneity=homo if scale=='boundary' else None)
            for suffix,v in [('distance',d),('D_raw',dr),('D_extracted',de)]:cache_arrays[key+'_'+suffix]=v
        boundary_models[name]=b;boundary_ex[name]=other
        dest=out/(name+'_boundary');dest.mkdir();save_agent(b,str(dest),'witness')
        reloaded=restore_agent(a,str(dest),'witness');assert tree_equal(b,reloaded)
        points[name+'_boundary']['checkpoint_sha256']=sha(dest/'params_witness.pkl')
    np.savez_compressed(out/'populations.npz',**cache_arrays)
    write(out/'population_points.json',points)
    # Independent native query audit; every point sees the exact same batch once.
    query_models={**{k+'_raw':v for k,v in models.items()},**{k+'_boundary':v for k,v in boundary_models.items()}}
    np.random.seed(4400);h3=hashlib.sha256();full=hashlib.sha256();query_rows=[];first32=None
    for batch_id in range(1024):
        batch=train.sample(1024)
        for key in ['observations','next_observations','value_goals']:h3.update(batch[key].tobytes())
        for key in sorted(batch):full.update(key.encode());full.update(np.asarray(batch[key]).tobytes())
        if batch_id==31:first32=h3.hexdigest()
        for key,a in query_models.items():
            j=np.asarray(distances(a,batch['observations'],batch['value_goals'])).astype(float);assert j.min()>=0
            pp=push_loss(j)
            query_rows.append(dict(batch=batch_id,point=key,P=float(pp.mean()),J_mean=float(j.mean()),J_min=float(j.min()),J_max=float(j.max())))
        if (batch_id+1)%32==0:
            rows(out/'queries.csv',query_rows);print('witness queries',batch_id+1,flush=True)
    expected=json.loads((repo/'results/native_margin_20261102_verified/og/audit.json').read_text())['stream_sha256'];assert first32==expected
    half=hoeffding_halfwidth(512*1024);comparisons=[]
    for key,p in points.items():
        all_v=np.array([r['P'] for r in query_rows if r['point']==key]);p['P_all1024']=float(all_v.mean())
        p['P']=float(all_v[512:].mean())
        p['P_plus_D_raw']=p['P']+p['D_raw'];p['P_plus_D_extracted']=p['P']+p['D_extracted']
    for scale in ['raw','boundary']:
        a,b=points['original_'+scale],points['E16_'+scale]
        da=np.array([r['P'] for r in query_rows if r['point']=='original_'+scale])[512:]
        db=np.array([r['P'] for r in query_rows if r['point']=='E16_'+scale])[512:];delta=db-da
        se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        for dyn in ['raw','extracted']:
            d=float(delta.mean()+b['D_'+dyn]-a['D_'+dyn])
            comparisons.append(dict(scale=scale,dynamics=dyn,both_feasible=int(a['feasible'] and b['feasible']),
                delta_P=float(delta.mean()),delta_D=b['D_'+dyn]-a['D_'+dyn],delta_primal=d,
                batch_SE=se,normal_lower=d-1.96*se,normal_upper=d+1.96*se,
                hoeffding_lower=d-half,hoeffding_upper=d+half))
    rows(out/'points.csv',[dict(point=k,**v) for k,v in points.items()]);rows(out/'comparisons.csv',comparisons)
    write(out/'query_audit.json',dict(batches=1024,batch_size=1024,queries=1048576,new_query_batches=512,new_queries=524288,three_field_sha256=h3.hexdigest(),
        full_field_sha256=full.hexdigest(),first32_exact=True,hoeffding_halfwidth=half,gradient_checks=grad_checks))
    # Actual FP32 boundary control, after all feasibility/objective values are fixed.
    bank=core.read_bank(g/'official_counterfactual/official_task3_bank.npz');engine=FastPointmaze(env,nthread=4)
    old=json.loads((g/'official_counterfactual_completion/analysis.json').read_text())['initials']
    episodes=[];first=[];controller_audits={}
    for name,a in boundary_models.items():
        dest=out/(name+'_boundary')/'greedy';dest.mkdir();cache=ExactScoreCache(a,audit=True)
        for case_name,case in bank.items():
            core.load_case(env,case);xy,_=engine.candidates(env)
            p=np.asarray(distances(models[name],xy,np.broadcast_to(case['goal'],xy.shape))).astype(float)
            q=np.asarray(distances(a,xy,np.broadcast_to(case['goal'],xy.shape))).astype(float)
            first.append(dict(critic=name,case=case_name,raw_action=int(np.argmin(p)),boundary_action=int(np.argmin(q)),homogeneity_max_error=float(abs(q-alphas[name]*p).max())))
            info,tr=engine.rollout(case,cache);tr['final_integration']=core.save_integration(env)
            np.testing.assert_array_equal(tr['observations'][0],case['observation']);assert len(tr['actions'])<=1000
            assert bool(info['success'])==bool(np.linalg.norm(tr['observations'][-1]-case['goal'])<=1)
            np.savez_compressed(dest/(case_name+'.npz'),**tr)
            parent=old['original' if name=='original' else 'improved'][case_name]
            episodes.append(dict(critic=name,case=case_name,success=int(info['success']),raw_success=int(parent['success']),steps=info['steps'],raw_steps=parent['steps']))
            rows(out/'episodes.csv',episodes)
            if len(episodes)%5==0:print('witness OG control',name,case_name,flush=True)
        controller_audits[name]=cache.statistics()
    engine.close();env.close();rows(out/'first.csv',first)
    assert tree_digest(base)==base_digest and all(tree_digest(v)==frozen[k] for k,v in models.items())
    for rel,digest in input_hashes.items():assert sha(raw/rel)==digest
    write(out/'audit.json',dict(inputs=input_hashes,training_updates=0,gradient_checks=grad_checks,scales=alphas,
        population_rows=len(ii),query_rows=1048576,points=points,comparisons=comparisons,controller=controller_audits,
        original_agents_unchanged=True,sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
