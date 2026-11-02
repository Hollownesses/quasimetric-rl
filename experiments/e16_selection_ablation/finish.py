"""Read-only paired analysis and artifact audit; no parameter updates."""
import argparse
import csv
import importlib.metadata
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from support import HERE,RAW,G,REPO,UNIFIED,read,write,sha,check
from common import BASE,load_base,tree_equal,tree_digest
from hard_goal import pool_statistics
from utils.flax_utils import restore_agent
sys.path.insert(0,str(REPO/'experiments/unified_mechanism'))
from metrics import write_rows


def records(path):
    with path.open() as f:return list(csv.DictReader(f))


def vector(rows,key):
    return np.array([float(r[key]) for r in rows])


def pair(direction,task,rows):
    a=vector(rows,'high_success').astype(bool);b=vector(rows,'random_success').astype(bool)
    return dict(controller=direction,task=task,n=len(a),high=int(a.sum()),random=int(b.sum()),
                gained=int((a&~b).sum()),lost=int((~a&b).sum()),
                both=int((a&b).sum()),neither=int((~a&~b).sum()))


def checkpoints(out):
    hp=read(G/'hard_goal_1000/protocol.json');h=read(G/'hard_goal_1000/training_complete.json')
    ep=read(G/'extraction_1000/training_complete.json')
    inputs={'base':(BASE/'params_100000.pkl',hp['base_sha256']),
            'high_critic':(G/'hard_goal_1000/model/params_101000.pkl',h['checkpoint_sha256'])}
    for stage,epoch in [('dynamics',102000),('actor',103000)]:
        inputs['high_'+stage]=(G/'extraction_1000/improved'/stage/f'params_{epoch}.pkl',
                               ep['checkpoints']['improved/'+stage]['sha256'])
    for stage in ['critic','dynamics','actor']:
        ck=read(out/(stage+'.json'))['checkpoint'];inputs['random_'+stage]=(out/stage/Path(ck['path']).name,ck['sha256'])
    result={k:dict(path=str(p.resolve()),sha256=check(p,d)) for k,(p,d) in inputs.items()}
    write(out/'checkpoints.json',result)
    return result


def plot(out,rows,official,greedy,paired):
    high=[r for r in rows if r['model']=='high'];rand=[r for r in rows if r['model']=='random']
    assert [r['case'] for r in high]==[r['case'] for r in rand]
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.25),layout='constrained')
    ax=axes[0];xx=100*vector(rand,'inversion');yy=100*vector(high,'inversion')
    ax.plot([0,100],[0,100],color='#888',lw=1,ls='--')
    ax.scatter(xx,yy,s=22,color='#246b7e',alpha=.75)
    ax.set(xlabel='Uniform selection (%)',ylabel='Highest-residual selection (%)',xlim=(-3,103),ylim=(-3,103))
    ax.set_aspect('equal');ax.set_title('A   Reversed action pairs per task3 start',loc='left',fontsize=10)
    ax=axes[1];bars=[r for r in paired if r['controller']=='official_actor' and r['task']!='all']
    x=np.arange(len(bars));w=.36
    for dx,arm,label,color in [(-w/2,'random','Uniform','#b98a50'),(w/2,'high','Highest residual','#246b7e')]:
        counts=[r[arm] for r in bars];bb=ax.bar(x+dx,counts,width=w,label=label,color=color)
        ax.bar_label(bb,fontsize=8,padding=2)
    ax.set(xticks=x,xticklabels=[f'Task {i}' for i in range(1,6)],ylim=(0,58),ylabel='Successful episodes / 50')
    ax.set_title('B   Matched official actor evaluation',loc='left',fontsize=10)
    ax.legend(loc='lower left',frameon=True,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'comparison.svg',metadata={'Date':None});plt.close(fig)

    tasks=[r for r in official if r['task']=='3']
    assert [r['case'] for r in tasks]==[r['case'] for r in greedy]
    data=np.array([[int(r[k+'_success']) for r in subset] for subset in [tasks,greedy] for k in ['random','high']])
    fig,ax=plt.subplots(figsize=(10.8,2),layout='constrained')
    ax.imshow(data,aspect='auto',vmin=0,vmax=1,interpolation='nearest',cmap=ListedColormap(['#e4ddd2','#246b7e']))
    ax.axhline(1.5,lw=2,color='white')
    ax.set(yticks=range(4),yticklabels=['Actor · uniform','Actor · max','Greedy · uniform','Greedy · max'],
           xticks=np.arange(0,50,5),xlabel='Official task3 episode ID (same start and goal)')
    ax.set_title('Task3 outcomes by case   •   teal = success, beige = timeout',loc='left',fontsize=10)
    fig.savefig(out/'task3_pairs.svg',metadata={'Date':None});plt.close(fig)


def run(out):
    freeze=read(out/'freeze.json');check(out/'PROTOCOL.md',freeze['protocol_sha256'])
    for f,v in freeze['train_code'].items():check(HERE/f,v)
    for p,v in read(out/'preflight.json')['paths_sha256'].items():check(p,v)
    manifest=checkpoints(out)
    base,_=load_base()
    models={a:restore_agent(base,str(Path(manifest[a+'_critic']['path']).parent),101000) for a in ['high','random']}
    old=read(G/'hard_goal_1000/training_complete.json');new=read(out/'critic.json')
    for k in ['paired','pool','uniform_queries']:assert old[k+'_stream_sha256']==new['streams'][k]
    archive=np.load(out/'selection.npz')
    expected=np.random.Generator(np.random.PCG64(20261102)).integers(0,8,(1000,128),dtype=np.int64)
    np.testing.assert_array_equal(archive['indices'],expected)
    assert int(archive['prototype_counts'].sum())==128000
    stages=read(G/'extraction_1000/training_complete.json')['stage_streams']
    for stage in ['actor','dynamics']:assert stages[stage]==read(out/(stage+'.json'))['streams']
    poolfile=G/'hard_goal_1000/validation_pool.npz';pp=read(G/'hard_goal_1000/protocol.json')
    check(poolfile,pp['validation_pool_sha256'])
    pool=np.load(poolfile)
    fit={m:pool_statistics(agent,pool['sources'],pool['goals'],pool['labels']) for m,agent in models.items()}
    old_metrics=records(G/'hard_goal_1000/metrics.csv')[-1]
    for k,v in fit['high'].items():np.testing.assert_allclose(v,float(old_metrics['pool_heldout/'+k]),rtol=1e-7,atol=1e-7)
    rows=records(out/'decisions.csv');paths=records(out/'paths.csv')
    compare=[];state_rows=[]
    def add(metric,h,r,unit,scope):
        compare.append(dict(scope=scope,metric=metric,unit=unit,high=float(h),random=float(r),high_minus_random=float(h-r)))
    def both(key,vals,unit,scope):add(key,*vals,unit,scope)
    ranked={m:[x for x in rows if x['model']==m] for m in models}
    assert [r['candidates_reference_sha256'] for r in ranked['high']]==[r['candidates_reference_sha256'] for r in ranked['random']]
    for k in ['reversed_pairs','strict_pairs','model_tied_strict','wrong']:
        both(k,[vector(rs,k).sum() for rs in ranked.values()],'count','50_official_starts')
    both('inversion_pooled',[vector(rs,'reversed_pairs').sum()/vector(rs,'strict_pairs').sum() for rs in ranked.values()],'fraction','50_official_starts')
    both('reference_margin_all',[vector(rs,'reference_margin').mean() for rs in ranked.values()],'steps','50_official_starts')
    both('reference_margin_when_wrong',[vector([r for r in rs if int(r['wrong'])],'reference_margin').mean() for rs in ranked.values()],'steps','wrong_only')
    for a,b in zip(ranked['high'],ranked['random']):
        state_rows.append(dict(case=a['case'],high_inversion=float(a['inversion']),random_inversion=float(b['inversion']),
                               high_excess=float(a['reference_margin']),random_excess=float(b['reference_margin']),
                               high_wrong=int(a['wrong']),random_wrong=int(b['wrong'])))
    write_rows(out/'ranking_pairs.csv',state_rows)
    local=records(out/'local.csv')
    for k in ['mean','p95','p99','maximum','positive_mean','positive_fraction','rms_positive']:
        both('h_'+k,[float(r[k]) for r in local],'fraction' if k=='positive_fraction' else 'steps','100000_val_edges')
    for scope in ['geometry_shortest','reference_env','official_grid']:
        for k in ['steps','sum_h','sum_tau','tau_over_cost','end','error_vs_path']:
            both(k,[vector([r for r in paths if r['model']==m and r['scope']==scope],k).mean() for m in models],
                 'fraction' if k=='tau_over_cost' else 'steps',scope)
    for k in fit['high']:both(k,[fit[m][k] for m in models],'steps' if 'rmse' in k else 'steps_squared','validation_teacher_pool')
    official=records(out/'official_pairs.csv');greedy=records(out/'greedy_pairs.csv')
    fixed=[]
    hh=G/'extraction_1000/improved_evaluation/fixed_bank';rr=out/'official_evaluation/fixed_bank'
    high,rand=(read(p/'summary.json')['cases'] for p in (hh,rr))
    for name in high:
        with np.load(hh/(name+'.npz')) as a,np.load(rr/(name+'.npz')) as b:
            np.testing.assert_array_equal(a['observations'][0],b['observations'][0]);np.testing.assert_array_equal(a['goal'],b['goal'])
        fixed.append(dict(case=name,task=int(name[4]),high_success=int(high[name]['success']),random_success=int(rand[name]['success'])))
    write_rows(out/'fixed_pairs.csv',fixed)
    paired=[]
    for label,rowset,count in [('official_actor',official,50),('fixed_actor',fixed,5)]:
        for i in range(1,6):
            subset=[r for r in rowset if int(r['task'])==i];assert len(subset)==count
            paired.append(pair(label,i,subset))
        paired.append(pair(label,'all',rowset))
    paired.append(pair('true_successor_greedy',3,greedy));assert len(greedy)==50
    write_rows(out/'navigation.csv',paired);write_rows(out/'comparison.csv',compare)
    plot(out,rows,official,greedy,paired)
    rng=read(out/'rng_audit.json')
    assert rng['matched_inputs']==250 and rng['actual_random_vs_reconstructed'] and len(rng['rows'])==250
    for m in models:
        actor=restore_agent(base,str(Path(manifest[m+'_actor']['path']).parent),103000)
        assert tree_equal(actor.network.params['modules_value'],models[m].network.params['modules_value'])
    write(out/'audit.json',dict(complete=True,preflight_and_pool_match=True,selection_replayed=True,frozen_critic_match=True,
          eval_inputs=250,greedy_inputs=50,reference_strict_pairs=int(vector(ranked['high'],'strict_pairs').sum()),
          max_path_closure=vector(paths,'prefix_identity_max').max(),historical_inputs_preserved=True,
          source_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file()},
          versions={n:importlib.metadata.version(n) for n in ['jax','jaxlib','numpy','mujoco','flax','optax','scipy']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
