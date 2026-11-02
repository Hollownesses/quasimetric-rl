"""Re-evaluate saved checkpoints and witnesses, assemble unrounded tables and figure."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from graph import HERE,HASH,NAMES,SRC,DST,G,Q,NEAR,TIE,cases,evaluate,feasible,graph,save,shortest,tag,write_rows
from iqe import NodeIQE,digest


def records(path):
    with path.open() as f:return {r['arm']:r for r in csv.DictReader(f)}


def qplot(out,tables,models):
    plt.rcParams.update({'svg.hashsalt':'edge-weight-graph','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,sharex='col',figsize=(10,6.4),gridspec_kw={'height_ratios':[2.3,1]})
    for col,delta in enumerate([1,2]):
        ax,detail=axes[:,col]
        table=[tables[tag(delta,q)] for q in Q]
        model=[models[tag(delta,q)] for q in Q]
        def num(rows,key):return np.array([float(r[key]) for r in rows])
        m=num(table,'m');lo=num(table,'m_min_outer');hi=num(table,'m_max_outer')
        neural=num(model,'m');ok=np.array([r['feasible']=='True' for r in model])
        hard=tables[tag(delta,None)];hard_iqe=models[tag(delta,None)]
        x=np.array(Q)
        ax.axhline(0,color='#222',linewidth=.8)
        ax.axhline(float(hard['m']),color='#678256',linewidth=1.2,linestyle='--',label='Hard table (all q)')
        ax.plot(x[0],float(hard_iqe['m']),marker='s',ms=6,color='#678256',fillstyle='none',label='Hard IQE (once)')
        ax.fill_between(x,lo,hi,color='#97c3cc')
        ax.plot(x,m,color='#17687b',lw=1.8,marker='.',ms=4,label='Mean-square table')
        ax.scatter(x[ok],neural[ok],s=34,facecolors='white',edgecolors='#17687b',zorder=4,label='IQE feasible')
        ax.scatter(x[~ok],neural[~ok],s=37,marker='x',color='#b84a2b',zorder=4,label='IQE infeasible')
        ax.set_title(f'Route-cost gap  Δ = {delta}',loc='left',fontweight='bold')
        ax.set_ylim(-9.8,2.75)
        if col==0:ax.set_ylabel('Action margin  m = J(b) − J(a)')
        ax.text(.03,.05,'m < 0 selects the costly route',transform=ax.transAxes,color='#575757')
        detail.axhline(0,color='#17687b',lw=.8)
        detail.fill_between(x,1e4*(lo-m),1e4*(hi-m),color='#b1d6db')
        detail.vlines(x,1e4*(lo-m),1e4*(hi-m),color='#17687b',linewidth=1.5)
        detail.set_title('Near-optimal margin interval, centered on table m',fontsize=9,loc='left')
        detail.set_ylim(-5.6,5.6)
        if col==0:detail.set_ylabel('Margin offset  × 10⁴')
        for panel in [ax,detail]:
            panel.set_xscale('log')
            panel.set_xlim(.25,6e-7)
            panel.grid(axis='y',alpha=.14)
        detail.set_xticks(x,['1/7','10⁻²','10⁻³','3·10⁻⁴','10⁻⁴','3·10⁻⁵','10⁻⁵','10⁻⁶'],
                          rotation=32,ha='right',fontsize=8)
        detail.minorticks_off()
        detail.set_xlabel('Weight q on edge u → g   (decreases to the right)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=5,frameon=False)
    fig.subplots_adjust(left=.085,right=.98,top=.87,bottom=.14,wspace=.24,hspace=.30)
    fig.savefig(out/'q_margin.svg',metadata={'Date':None})
    plt.close(fig)


def run(out):
    torch.set_num_threads(1)
    saved=out/'PROTOCOL.md'
    assert hashlib.sha256(saved.read_bytes()).hexdigest()==HASH
    tables,models,witnesses=[records(out/name) for name in ['table.csv','iqe.csv','witness.csv']]
    required={tag(d,q) for d,q in cases()}
    assert set(tables)==set(models)==required
    model_hash=digest(NodeIQE())
    triggers={key for key in required if tables[key]['interval_class']=='positive' and models[key]['action']=='b'}
    assert set(witnesses)==triggers
    metrics=[];edges=[];max_end_width=0.
    for dlt,q in cases():
        key=tag(dlt,q)
        t=tables[key]
        lower,upper=float(t['m_min_outer']),float(t['m_max_outer'])
        opt=float(t['F_upper']);c,w=graph(dlt,q)
        entries=[('table',t),('iqe',models[key])]
        if key in triggers:entries.append(('witness',witnesses[key]))
        for method,record in entries:
            pack=np.load(out/method/(key+'.npz'));matrix=pack['D']
            row=evaluate(matrix,dlt,q)
            for k,val in row.items():
                if isinstance(val,(float,np.floating)):
                    assert abs(val-float(record[k]))<=1e-12,(key,k)
            if method=='table':
                assert feasible(row,table=True)
                for suffix,bound,sign in [('min',lower,-1),('max',upper,1)]:
                    r=evaluate(pack[suffix+'_matrix'],dlt,q)
                    assert feasible(r,table=True) and r['F']>=opt-NEAR-2e-12
                    width=sign*(bound-r['m'])
                    assert -1e-10<=width<=2e-5
                    max_end_width=max(max_end_width,width)
                if q is None:np.testing.assert_allclose(matrix[:,G],shortest(c)[:,G],rtol=0,atol=1e-8)
            else:
                net=NodeIQE()
                params={k:torch.from_numpy(pack[k]) for k in net.state_dict()}
                net.load_state_dict(params)
                np.testing.assert_array_equal(net.full().detach().numpy(),matrix)
                logs=json.loads((out/method/(key+'.json')).read_text())
                assert logs['initial_sha256']==model_hash and logs['final_sha256']==digest(net)
                assert [x['step'] for x in logs['logs']]==list(range(100,12001,100))
            ok=feasible(row,table=method=='table')
            assert ok==(record['feasible']=='True')
            metrics.append(dict(arm=key,method=method,**row,feasible=ok,F_upper=opt,
                                objective_gap=opt-row['F'],near_optimal=bool(ok and row['F']>=opt-NEAR),
                                m_min_outer=lower,m_max_outer=upper,interval_class=t['interval_class']))
            for e,(s,v) in enumerate(zip(SRC,DST)):
                edge=matrix[s,v];h=edge-c[e]
                edges.append(dict(arm=key,method=method,src=NAMES[s],dst=NAMES[v],c=c[e],w=w[e],
                                  d=edge,h=h,weighted_sq=w[e]*max(h,0)**2,
                                  tau=edge+matrix[v,G]-matrix[s,G]))
    write_rows(out/'metrics.csv',metrics);write_rows(out/'edges.csv',edges)
    audit=dict(protocol_sha256=HASH,initial_sha256=model_hash,training_arms=len(models),
               conditional_fits=sorted(triggers),feasible_iqe=sum(r['feasible'] for r in metrics if r['method']=='iqe'),
               near_optimal_iqe=sum(r['near_optimal'] for r in metrics if r['method']=='iqe'),
               max_interval_endpoint_bracket=max_end_width,
               max_table_F_gap=max(float(r['primal_gap']) for r in tables.values()),
               max_matrix_LP_upper_minus_F=max(float(r['F_matrix_upper'])-float(r['F']) for r in tables.values()),
               max_decomposition_error=max(r['identity_error'] for r in metrics),
               versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                             torch=torch.__version__,torchqmet=importlib.metadata.version('torchqmet'),
                             matplotlib=matplotlib.__version__,architecture=platform.machine()),
               source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.is_file()})
    save(out/'audit.json',audit)
    qplot(out,tables,models)
    print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path)
    run(parser.parse_args().out)
