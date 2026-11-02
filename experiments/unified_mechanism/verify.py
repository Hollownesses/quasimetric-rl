"""Audit final outputs against primary manifests and graph/action identities."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from metrics import read, sha, verified, write_rows
from summarize import group, load, values


def plot_attribution(rows,output):
    import matplotlib.pyplot as plt
    labels=['R0','R1','100k','E6','E16']
    parts=['delta_sum_h','delta_sum_tau','delta_snap','delta_terminal']
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained')
    groups=group(rows,'model')
    positions=np.arange(len(labels))
    # A common denominator within each model preserves additive identities.
    denominator=np.array([values(groups[m,],'gap').mean() for m in labels])
    for i,(key,label,color) in enumerate(zip(parts,['Edge error','Triangle slack (signed)','Mapping','Terminal'],
                                             ['#246378','#d79b41','#5b896a','#989ba6'])):
        vals=np.array([np.mean([float(r.get(key,0)) for r in groups[m,]]) for m in labels])/denominator
        ax.bar(positions+(i-1.5)*.16,-vals if key=='delta_sum_tau' else vals,width=.16,label=label,color=color)
    total=np.array([values(groups[m,],'delta_error').mean() for m in labels])/denominator
    ax.scatter(positions,total,marker='D',s=28,color='#222',label='Net successor error',zorder=3)
    ax.axhline(-1,ls='--',lw=1,color='#555',label='Reference gap cancelled')
    ax.axhline(0,color='#777',lw=.65)
    ax.set_xticks(positions,[f"{m} (n={len(groups[m,])})" for m in labels])
    ax.set_ylabel('Mean contribution / mean reference action gap')
    ax.set_title('Why a lower-ranked successor wins',loc='left')
    ax.legend(ncol=3,fontsize=8,frameon=False,loc='lower left',bbox_to_anchor=(0,1.09))
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(output);plt.close(fig)


def run(raw,out):
    scripts=Path(__file__).resolve().parent
    repo=scripts.parents[1]
    og=raw/'results/ogbench_critic_20261029'
    mp=raw/'results/r_series_mppi_20261026/main'
    manifest=read(mp/'final_integrity.json')
    assert manifest['passed'] and manifest['all_actions_replayed_exact']
    checked={}
    for filename,digest in {**read(out/'u/inputs.json'),**read(out/'og/inputs.json')}.items():
        checked[filename]=verified(raw/filename,digest)
    hist=manifest['artifacts']
    for p in [mp/'states.json',*(mp/'arms').glob('*/episodes.jsonl'),*(mp/'arms').glob('*/*.npz')]:
        name=str(p.relative_to(raw))
        checked[name]=verified(p,hist[name])
    proto=read(og/'official_counterfactual/protocol.json')
    for k,v in proto['trace_sha256'].items():
        arm,case=k.split('/')
        p=og/'extraction_1000'/(arm+'_evaluation')/'official'/(case+'.npz')
        checked[str(p.relative_to(raw))]=verified(p,v)
    # Verify only source modules consumed through the external OG runtime.
    for filename in ['experiments/ogbench_critic/common.py','experiments/ogbench_official_qrl/runtime.py',
                     'experiments/ogbench_diagnostics/core.py','experiments/ogbench_diagnostics/fast_pointmaze.py']:
        checked[filename]=verified(raw/filename,sha(repo/filename))
    frozen=read(og/'mean_extraction_1000/report_audit.json')
    assert frozen['passed'] and frozen['matched_official_inputs']==250 and frozen['matching_e17_training_streams']
    op=load(out/'og/official.csv')
    got=group(op,'model','task')
    for name,key in [('100k','A 原100k'),('E6','C E6 mean'),('E16','B E16')]:
        assert [int(values(got[name,str(t)],'success').sum()) for t in range(1,6)]==frozen['official_counts'][key]
    paths=load(out/'u/paths.csv')+load(out/'og/paths.csv')
    prefix_max=max(values(paths,'prefix_identity_max'))
    for p in paths:
        a,b,jt,h,t=[float(p[k]) for k in ['start','cost','end','sum_h','sum_tau']]
        assert abs(a-b-jt-h+t)<1e-7
    # An independent reconstruction of why a graph winner changed order:
    # e_i = sum(h)_i - sum(tau)_i on its reference continuation.
    ref=np.load(raw/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz')
    src,dst,refv=(ref[k] for k in ['source','target','distance'])
    replay=torch.load(raw/'results/connected_expert_utrap_20261021/main/verified_replay.pt',weights_only=True)
    cost=-replay['raw']['rewards'].numpy().astype(float)
    i=np.flatnonzero(ref['physical'])
    _,u=np.unique(np.column_stack((src[i],dst[i],cost[i])),axis=0,return_index=True)
    i=i[u]
    action=group(load(out/'u/decisions.csv'),'model','scope','case')
    attribution=[]
    for model in ['R0','R1']:
        vertex={int(p['case']):p for p in paths if p['model']==model and p['scope']=='graph_shortest'}
        def col(k):return np.array([float(vertex[s][k]) if s in vertex else 0 for s in range(len(refv))])
        j,hs,ts=map(col,['start','sum_h','sum_tau'])
        for s in np.unique(src[i]):
            e=i[src[i]==s]
            if len(e)<2:continue
            qa=cost[e]+j[dst[e]];qr=cost[e]+refv[dst[e]]
            a,o=np.argmin(qa),np.argmin(qr)
            da,do=dst[e[[a,o]]];gap=qr[a]-qr[o]
            dh,dt=hs[da]-hs[do],ts[da]-ts[do]
            expected,=action[model,'graph',str(s)]
            assert abs(gap-float(expected['reference_margin']))<1e-8
            assert abs(dh-dt-float(expected['chosen_error_difference']))<1e-8
            attribution.append(dict(model=model,source=s,chosen_target=da,reference_target=do,
                               gap=gap,wrong=int(gap>1e-4),delta_sum_h=dh,delta_sum_tau=dt,
                               delta_error=dh-dt,model_gap=qa[a]-qa[o]))
    assert len(attribution)==116
    write_rows(out/'graph_attribution.csv',attribution)
    # Main vs sensitivity pipeline must agree on the entire coarse-grid denominator.
    starts={(r['model'],r['case']):r for r in load(out/'og/decisions.csv') if r['scope']=='start'}
    sens=load(out/'resolution.csv')
    assert len(starts)==len(sens)==150 and len({(r['model'],r['case']) for r in sens})==150
    for r in sens:
        main=starts[r['model'],r['case']]
        for k,t in [('strict_pairs','coarse_pairs'),('reversed_pairs','coarse_reversed'),('wrong','coarse_wrong')]:
            assert main[k]==r[t]
    att=load(out/'og_grid_attribution.csv')
    wrong={k for k,r in starts.items() if int(r['wrong'])}
    assert {(r['model'],r['case']) for r in att}==wrong and len(att)==len(wrong)
    for r in att:
        main=starts[r['model'],r['case']]
        de=sum(float(r[k]) for k in ['delta_snap','delta_terminal','delta_sum_h'])-float(r['delta_sum_tau'])
        assert abs(de-float(main['chosen_error_difference']))<1e-7
        assert abs(float(r['gap'])+de-float(main['chosen_model_gap']))<1e-7
    plot_attribution([r for r in attribution if r['wrong']]+att,out/'action_gap.svg')
    audit=dict(raw_root=str(raw.resolve()),historical_code_commit=subprocess.check_output(
        ['git','-C',str(raw),'rev-parse','HEAD'],text=True).strip(),paths=len(paths),max_prefix_closure=prefix_max,
        decision_counts=dict(U=len(load(out/'u/decisions.csv')),OG=len(load(out/'og/decisions.csv'))),
        coarse_sensitivity_identity=True,verified_sha256=checked,
        analysis_source_sha256={p.name:sha(p) for p in sorted(scripts.iterdir()) if p.is_file()})
    (out/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print('Verified',len(paths),'paths, closure',prefix_max,len(checked),'input fingerprints')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.raw.resolve(),args.out)
