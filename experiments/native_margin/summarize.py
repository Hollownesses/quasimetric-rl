"""Aggregate at probe/state level, preserving the batch-level local pressure."""
import argparse
import csv
from pathlib import Path
import numpy as np
from shared import rows, write

def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def run(out):
    records=[];summaries=[]
    for scene,folder,parts in [('U','u',['task','b1','local','local_equal','total','adam','adam_linear']),
                               ('OG','og',['push','local','dynamics','local_equal','total','adam','adam_linear'])]:
        probes=read(out/folder/'probes.csv');pressure=read(out/folder/'pressure.csv')
        for p in probes:
            rr=[r for r in pressure if r['case']==p['case']];assert len(rr)==32
            rec=dict(scene=scene,**p,wrong=int(float(p['margin']) < -1e-4),tied=int(abs(float(p['margin']))<=1e-4))
            for part in parts:
                vv=np.array([float(r[part]) for r in rr])
                rec[part+'_mean']=float(vv.mean());rec[part+'_positive_fraction']=float((vv>0).mean())
            rec['push_mean']=rec.get('push_mean',rec.get('task_mean',0)+rec.get('b1_mean',0))
            rec['equal_total_mean']=rec['total_mean']-rec['local_mean']+rec['local_equal_mean']
            rec['margin_relative']=float(p['margin'])/float(p['ref_gap'])
            rec['sample_min']=min(float(p['good_sample_visits']),float(p['bad_sample_visits']))
            records.append(rec)
        ss=[r for r in records if r['scene']==scene]
        for group in ['all','wrong','correct','tied']:
            selected=[r for r in ss if group=='all' or (group=='wrong' and r['wrong']) or (group=='correct' and not r['wrong'] and not r['tied']) or (group=='tied' and r['tied'])]
            if not selected:continue
            rec=dict(scene=scene,group=group,n=len(selected))
            for part in ['push','local','local_equal','total','equal_total','adam']:
                vv=np.array([r[part+'_mean'] for r in selected])
                rec[part+'_positive']=int((vv>0).sum());rec[part+'_mean']=float(vv.mean())
                rec[part+'_normalized_median']=float(np.median(vv/np.array([float(r['ref_gap']) for r in selected])))
            rec['local_correct_push_opposes']=sum(r['local_mean']>0 and r['push_mean']<0 for r in selected)
            rec['local_correct_push_overrides']=sum(r['local_mean']>0 and r['push_mean']<0 and r['total_mean']<0 for r in selected)
            rec['adam_positive_after_zero_subtraction']=sum(r['adam_mean']-float(r['zero_adam'])>0 for r in selected)
            rec['both_local_push_harm']=sum(r['local_mean']<0 and r['push_mean']<0 for r in selected)
            rec['no_sample_near_either']=sum(r['sample_min']==0 for r in selected)
            summaries.append(rec)
    # Descriptive common-support cells; not an estimate of exposure causality.
    strata=[]
    for scene in ['U','OG']:
        bins={}
        for r in records:
            if r['scene']!=scene:continue
            support=min(float(r['good_source_records']),float(r['bad_source_records']))
            key=(int(np.floor(np.log2(float(r['ref_gap'])))),int(np.floor(np.log2(support+1))))
            bins.setdefault(key,[]).append(r)
        for (gap,support),rr in sorted(bins.items()):
            for wrong in [0,1]:
                selected=[r for r in rr if r['wrong']==wrong]
                if not selected:continue
                strata.append(dict(scene=scene,gap_log2=gap,support_plus1_log2=support,wrong=wrong,n=len(selected),
                                   both_classes=int(any(r['wrong']!=wrong for r in rr)),
                                   push_mean=np.mean([r['push_mean'] for r in selected]),
                                   local_mean=np.mean([r['local_mean'] for r in selected]),
                                   total_mean=np.mean([r['total_mean'] for r in selected])))
    rows(out/'strata.csv',strata)
    rows(out/'per_probe.csv',records);rows(out/'summary.csv',summaries)
    write(out/'summary.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for col,scene in enumerate(['U','OG']):
        ss=[r for r in records if r['scene']==scene]
        ax=axes[0,col]
        for wrong,color,label in [(0,'#377e91','initially correct / tied'),(1,'#d17b35','initially reversed')]:
            rr=[r for r in ss if r['wrong']==wrong]
            ax.scatter([r['local_mean']/float(r['ref_gap']) for r in rr],
                       [r['push_mean']/float(r['ref_gap']) for r in rr],c=color,label=label,s=24,alpha=.8)
        ax.axhline(0,c='#777',lw=.7);ax.axvline(0,c='#777',lw=.7)
        ax.set_xscale('symlog',linthresh=1e-7);ax.set_yscale('symlog',linthresh=1e-7)
        ax.set_xlabel('local contribution / reference gap');ax.set_ylabel('push contribution / reference gap')
        ax.set_title(scene+'   Native SGD pressure (positive repairs order)',loc='left');ax.legend(fontsize=7,frameon=False)
        ax=axes[1,col];ss=sorted(ss,key=lambda r:float(r['margin_relative']))
        xx=np.arange(len(ss));ax.scatter(xx,[r['total_mean']/float(r['ref_gap']) for r in ss],s=14,label='native SGD')
        ax.scatter(xx,[r['adam_mean']/float(r['ref_gap']) for r in ss],s=14,marker='x',label='native Adam')
        ax.axhline(0,c='#777',lw=.7);ax.set_yscale('symlog',linthresh=1e-7)
        ax.set_xlabel('probe ordered by initial margin / reference gap');ax.set_ylabel('mean change / reference gap')
        ax.legend(fontsize=8,frameon=False);ax.set_title(scene+'   Same checkpoint, 32 independent one-step updates',loc='left')
    fig.savefig(out/'pressure.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)
