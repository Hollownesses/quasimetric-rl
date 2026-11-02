"""Correct query provenance without more model evaluations or changed candidates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO))
from experiments.native_margin.shared import rows,write,sha
from experiments.native_objective_witness.core import hoeffding_halfwidth

def run(out):
    history=json.loads((out/'query_history.json').read_text());assert history['historical_512_exact']
    audit=json.loads((out/'audit.json').read_text());qa=json.loads((out/'query_audit.json').read_text())
    assert qa['three_field_sha256']==history['prefix_sha256']['1024'] and qa['full_field_sha256']==history['full_fields_sha256']
    with (out/'queries.csv').open() as f:query=list(csv.DictReader(f))
    assert len(query)==4096
    for name in ['audit.json','query_audit.json','points.csv','comparisons.csv']:
        if not (out/('before_provenance_correction_'+name)).exists():shutil.copy2(out/name,out/('before_provenance_correction_'+name))
    points=audit['points']
    for key,r in points.items():
        rr=sorted([x for x in query if x['point']==key],key=lambda x:int(x['batch']))
        assert [int(x['batch']) for x in rr]==list(range(1024))
        p=np.array([float(x['P']) for x in rr]);r['P_all1024']=float(p.mean());r['P_old512']=float(p[:512].mean());r['P']=float(p[512:].mean())
        r['P_plus_D_raw']=r['P']+r['D_raw'];r['P_plus_D_extracted']=r['P']+r['D_extracted']
    half=hoeffding_halfwidth(524288);comp=[]
    for scale in ['raw','boundary']:
        a,b=points['original_'+scale],points['E16_'+scale]
        pp=[]
        for name in ['original','E16']:
            rr=sorted([x for x in query if x['point']==name+'_'+scale],key=lambda x:int(x['batch']))
            pp.append(np.array([float(x['P']) for x in rr])[512:])
        delta=pp[1]-pp[0];se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
        for dyn in ['raw','extracted']:
            dd=b['D_'+dyn]-a['D_'+dyn];d=float(delta.mean()+dd)
            comp.append(dict(scale=scale,dynamics=dyn,both_feasible=int(a['feasible'] and b['feasible']),delta_P=float(delta.mean()),delta_D=dd,
                delta_primal=d,batch_SE=se,normal_lower=d-1.96*se,normal_upper=d+1.96*se,hoeffding_lower=d-half,hoeffding_upper=d+half))
    audit['points']=points;audit['comparisons']=comp
    audit['query_reanalysis']=dict(old_batches_excluded=512,new_batches=512,new_queries=524288,
        reason='Prefix512 had already been used by native_ablation. No new model evaluation or candidate change.',
        query_history_sha256=sha(out/'query_history.json'),analysis_source_sha256=sha(__file__))
    qa.update(new_query_batches=512,new_queries=524288,old_query_batches=512,first512_history_exact=True,hoeffding_halfwidth=half)
    write(out/'audit.json',audit);write(out/'query_audit.json',qa)
    rows(out/'points.csv',[dict(point=k,**v) for k,v in points.items()]);rows(out/'comparisons.csv',comp)
    print('REAGGREGATED last512 batches; query/model/control budgets unchanged')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);run(p.parse_args().out)
