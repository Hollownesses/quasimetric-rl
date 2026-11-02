"""Replay sampling only: identify historical overlap, never query a network."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def run(raw,out):
    repo=Path(__file__).resolve().parents[2];sys.path[:0]=[str(raw/'experiments/ogbench_critic'),str(raw),str(repo)]
    from common import load_base,load_data
    from experiments.native_margin.shared import write
    import numpy as np
    _,cfg=load_base();env,train,_=load_data(cfg);env.close()
    np.random.seed(4400);h=hashlib.sha256();full=hashlib.sha256();prefix={}
    for i in range(1024):
        batch=train.sample(1024)
        for key in ['observations','next_observations','value_goals']:h.update(batch[key].tobytes())
        for key in sorted(batch):full.update(key.encode());full.update(np.asarray(batch[key]).tobytes())
        if i+1 in [32,512,1024]:prefix[str(i+1)]=h.hexdigest()
    old=json.loads((repo/'results/native_ablation_20261102/og/training.json').read_text())
    assert prefix['512']==old['stream_sha256']
    old32=json.loads((repo/'results/native_margin_20261102_verified/og/audit.json').read_text())['stream_sha256']
    assert prefix['32']==old32
    value=dict(prefix_sha256=prefix,full_fields_sha256=full.hexdigest(),historical_512_exact=True,
        historical_source=str(repo/'results/native_ablation_20261102/og/training.json'),
        total_batches=1024,old_batches=512,new_batches=512,new_queries=524288,network_queries_added=0)
    write(out/'query_history.json',value);print('VERIFIED historical overlap: first512, independent remainder: last512')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.raw.resolve(),a.out)
