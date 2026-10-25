"""Read-only audit matching; no graph/reference access or stopping changes."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path


def finite(v):
    if isinstance(v,float):assert math.isfinite(v)
    elif isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)


def audits(path):
    records={}
    if not path.exists():return records
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        finite(r)
        if 'connected_expert' in r:records[r['step']]=r
    return records


def snapshot(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());half=audits(root/'Rhalf/training.jsonl')
    if not half:return dict(starting=True)
    for arm in ('R0','R1'):
        old=audits(Path(p['prior_root'])/arm/'training.jsonl')
        for step,r in half.items():
            assert r['anchor_max']==0
            for key in ('sampling_chain_sha256','connected_expert_visits','push_counters','epochs'):assert r[key]==old[step][key],(arm,step,key)
    step=max(half);r=half[step]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),step=step,seconds=r['seconds'],
        max_violation=r['connected_expert']['max_violation'],mean_positive=r['connected_expert']['mean_positive'],
        paired_audits=len(half),paired_to_both_old_arms=True,complete=(root/'Rhalf/COMPLETE').exists(),reference_read=False)
    with (root/'supervision.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);print(json.dumps(snapshot(p.parse_args().root),indent=2))
