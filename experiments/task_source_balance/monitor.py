"""Read-only training supervision. Never reads graph labels or changes budgets."""
import argparse
import json
import math
from pathlib import Path
from datetime import datetime,timezone


def finite(value):
    if isinstance(value,float):assert math.isfinite(value)
    elif isinstance(value,dict):
        for item in value.values():finite(item)
    elif isinstance(value,list):
        for item in value:finite(item)


def snapshot(root):
    root=Path(root);logs={};latest={}
    for arm in ('R0','R1'):
        path=root/arm/'training.jsonl';records=[]
        if not path.exists():return dict(starting=True)
        for line in path.read_text().splitlines():
            try:record=json.loads(line)
            except json.JSONDecodeError:continue  # concurrent partial last write
            finite(record)
            if 'connected_expert' in record:
                assert record['anchor_max']==0
                visits=record['connected_expert_visits'];assert visits['max']-visits['min']<=1
                counts=record['push_counters']
                assert counts['connected_expert_task_visits']==counts['connected_expert_pair_visits']
                assert counts['connected_expert_empty_task_batches']==0
                records.append(record)
        if not records:return dict(starting=True)
        logs[arm]={r['step']:r for r in records}
        r=records[-1];latest[arm]=dict(step=r['step'],seconds=r['seconds'],
            max_violation=r['connected_expert']['max_violation'],
            mean_positive=r['connected_expert']['mean_positive'],
            dual_max=r['connected_expert']['dual_max'],complete=(root/arm/'COMPLETE').exists())
    shared=sorted(logs['R0'].keys() & logs['R1'].keys())
    for step in shared:
        for key in ('sampling_chain_sha256','connected_expert_visits','push_counters','epochs'):
            assert logs['R0'][step][key]==logs['R1'][step][key],(step,key)
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),latest=latest,
        matched_audits=len(shared),latest_matched_step=shared[-1],finite=True,paired=True,reference_read=False)
    with (root/'supervision.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(json.dumps(snapshot(p.parse_args().root),indent=2))
