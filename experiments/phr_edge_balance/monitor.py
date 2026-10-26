"""Read-only reference-free full-audit pairing supervision."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.task_partial_balance.monitor import audits


def snapshot(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());new=audits(root/'R2/training.jsonl')
    if not new:return dict(starting=True)
    for arm,folder in p['baseline_roots'].items():
        old=audits(Path(folder)/'training.jsonl')
        for step,r in new.items():
            assert r['anchor_max']==0
            for key in ('sampling_chain_sha256','connected_expert_visits','push_counters','epochs'):assert r[key]==old[step][key],(arm,step,key)
    step=max(new);r=new[step]
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),step=step,seconds=r['seconds'],
        max_violation=r['connected_expert']['max_violation'],mean_positive=r['connected_expert']['mean_positive'],
        paired_audits=len(new),paired_to_all_baselines=True,complete=(root/'R2/COMPLETE').exists(),reference_read=False)
    with (root/'supervision.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);print(json.dumps(snapshot(p.parse_args().root),indent=2))
