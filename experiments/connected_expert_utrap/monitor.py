"""Read-only milestone supervision; never changes the training configuration."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return True


def inspect(root,arm):
    lines=(root/'training.jsonl').read_text().splitlines()
    records=[]
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except json.JSONDecodeError:
            if i==len(lines)-1:break  # Writer may be midway through a line.
            raise
        assert finite(r),(arm,'nonfinite',r['step'])
        if arm=='qrl_task' and 'connected_expert_components' in r:
            assert r['connected_expert_components']['optimized_pair_rows']==0
        if 'connected_expert' in r or r.get('kind')=='scan':
            assert r['step']%5000==0
            if arm=='qrl_task':
                assert r['anchor_max']==0
                visits=r['connected_expert_visits']
                assert visits['max']-visits['min']<=1
            else:assert r['terminal_distance']==0
            records.append(r)
    return records


def run(args):
    roots={'qrl_task':Path(args.qrl_task),'td':Path(args.td)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    if out.exists():
        seen={(r['arm'],r['step']) for r in map(json.loads,out.read_text().splitlines())}
    while True:
        with out.open('a') as handle:
            for arm,root in roots.items():
                if not (root/'training.jsonl').exists():continue
                for rec in inspect(root,arm):
                    key=(arm,rec['step'])
                    if key in seen:continue
                    row=dict(arm=arm,step=rec['step'],passed=True,
                             observed_utc=datetime.now(timezone.utc).isoformat(),scan=rec)
                    handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                    print(json.dumps(row,allow_nan=False),flush=True);seen.add(key)
        if all((r/'COMPLETE').exists() for r in roots.values()):
            assert seen=={(arm,step) for arm in roots for step in range(0,50001,5000)}
            break
        if not args.watch:break
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('qrl-task','td','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--watch',action='store_true');run(p.parse_args())
