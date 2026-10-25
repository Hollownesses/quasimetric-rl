import json
from pathlib import Path
from types import SimpleNamespace
import torch
from experiments.task_source_balance.train import run,equal,write
root=Path('results/task_source_balance_20261025')
p=json.loads((root/'main/protocol.json').read_text())
p.update(phr_steps=2,decay_steps=2,audit_interval=2,threads=1)
(root/'smoke').mkdir(exist_ok=False)
write(root/'smoke/protocol.json',p)
checks={}
for arm,mode in [('R0','record'),('R1','source')]:
    out=root/'smoke'/arm
    args=SimpleNamespace(protocol=str(root/'smoke/protocol.json'),output=str(out),mode=mode,resume=None)
    run(args)
    expected=torch.load(out/'final.pt',weights_only=True)
    args.resume=str(out/'phr.pt');run(args)
    actual=torch.load(out/'final.pt',weights_only=True)
    assert equal(expected,actual)
    args.mode='source' if mode=='record' else 'record'
    try:run(args)
    except AssertionError:pass
    else:raise AssertionError('cross-mode resume incorrectly accepted')
    checks[arm]=dict(full_checkpoint_exact_after_resume=True,wrong_mode_rejected=True)
checks['formal_protocol_unchanged']=True
write(root/'integration_smoke.json',checks)
print(checks)
