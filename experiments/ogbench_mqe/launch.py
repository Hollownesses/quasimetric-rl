"""Run an immutable native MQE protocol once; preserve every exit and log."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from bootstrap import ROOT,verify

out=Path(sys.argv[1]).resolve();frozen=out/'frozen_config.json'
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==(out/'frozen_config.sha256').read_text().strip()
manifest=json.loads(frozen.read_text());assert manifest['argv']==json.loads((out/'argv.json').read_text())
assert hashlib.sha256((ROOT/'SOURCE_BLOBS').read_bytes()).hexdigest()==manifest['source_lock_sha256']
for name,sha in manifest['local_source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
verify()
with (out/'LAUNCHED').open('x') as f:f.write(str(time.time())+'\n')
(out/'tmp').mkdir(exist_ok=True)
env=dict(os.environ,WANDB_MODE='offline',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',
         PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tmp'))
command=[sys.executable,str(ROOT/'run_official.py'),*manifest['argv']]
(out/'command.json').write_text(json.dumps(command,indent=2)+'\n');start=time.time()
with (out/'console.log').open('x') as log:
    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'pid.json').write_text(json.dumps(dict(launcher=os.getpid(),training=process.pid))+'\n')
    code=process.wait()
(out/'runtime.json').write_text(json.dumps(dict(exit_code=code,start_unix=start,end_unix=time.time(),
                                               elapsed_seconds=time.time()-start),indent=2)+'\n')
if code==0:(out/'TRAINING_COMPLETE').write_text('Native MQE main returned successfully.\n')
sys.exit(code)
