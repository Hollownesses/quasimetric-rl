"""Run an already frozen protocol; refuse accidental overwrites."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime import ROOT

out = Path(sys.argv[1]).resolve()
frozen = out/'frozen_config.json'
assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (out/'frozen_config.sha256').read_text().strip()
manifest = json.loads(frozen.read_text())
assert manifest['argv'] == json.loads((out/'argv.json').read_text())
with (out/'LAUNCHED').open('x') as f:
    f.write(str(time.time()))
env = dict(os.environ, WANDB_MODE='offline', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
# Upstream creates a temporary W&B directory: keep that temp tree with this run.
(out/'tmp').mkdir(exist_ok=True)
env['TMPDIR'] = str(out/'tmp')
start = time.time()
command = [sys.executable, str(ROOT/'run_official.py'), *manifest['argv']]
(out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
with (out/'console.log').open('x') as log:
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out/'pid.json').write_text(json.dumps(dict(launcher=os.getpid(), training=process.pid)))
    code = process.wait()
(out/'runtime.json').write_text(json.dumps(dict(exit_code=code, start_unix=start,
    end_unix=time.time(), elapsed_seconds=time.time()-start), indent=2)+'\n')
if code == 0:
    (out/'TRAINING_COMPLETE').write_text('Official main returned successfully.\n')
sys.exit(code)
