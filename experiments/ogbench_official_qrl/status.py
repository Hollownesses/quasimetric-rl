"""Read current progress without importing JAX or disturbing training."""
import json
from pathlib import Path
import re
import sys
import time

root = Path(sys.argv[1]).resolve()
run = root/'main'
log = run/'console.log'
with log.open('rb') as f:
    f.seek(max(0, log.stat().st_size - 131072))
    tail = f.read().decode(errors='replace')
matches = re.findall(r'(\d+)/100000\s+\[([^\]]+)\]', tail)
step = int(matches[-1][0]) if matches else 0
progress = matches[-1][1] if matches else 'initializing'
status = dict(checked_unix=time.time(), completed_updates_in_progress_bar=step,
              progress_display=progress, complete=(root/'COMPLETE').exists(),
              official_main_returned=(run/'TRAINING_COMPLETE').exists(),
              runtime=json.loads((run/'runtime.json').read_text()) if (run/'runtime.json').exists() else None,
              pids=json.loads((run/'pid.json').read_text()),
              checkpoint_paths=[str(p) for p in sorted(run.glob('OGBench/*/*/params_*.pkl'))],
              training_logs=[str(p) for p in run.glob('OGBench/*/*/train.csv')],
              evaluation_logs=[str(p) for p in run.glob('OGBench/*/*/eval.csv')])
print(json.dumps(status,indent=2))
