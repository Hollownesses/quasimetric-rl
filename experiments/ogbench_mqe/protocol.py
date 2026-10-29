"""Freeze native MQE defaults and declared benchmark/budget overrides."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from runtime import prepare,ROOT,UPSTREAM,COMMIT,BENCHMARK_COMMIT


def arguments(mode,output):
    args=['--env_name=pointmaze-medium-navigate-v0','--seed=0',f'--agent={UPSTREAM}/impls/agents/mqe.py',
          '--agent.alpha=0.03','--agent.diag_backup=0.5',f'--save_dir={output}',
          f'--run_group=mqe_navigate_{mode}','--video_episodes=0','--eval_episodes=50']
    if mode=='smoke':
        args+=['--train_steps=2','--log_interval=1','--eval_interval=2','--save_interval=2','--eval_episodes=1']
    elif mode=='100k':
        args+=['--train_steps=100000','--log_interval=1000','--eval_interval=10000','--save_interval=10000']
    elif mode!='full':raise ValueError(mode)
    return args


if __name__=='__main__':
    mode,output=sys.argv[1:];out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'frozen_config.json').exists()
    prepare();args=arguments(mode,out);sys.argv=['main.py',*args]
    import main as official
    import jax
    from utils.log_utils import get_flag_dict
    official.FLAGS(sys.argv)
    frozen=dict(mqe_commit=COMMIT,benchmark_commit=BENCHMARK_COMMIT,mode=mode,argv=args,
        flags=get_flag_dict(),python=sys.version,platform=platform.platform(),
        devices=[str(d) for d in jax.devices()],endpoint=dict(smoke=2,**{'100k':100000,'full':1000000})[mode],
        selection='fixed endpoint; no best checkpoint selection',
        alpha_source='arXiv:2511.07730v1 Table 3 pointmaze=0.03',
        diagonal_weight_source='fixed source default 0.5; task value not specified in paper',
        source_acquisition='hash-verified runtime subset; not a full Git clone',
        source_lock_sha256=hashlib.sha256((ROOT/'SOURCE_BLOBS').read_bytes()).hexdigest(),
        adapters=['external official OGBench pinned to QRL benchmark; no fork action noise',
                  'same local original dataset_dir','offline W&B','video_episodes=0','CPU JAX',
                  'macOS: preinitialize MuJoCo with native CGL before main assigns Linux EGL'],
        local_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    text=json.dumps(frozen,indent=2,default=str)+'\n'
    (out/'frozen_config.json').write_text(text)
    (out/'frozen_config.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'\n')
    (out/'argv.json').write_text(json.dumps(args,indent=2)+'\n')
    (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(out/'frozen_config.json')
