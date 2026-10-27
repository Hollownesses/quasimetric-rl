"""Freeze complete official defaults plus explicitly declared run overrides."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from runtime import COMMIT, ROOT, UPSTREAM, prepare

prepare()
import jax
from utils.log_utils import get_flag_dict


def arguments(task, mode, output):
    args = [f'--env_name=pointmaze-medium-{task}-v0', '--seed=0',
            f'--agent={UPSTREAM}/impls/agents/qrl.py', '--agent.alpha=0.0003',
            '--eval_episodes=50', '--video_episodes=0', f'--save_dir={output}',
            f'--run_group={task}_{mode}']
    if task == 'stitch':
        args += ['--agent.actor_p_randomgoal=0.5', '--agent.actor_p_trajgoal=0.5']
    if mode == '100k':
        args += ['--train_steps=100000', '--log_interval=1000',
                 '--eval_interval=10000', '--save_interval=10000']
    elif mode == 'smoke':
        args += ['--train_steps=1', '--log_interval=1', '--eval_interval=1',
                 '--save_interval=1', '--eval_episodes=1']
    elif mode != 'full':
        raise ValueError(mode)
    return args


if __name__ == '__main__':
    task, mode, output = sys.argv[1:]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'frozen_config.json'
    if frozen.exists():
        raise FileExistsError(f'Will not overwrite frozen configuration: {frozen}')
    args = arguments(task, mode, out)
    # ml_collections discovers config overrides from sys.argv at flag definition.
    sys.argv = ['main.py', *args]
    import main as official
    official.FLAGS(sys.argv)
    manifest = dict(official_commit=COMMIT, task=task, mode=mode, argv=args,
                    flags=get_flag_dict(), python=sys.version, platform=platform.platform(),
                    jax_backend=jax.default_backend(), jax_devices=[str(x) for x in jax.devices()],
                    formal_endpoint=100000 if mode=='100k' else None,
                    selection='fixed endpoint; no best checkpoint selection',
                    adapters=['local dataset_dir', 'explicit offline W&B mode', 'video_episodes=0'],
                    source_sha256={str(p.relative_to(UPSTREAM)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in UPSTREAM.rglob('*.py') if '.git' not in p.parts})
    frozen.write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    (out/'frozen_config.sha256').write_text(hashlib.sha256(frozen.read_bytes()).hexdigest()+'\n')
    (out/'argv.json').write_text(json.dumps(args, indent=2)+'\n')
    (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(frozen)
