"""I/O-only adapters around the pinned, unmodified official implementation."""
import functools
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'vendor' / 'ogbench'
COMMIT = (ROOT / 'OFFICIAL_COMMIT').read_text().strip()


def prepare():
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == COMMIT, (actual, COMMIT)
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'diff', 'HEAD'], text=True)
    # Resolve the pinned package directly, even when an editable-install .pth
    # is ignored by Python startup. This changes imports only, not official code.
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / 'impls'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('WANDB_MODE', 'offline')
    import ogbench
    assert Path(ogbench.__file__).resolve().is_relative_to(UPSTREAM.resolve())
    # Only redirect storage; compact data creation and goal sampling stay official.
    ogbench.make_env_and_datasets = functools.partial(
        ogbench.make_env_and_datasets, dataset_dir=str(ROOT / 'data'))
    import utils.log_utils as log_utils
    original_setup = log_utils.setup_wandb

    def setup_wandb(*args, **kwargs):
        # Upstream explicitly supplies mode='online', overriding WANDB_MODE.
        kwargs['mode'] = os.environ['WANDB_MODE']
        return original_setup(*args, **kwargs)

    log_utils.setup_wandb = setup_wandb
