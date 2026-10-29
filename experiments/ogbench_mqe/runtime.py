"""Unmodified native MQE with the same external official OGBench as QRL."""
import functools
import os
from pathlib import Path
import platform
import subprocess
import sys

from bootstrap import ROOT,UPSTREAM,COMMIT,verify

BENCHMARK=ROOT.parent/'ogbench_official_qrl/vendor/ogbench'
BENCHMARK_COMMIT='a977253ce5b88434f4e4f531c50047cc95993a51'
DATA=ROOT.parent/'ogbench_official_qrl/data'


def prepare():
    verify()
    assert subprocess.check_output(['git','-C',str(BENCHMARK),'rev-parse','HEAD'],text=True).strip()==BENCHMARK_COMMIT
    assert not subprocess.check_output(['git','-C',str(BENCHMARK),'diff','HEAD'],text=True)
    # MQE requirements recommend an external OGBench. Its bundled fork adds
    # action noise and eager rendering; do not put that package on sys.path.
    sys.path.insert(0,str(BENCHMARK))
    sys.path.insert(0,str(UPSTREAM/'impls'))
    os.environ.setdefault('MPLBACKEND','Agg')
    os.environ.setdefault('WANDB_MODE','offline')
    if platform.system()=='Darwin':
        # Native main later assigns Linux EGL unconditionally. Initialize the
        # MuJoCo binding with macOS CGL first; numerical dynamics are unchanged.
        os.environ['MUJOCO_GL']='cgl'
        import mujoco
        assert mujoco.__version__
    import ogbench
    assert Path(ogbench.__file__).resolve().is_relative_to(BENCHMARK.resolve())
    ogbench.make_env_and_datasets=functools.partial(ogbench.make_env_and_datasets,dataset_dir=str(DATA))
    import utils.log_utils as logs
    assert Path(logs.__file__).resolve().is_relative_to(UPSTREAM.resolve())
    original=logs.setup_wandb
    def setup_wandb(*args,**kwargs):
        kwargs['mode']=os.environ['WANDB_MODE']
        return original(*args,**kwargs)
    logs.setup_wandb=setup_wandb
