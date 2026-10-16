"""Short, pre-main calibration using push and full constraints, never graph labels."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run
from types import SimpleNamespace

root=Path(sys.argv[1])
for parameterization,method,lr,rho,dual_step in [
    ('latent','barrier',.01,10.,.1),('latent','barrier',.05,10.,.1),
    ('latent','phr',.05,10.,.1),('latent','phr',.05,100.,1.),
    ('encoder','barrier',2e-5,10.,.1),
    ('encoder','phr',2e-5,10.,.1),('encoder','phr',2e-5,100.,1.)]:
    name=f'{parameterization}_{method}_lr{lr}_rho{rho}'
    run.run(SimpleNamespace(output=str(root/name),parameterization=parameterization,method=method,
        lr=lr,rho=rho,dual_step=dual_step,steps=300,critic=0,seed=42,threads=2,eval_interval=100))
