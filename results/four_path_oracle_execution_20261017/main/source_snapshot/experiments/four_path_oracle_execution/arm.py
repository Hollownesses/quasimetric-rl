import argparse
from pathlib import Path
from experiments.four_path_oracle_execution.run import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();evaluate(a.root,a.arm)
