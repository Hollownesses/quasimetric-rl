"""Verify actual continuation checkpoints and frozen actor outputs."""
import argparse
import json
from pathlib import Path

from common import load_base, tree_equal, tree_digest
from train import freeze_proof
from evaluate import read_bank
import jax
import numpy as np
from utils.flax_utils import restore_agent


def run(root,steps,bank):
    root=Path(root)
    base,_=load_base()
    completion=json.loads((root/'training_complete.json').read_text())
    assert completion['steps']==steps
    cases=read_bank(bank)
    obs=np.array([c['observation'] for c in cases.values()])
    goals=np.array([c['goal'] for c in cases.values()])
    key=jax.random.PRNGKey(123)
    reference=base.sample_actions(obs,goals,seed=key,temperature=0)
    results={}
    for arm in ['pair','goal']:
        agent=restore_agent(base,str(root/arm),100000+steps)
        assert int(agent.network.opt_state[0].count)==100000+steps
        assert int(agent.network.step)==100001+steps
        assert freeze_proof(agent)==freeze_proof(base)
        assert not tree_equal(agent.network.params['modules_value'],base.network.params['modules_value'])
        assert tree_equal(agent.sample_actions(obs,goals,seed=key,temperature=0),reference)
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent))
        assert tree_digest(agent.network.params['modules_value'])==completion['final'][arm]['critic_digest']
        results[arm]=dict(passed=True,adam_count=int(agent.network.opt_state[0].count),
            frozen_parameters_and_moments=True,bank_actor_actions_equal=True,all_leaves_finite=True)
    (root/'checkpoint_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--steps',type=int,required=True);p.add_argument('--bank',required=True)
    a=p.parse_args();run(a.root,a.steps,a.bank)
