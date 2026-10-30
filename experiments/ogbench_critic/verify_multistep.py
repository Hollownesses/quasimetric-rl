"""Verify authoritative E2 state, initial unit conversion and frozen outputs."""
import argparse
import json
from pathlib import Path

from common import load_base,load_data,tree_digest,tree_equal
from train import freeze_proof
from evaluate import read_bank
import multistep as m
import jax
import numpy as np
from scipy.stats import spearmanr


def run(root,steps,bank):
    root=Path(root).resolve();base,config=load_base()
    done=json.loads((root/'training_complete.json').read_text());manifest=json.loads((root/'protocol.json').read_text())
    assert done['steps']==steps and manifest['steps']==steps
    env,_,val=load_data(config);env.close()
    np.random.seed(9001);batch=val.sample(2048)
    assert tree_digest(batch)==manifest['heldout_sha256']
    cases=read_bank(bank);obs=np.array([c['observation'] for c in cases.values()]);goals=np.array([c['goal'] for c in cases.values()])
    ref=base.sample_actions(obs,goals,seed=jax.random.PRNGKey(123),temperature=0)
    old=np.asarray(base.network.select('value')(batch['observations'],batch['value_goals']))
    output=root/'checkpoint_verification';output.mkdir(exist_ok=False)
    result={}
    for arm in ['one','multi']:
        initial,_=m.load_state(base,root/arm,0)
        assert tree_digest(initial)==manifest['initial_state_digest']
        initial_d=np.asarray(m.embeddings(base,initial.params,batch['observations'],batch['value_goals'])[0])
        error=float(np.max(np.abs(initial_d-m.UNIT*old)))
        rho=float(spearmanr(old,initial_d).statistic)
        assert error<5e-5 and rho>.999999
        state,bundle=m.load_state(base,root/arm,steps)
        assert bundle['mode']==arm and tree_digest(state)==done['final'][arm]['state_digest']
        assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(state))
        assert not tree_equal(initial.params,state.params)
        view=m.as_agent(base,state)
        assert freeze_proof(view)==manifest['frozen']==freeze_proof(base)
        assert tree_equal(view.sample_actions(obs,goals,seed=jax.random.PRNGKey(123),temperature=0),ref)
        rng=np.random.RandomState();rng.set_state(bundle['rng'])
        m.save(base,state,output/arm,arm,rng)
        again,re=m.load_state(base,output/arm,steps)
        assert tree_equal(again,state) and tree_equal(re['rng'],bundle['rng'])
        result[arm]=dict(passed=True,extra_updates=int(state.step),adam_count=int(state.opt_state[0].count),
            initial_max_unit_scaling_error=error,initial_rank_spearman=rho,
            full_state_roundtrip=True,frozen_parameters_and_moments=True,bank_actor_actions_equal=True)
    (root/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--steps',type=int,required=True);p.add_argument('--bank',required=True)
    a=p.parse_args();run(a.root,a.steps,a.bank)
