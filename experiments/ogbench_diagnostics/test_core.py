"""Fidelity checks against frozen E0/E1 tools, including contact-state episodes."""
import ast
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'experiments/ogbench_critic'))
from common import QRLAgent,load_data,distances
import evaluate as old_evaluate
from experiments.ogbench_diagnostics import core
import jax
import numpy as np


class CoreTests(unittest.TestCase):
    def test_original_geometry_and_restore_functions_are_identical(self):
        actual={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(core.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
        for filename,names in [('field.py',['geometry','point_clear','segment_clear','build_graph','graph_metrics']),
            ('evaluate.py',['wrappers','capture','restore','save_integration','load_case','read_bank','candidates'])]:
            source={n.name:n for n in ast.parse((REPO/'experiments/ogbench_critic'/filename).read_text()).body if isinstance(n,ast.FunctionDef)}
            for name in names:self.assertEqual(ast.dump(source[name],include_attributes=False),actual[name],name)

    def test_contact_candidates_and_full_episode_match_original_tools(self):
        from agents.qrl import get_config
        cfg=get_config().to_dict();cfg.update(actor_hidden_dims=(16,16),value_hidden_dims=(16,16),latent_dim=16)
        agent=QRLAgent.create(88,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
        env,_,_=load_data(cfg)
        try:
            env.reset(seed=3100,options=dict(task_id=3));env.unwrapped.set_xy(core.ANCHOR)
            case=dict(seed=3100,task=3,observation=env.unwrapped.get_ob().copy(),goal=core.GOAL.copy(),integration=core.save_integration(env))
            case['goal']=core.GOAL.copy()
            old_evaluate.load_case(env,case);before=old_evaluate.capture(env)
            original,_=old_evaluate.candidates(env)
            old_evaluate.restore(env,before);copied,_=core.candidates(env)
            self.assertTrue(np.array_equal(original,copied))
            for mode in ['real-greedy','actor']:
                info,trajectory=old_evaluate.rollout(agent,env,case,mode)
                if mode=='real-greedy':
                    actual,new=core.rollout(env,case,score=lambda obs,g:distances(agent,obs,g))
                else:
                    actual,new=core.rollout(env,case,actor=lambda ob,g,step:agent.sample_actions(ob,g,seed=jax.random.PRNGKey(step),temperature=0))
                self.assertEqual(info['success'],actual['success']);self.assertEqual(info['steps'],actual['steps'])
                for key in trajectory:self.assertTrue(np.array_equal(trajectory[key],new[key]),(mode,key))
        finally:env.close()


if __name__=='__main__':unittest.main()
