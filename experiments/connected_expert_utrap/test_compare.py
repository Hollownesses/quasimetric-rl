"""Evaluation checks independent of learner updates."""
import numpy as np
import torch
from types import SimpleNamespace
from .compare import statistics,decisions,equality,graph_structure


def test_nrmse_definition_and_recursive_state_check():
    metric=statistics(np.array([2.,6.]),np.array([1.,3.]))
    assert metric['nrmse_percent']==100
    assert equality({'a':[torch.tensor([1,2]),3]},{'a':[torch.tensor([1,2]),3]})
    assert not equality({'a':torch.tensor([1])},{'a':torch.tensor([2])})


def test_branch_choice_uses_cost_and_reference_only_for_scoring():
    # From0: direct expensive edge, or cheaper two-edge route through1.
    data=SimpleNamespace(cost=torch.tensor([9.,2.,3.]),dst=torch.tensor([2,1,2]),
                         nodes=torch.zeros(3,42),raw=SimpleNamespace(actions=torch.zeros(3,1)))
    graph=SimpleNamespace(goal=2,outgoing=[[0,1],[2],[]])
    truth=np.array([5.,3.,0.]);starts=np.array([0,1])
    good,_=decisions(data,graph,truth,truth,starts,300)
    assert good['summary']['optimal_branch_choices']==1
    assert good['summary']['mean_successful_regret']==0
    wrong,_=decisions(data,graph,truth,np.array([5.,20.,0.]),starts,300)
    assert wrong['branches'][0]['chosen_row']==0
    assert wrong['summary']['optimal_branch_choices']==0
    assert wrong['summary']['mean_successful_regret']==2
    data.original=torch.arange(3)
    structure=graph_structure(data,graph,np.array([0,0,1]))
    assert structure['acyclic'] and structure['longest_possible_recorded_rollout']==2
    assert structure['unique_starts']==2
