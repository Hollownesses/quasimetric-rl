from types import SimpleNamespace
import numpy as np
import pytest
import torch
from .reference import task_goals, validate_reference


def graph():
    nodes=torch.zeros(3,42);nodes[2,41]=1
    return SimpleNamespace(nodes=nodes,original=torch.tensor([0,1]),
        goal=torch.tensor([2,2,0]),src=torch.tensor([0,1,2]),dst=torch.tensor([1,2,2]),
        cost=torch.tensor([2.,3.,0.]))


def test_anchor_placeholder_is_not_a_goal():
    d=graph();assert task_goals(d).tolist()==[2]
    validate_reference(d,np.array([5.,3.,0.]))
    with pytest.raises(ValueError,match='Zero-distance'):
        validate_reference(d,np.array([0.,3.,0.]))


def test_inequality_alone_does_not_certify_shortest_distances():
    with pytest.raises(ValueError,match='Bellman'):
        validate_reference(graph(),np.array([1.,2.,0.]))


def test_noncanonical_goal_rejected():
    d=graph();d.goal[0]=0
    with pytest.raises(ValueError,match='canonical'):
        task_goals(d)
