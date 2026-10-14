"""Search regressions independent of UAV discretization and wall-clock speed."""
import numpy as np
import pytest

from minimal_qrl.baselines import hybrid_astar as astar


class GraphEnv:
    omega_max = 1.0
    observation_violation_cost_weight = 0.0
    communication_violation_cost_weight = 0.0

    def __init__(self, goals=(2,), horizon=20):
        self.state = np.array([0., 0., 0.], np.float32)
        self.t = 0
        self.goals = goals
        self.max_episode_steps = horizon

    def get_state(self):
        return {'node': int(self.state[0]), 't': self.t}

    def set_state(self, snapshot):
        self.state = np.array([snapshot['node'], 0., 0.], np.float32)
        self.t = snapshot['t']

    def is_task_feasible(self, state):
        return int(state[0]) in self.goals


class GraphSearch(astar.HybridAStarController):
    def __init__(self, edges, *, bins=None, **config):
        super().__init__(astar.HybridAStarConfig(
            primitive_scales=(0., 1.), terminal_samples=0, **config))
        self.edges = edges
        self.bins = bins or {}

    def _key(self, env, state):
        node = int(state[0])
        return self.bins.get(node, node)

    def _heuristic(self, env, state):
        return 0.0

    def _simulate_primitive(self, env, node, omega):
        edges = self.edges.get(int(node.state[0]), [])
        if int(omega) >= len(edges):
            return None
        end, cost, duration = edges[int(omega)]
        t = node.env_state['t'] + duration
        if t > env.max_episode_steps:
            return None
        snapshot = {'node': end, 't': t}
        env.set_state(snapshot)
        return snapshot, env.state.copy(), cost, (omega,) * duration, end in env.goals


def test_first_generated_goal_does_not_hide_cheaper_queued_path():
    # Old search returned 100 immediately; even h=0 could not correct it.
    env = GraphEnv()
    planner = GraphSearch({0: [(2, 100., 1), (1, 1., 1)], 1: [(2, 1., 1)]})
    result = planner._plan(env)
    assert result['planned_cost'] == 2.
    assert result['planner_search_stop_reason'] == 'goal_popped'
    assert not result['planner_optimality_guaranteed']
    assert env.get_state() == {'node': 0, 't': 0}


@pytest.mark.parametrize('budget', ['expansions', 'timeout'])
def test_budget_returns_best_incumbent_and_preserves_stop_reason(budget, monkeypatch):
    config = {'max_expansions': 1} if budget == 'expansions' else {'timeout_sec': 1.}
    if budget == 'timeout':
        times = iter([0., 0., 2., 2.])
        monkeypatch.setattr(astar, 'perf_counter', lambda: next(times))
    planner = GraphSearch({0: [(2, 100., 1), (3, 2., 1)]}, **config)
    result = planner._plan(GraphEnv(goals=(2, 3)))
    assert result['planner_success'] and result['planned_cost'] == 2.
    assert result['planner_failure_reason'] == ''
    assert result['planner_search_stop_reason'] == ('max_expansions' if budget == 'expansions' else 'timeout')
    assert not result['planner_optimality_guaranteed']


def test_later_cheaper_arrival_does_not_prune_earlier_feasible_route():
    planner = GraphSearch({0: [(1, 1., 2), (1, 2., 1)], 1: [(2, 1., 2)]})
    result = planner._plan(GraphEnv(horizon=3))
    assert result['planner_success'] and result['planned_cost'] == 3.


def test_nonterminal_pose_bin_cannot_prune_a_real_goal():
    planner = GraphSearch({0: [(1, 1., 1), (2, 2., 1)]}, bins={1: 1, 2: 1})
    result = planner._plan(GraphEnv())
    assert result['planner_success'] and result['planned_cost'] == 2.


def test_later_more_expensive_loop_is_dominated_without_expanding_every_time_layer():
    result = GraphSearch({0: [(0, 1., 1)]})._plan(GraphEnv(horizon=300))
    assert not result['planner_success']
    assert result['expanded_nodes'] == 1
    assert result['planner_search_stop_reason'] == 'open_set_exhausted'


def test_failed_search_retains_failure_reason_and_restores_environment():
    env = GraphEnv()
    result = GraphSearch({0: [(1, 1., 1)]})._plan(env)
    assert not result['planner_success']
    assert result['planner_failure_reason'] == 'open_set_exhausted'
    assert env.get_state() == {'node': 0, 't': 0}


def test_goal_set_heuristic_ignores_sparse_terminal_samples():
    class Env:
        inspection_target = (0., 0.)
        observation_min_distance = .2
        observation_max_distance = 1.
        v = 1.
    planner = astar.HybridAStarController(astar.HybridAStarConfig())
    planner._terminal_states = np.array([[100., 100., np.pi]], np.float32)
    assert planner._heuristic(Env(), np.array([3., 0., 0.])) == pytest.approx(2.)
    assert planner._heuristic(Env(), np.array([.8, 0., np.pi])) == 0.


def test_initial_goal_has_zero_cost_and_no_actions():
    result = GraphSearch({})._plan(GraphEnv(goals=(0,)))
    assert result['planned_cost'] == 0. and result['planned_action_count'] == 0


def test_negative_costs_are_rejected_and_environment_restored():
    env = GraphEnv()
    with pytest.raises(ValueError, match='nonnegative'):
        GraphSearch({0: [(2, -1., 1)]})._plan(env)
    assert env.get_state() == {'node': 0, 't': 0}
