"""Oracle port, cache/goal semantics, and unchanged MPPI sampling contract."""
import importlib.util
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from experiments.four_path_execution.common import load_data, read
from experiments.four_path_oracle_execution.protocol import CONFIG, ORACLE_CONFIG, PREVIOUS_ROOT, HISTORICAL_COMMIT
from minimal_qrl.baselines.hybrid_astar_value import HybridAStarValueOracle
from minimal_qrl.baselines.mppi import MPPIController, simulate_action_sequences
from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.test_comm_inspection_baselines import make_env


class ConstantTerminalValue:
    def begin_episode(self,env,*,seed):self.queries=0;return {}
    def batch_value(self,env,states):
        self.queries+=len(states);return np.full(len(states),7.,np.float32)
    def end_episode(self):return {'oracle_value_queries':self.queries}


def test_historical_oracle_port_is_exact():
    actual=Path('minimal_qrl/baselines/hybrid_astar_value.py').read_bytes()
    assert actual==subprocess.check_output(['git','show',HISTORICAL_COMMIT+':minimal_qrl/baselines/hybrid_astar_value.py'])


def test_hybrid_astar_value_oracle_builds_and_reuses_exact_lattice_table(tmp_path):
    env=make_env(bounds=(0.,0.,6.,6.),start=(2.,4.,0.));env.reset(seed=11)
    cfg=HybridAStarConfig(position_resolution=.5,heading_bins=12,primitive_steps=2,
                         primitive_scales=(-1.,0.,1.),terminal_samples=0)
    oracle=HybridAStarValueOracle(cfg,cache_dir=tmp_path);diag=oracle.begin_episode(env,seed=11)
    terminal=env.sample_task_terminal_state(seed=91)
    grid,_=oracle._grid(env)
    reachable=np.flatnonzero(np.isfinite(oracle._active_values)&(oracle._active_values>0))[0]
    states=np.stack([grid[reachable],terminal,[-1.,-1.,0.]])
    values=oracle.batch_value(env,states)
    assert not diag['oracle_value_cache_hit'] and diag['oracle_value_reachable_fraction']>0
    assert 0<values[0]<1e6 and values[1]==0 and values[2]==1e6
    reloaded=HybridAStarValueOracle(cfg,cache_dir=tmp_path)
    assert reloaded.begin_episode(env,seed=999)['oracle_value_cache_hit']
    np.testing.assert_array_equal(values,reloaded.batch_value(env,states))
    # Running costs must invalidate the cache, not silently reuse a prior table.
    env.observation_failure_cost+=.1
    changed=reloaded.begin_episode(env,seed=11)
    assert not changed['oracle_value_cache_hit'] and changed['oracle_value_table_digest']!=diag['oracle_value_table_digest']


def test_provider_scoring_and_historical_controller_match(tmp_path):
    path=tmp_path/'legacy_mppi.py'
    path.write_bytes(subprocess.check_output(['git','show',HISTORICAL_COMMIT+':minimal_qrl/baselines/mppi.py']))
    spec=importlib.util.spec_from_file_location('legacy_oracle_mppi',path);legacy=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    env=load_data().env;env.reset(seed=8,options={'device_id':'u_trap_target','start':[4.,3.65,3.0]})
    current=MPPIController(CONFIG,terminal_mode='oracle',terminal_value_provider=ConstantTerminalValue())
    old=legacy.MPPIController(legacy.MPPIConfig(**asdict(CONFIG)),terminal_mode='oracle',terminal_value_provider=ConstantTerminalValue())
    none=MPPIController(CONFIG,terminal_mode='none')
    for c in (current,old,none):c.begin_episode(env,env.abstract_goal_observation(),seed=8)
    # Exact additive cost of seven, not seven times dt; identical candidates.
    actions=none.sample_candidates(env);rollout=simulate_action_sequences(env,env.state,actions)
    costs,_=current.score_candidates(env,rollout);base,_=none.score_candidates(env,rollout)
    np.testing.assert_array_equal(costs-base,7.*(~(rollout['success']|rollout['invalid'])))
    for _ in range(5):
        a,d=current.act(None,env);b,e=old.act(None,env)
        np.testing.assert_array_equal(a,b);assert d['best_rollout_cost']==e['best_rollout_cost']
        assert d['model_rollouts']==e['model_rollouts']==128
        env.step(a)


def test_only_requested_mppi_configuration_changes():
    previous=read(PREVIOUS_ROOT/'manifest.json')['config'];expected=dict(previous,horizon=10,num_samples=128)
    assert asdict(CONFIG)==expected
    assert ORACLE_CONFIG.position_resolution==.25 and ORACLE_CONFIG.heading_bins==24
    assert ORACLE_CONFIG.primitive_steps==5
