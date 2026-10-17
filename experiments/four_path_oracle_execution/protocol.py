"""Frozen Oracle / five repaired QRL / no-terminal execution comparison."""
from dataclasses import replace
from experiments.four_path_execution.common import (
    ROOT, SEEDS, CONFIG as PREVIOUS_CONFIG, TimedMPPIController, adapter, checkpoint,
)
from minimal_qrl.baselines.hybrid_astar import HybridAStarConfig
from minimal_qrl.baselines.hybrid_astar_value import HybridAStarValueOracle

HISTORICAL_COMMIT='34d1c011f637b3352ac1dc1134650f6f28c20b17'
PREVIOUS_ROOT=ROOT/'results/four_path_execution_20261017/main'
ARMS=('none','oracle')+tuple(f'qp_{seed}' for seed in SEEDS)
CONFIG=replace(PREVIOUS_CONFIG,horizon=10,num_samples=128)
ORACLE_CONFIG=HybridAStarConfig()


def controller_factory(root):
    def make(data,arm):
        if arm=='none':return TimedMPPIController(CONFIG,terminal_mode='none')
        if arm=='oracle':
            return TimedMPPIController(CONFIG,terminal_mode='oracle',
                terminal_value_provider=HybridAStarValueOracle(ORACLE_CONFIG,cache_dir=root/'oracle_cache'))
        assert arm in ARMS
        adapted,_=adapter(data,checkpoint(arm))
        return TimedMPPIController(CONFIG,terminal_mode='qrl',qrl_agent=adapted)
    return make
