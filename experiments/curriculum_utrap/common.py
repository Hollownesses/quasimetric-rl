from pathlib import Path
import hashlib
import json
from minimal_qrl.envs import CommInspectionDubinsUAV2D
from minimal_qrl.industry_exp.scalability_scenarios import scenario_to_env_kwargs

ROOT=Path(__file__).resolve().parents[2]
SCENARIO=ROOT/'results/full_replay_four_path_20261017/main/scenario.json'
DEVICE='u_trap_target'
FIELDS=('episode_lengths','all_observations','actions','rewards','terminals','timeouts',
        'observation_infos','transition_infos')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def make_env(scenario):
    env=CommInspectionDubinsUAV2D(**scenario_to_env_kwargs(scenario))
    env.set_task_by_device_id(DEVICE)
    return env
