"""Training-side loader: only transitions, production quotient and fixed B1 pairs.

Does not import the auditor or load its graph-distance files. Compatible with
IndexedIQE / training_objective's data interface, with a fresh initialization.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import torch

from experiments.four_path_factorial.run import Replay, canonical_nodes, tensor_digest
from quasimetric_rl.data import MultiEpisodeData
from quasimetric_rl.modules.quasimetric_critic.models.encoder import Encoder
from minimal_qrl.global_push_sampling import replay_fingerprint, configure_global_push_sampling
from minimal_qrl.path_support import validate_path_only_replay
from minimal_qrl.eval.comm_inspection_oracle_bank import environment_signature
from .build import make_env, sha256


def load_data(root, *, task_mode='all', seed=42):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    for name in ('verified_replay.pt','scenario.json'):
        if sha256(root/name) != manifest['files'][name]:
            raise ValueError(f'file fingerprint mismatch: {name}')
    payload=torch.load(root/'verified_replay.pt',map_location='cpu',weights_only=True)
    raw=MultiEpisodeData(**payload['raw'])
    if replay_fingerprint(raw) != payload['sha256'] or payload['sha256'] != manifest['replay_sha256']:
        raise ValueError('replay fingerprint mismatch')
    env=make_env(json.loads((root/'scenario.json').read_text()))
    if environment_signature(env)['digest'] != manifest['environment_signature']['digest']:
        raise ValueError('environment signature mismatch')
    replay=Replay(raw,env)
    configure_global_push_sampling(replay,state_mode='trajectory',task_mode=task_mode,seed=seed)
    validate_path_only_replay(raw,batch_size=1)
    nodes,ids,mapping=canonical_nodes(raw.all_observations)
    src_obs=torch.arange(len(replay))+replay.indices_to_episode_indices
    src,dst=ids[src_obs],ids[src_obs+1]
    physical=~raw.transition_infos['abstract_goal_edge'].bool();rows=physical.nonzero().flatten()
    goals=Encoder._canonicalize_comm_goal_set(None,raw.transition_infos['task_goal_observations'][rows])
    goals[goals==0]=0
    goal=torch.tensor([mapping[g.numpy().tobytes()] for g in goals])
    pair_obs=raw.transition_infos['global_push_pair_goal_obs_index'][rows]
    data=SimpleNamespace(raw=raw,replay=replay,env=env,nodes=nodes,obs_ids=ids,src_obs=src_obs,
        src=src,dst=dst,cost=-raw.rewards,physical=physical,rows=rows,task_src=src[rows],
        goal=goal,pair=ids[pair_obs],task_mode=task_mode)
    data.fingerprint=dict(replay=payload['sha256'],file=manifest['files']['verified_replay.pt'],
        nodes=tensor_digest(nodes,ids),pairs=tensor_digest(data.task_src,goal,data.pair),
        costs=tensor_digest(data.cost),task_mask=tensor_digest(raw.transition_infos['global_push_task_mask']),
        b1_seed=seed)
    return data
