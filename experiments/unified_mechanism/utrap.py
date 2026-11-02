"""Read-only Torch analysis. Run with the U-trap environment, see run.sh."""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from metrics import path_metrics, ranking, read, sha, stats, verified, write_rows
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.connected_expert_utrap.data import load_training_data
from experiments.connected_expert_utrap.reference import validate_reference
from experiments.r_series_mppi.common import DATA, TRAIN, adapter, CONFIG, ORACLE_CONFIG
from minimal_qrl.baselines.hybrid_astar_value import HybridAStarValueOracle
from minimal_qrl.baselines.mppi import MPPIController, simulate_action_sequences


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    mp = root/'results/r_series_mppi_20261026/main'
    ref = root/'results/connected_reference_fix_20261024/main/graph_reference_audit_only.npz'
    data = load_training_data(root/DATA)
    cache = next((mp/'oracle_cache').glob('*.npz'))
    identities = {}
    def check(p, expected):
        identities[str(p.relative_to(root))] = verified(p, expected)
    check(root/DATA/'verified_replay.pt', '8c221bb5e626de8cabdec381d2d440293ce1460225cea411c2222fcde5f26d77')
    check(ref, 'cea569cd7a60e00608330dd35866986d2909271aab6af2dbafe23d349308d80f')
    check(cache, '2a01ed8b720ef47430c5766d900c7c8a3704535d986898f5747b6451270bbcf7')
    refdata = np.load(ref)
    v, next_edge = refdata['distance'], refdata['next_edge']
    goal_id, = validate_reference(data, v)
    nodes, src, dst, cost = [x.numpy() for x in (data.nodes,data.src,data.dst,data.cost)]
    physical = data.original.numpy()
    for key, x in [('source',src),('target',dst),('physical',data.physical.numpy())]:
        np.testing.assert_array_equal(refdata[key],x)
    _, indices = np.unique(np.column_stack((src[physical],dst[physical],cost[physical])), axis=0,return_index=True)
    edges = physical[indices]
    by_source = defaultdict(list)
    for i in edges:
        by_source[src[i]].append(i)
    env = data.env
    def reset(p):
        env.reset(seed=0,options=dict(device_id='u_trap_target',start=np.array(p,np.float32)))
    starts = read(mp/'states.json')['records']
    reset(starts[0]['pose'])
    goal = env.abstract_goal_observation()
    tmp = tempfile.TemporaryDirectory()
    shutil.copy2(cache,tmp.name)
    oracle = HybridAStarValueOracle(ORACLE_CONFIG,cache_dir=Path(tmp.name))
    assert oracle.begin_episode(env,seed=0)['oracle_value_cache_hit']
    metrics, paths, decisions = [], [], []
    for name, expected in [('R0','4225998ede46ac2cb34ab4c6114d9989b413f682863ebc498fa9e709db772c44'),
                           ('R1','6751085907c40ea7b8a4d087b95c74e124e6773133661b1f8b5d1c50a27db7c8')]:
        ck = root/TRAIN/name/'final.pt'
        check(ck,expected)
        model,_ = adapter(data.small,ck)
        def score(x,y):
            return np.concatenate([model.batch_value(x[i:i+512],y[i:i+512]) for i in range(0,len(x),512)]).astype(float)
        def task(x):
            return score(x,np.repeat(goal[None],len(x),axis=0))
        def summary(scope,metric,x):
            metrics.append(dict(scene='U',model=name,scope=scope,metric=metric,**stats(x)))
        j = task(nodes)
        assert j[goal_id]==0
        d = score(nodes[src],nodes[dst])
        h = d-cost
        for scope, ix in [('graph_records',physical),('graph_unique',edges)]:
            summary(scope,'h',h[ix])
            summary(scope,'h/c',h[ix]/cost[ix])
            summary(scope,'tau',d[ix]+j[dst[ix]]-j[src[ix]])
        for s, ids in by_source.items():
            ids=np.array(ids)
            q=cost[ids]+j[dst[ids]]
            row=dict(model=name,scope='graph',case=s, b=q.min()-j[s],b_norm=(q.min()-j[s])/np.median(cost[ids]))
            if len(ids)>1:
                row.update(ranking(cost[ids],j[dst[ids]],v[dst[ids]]))
            decisions.append(row)
            route=[]
            node=s
            while node!=goal_id:
                e=next_edge[node]
                assert e>=0 and src[e]==node and v[dst[e]]<v[node]
                route.append(e)
                node=dst[e]
            assert abs(cost[route].sum(dtype=float)-v[s])<1e-8
            paths.append(dict(model=name,scope='graph_shortest',case=s,
                         **path_metrics(cost[route],d[route],j[np.r_[s,dst[route]]])))
        for st in starts:
            pos=st['pose']
            costs,obs,poses=[],[],[]
            for a in np.linspace(-3,3,13):
                reset(pos)
                ob, reward, term, trunc, info=env.step(float(a))
                if not (info['collision'] or info['out_of_bounds']):
                    costs.append(-reward); obs.append(ob);poses.append(env.state.copy())
            reset(pos)
            j0=task(np.array([env.state_to_observation(env.state)]))[0]
            if costs:
                c=np.array(costs); pred=task(np.array(obs)); refv=oracle.batch_value(env,np.array(poses))
                ok=refv<1e6
                row=dict(model=name,scope='start_1',case=st['id'],legal=len(c),finite=int(ok.sum()),
                         b=(c+pred).min()-j0,b_norm=((c+pred).min()-j0)/np.median(c))
                if ok.any(): row.update(ranking(c[ok],pred[ok],refv[ok]))
                decisions.append(row)
        episode_path=mp/'arms'/name/'episodes.jsonl'
        identities[str(episode_path.relative_to(root))]=sha(episode_path)
        for line in episode_path.read_text().splitlines():
            ep=json.loads(line)
            key=f"{ep['state_id']}_{ep['planner_seed']}"
            p=mp/'arms'/name/(key+'.npz')
            tr=np.load(p)
            reset(ep['pose'])
            ob=np.array([env.state_to_observation(s) for s in tr['states']])
            c=-tr['rewards']; vseq=task(ob)
            end='success' if ep['success'] else 'collision'
            paths.append(dict(model=name,scope='mppi_'+end,case=key,
                              **path_metrics(c,score(ob[:-1],ob[1:]),vseq)))
            planner=MPPIController(CONFIG,terminal_mode='qrl',qrl_agent=model)
            planner.begin_episode(env,goal,ep['planner_seed'])
            bank=planner.sample_candidates(env)
            assert hashlib.sha256(bank.tobytes()).hexdigest()==ep['first_candidate_sha256']
            roll=simulate_action_sequences(env,env.state,bank)
            np.testing.assert_array_equal(roll['final_states'],tr['candidate_final_states'][0])
            q,w=planner.score_candidates(env,roll)
            u=float(np.sum(w*bank[:,0]))
            np.testing.assert_allclose(u,tr['actions'][0,0],atol=1e-6,rtol=0)
            oracle_record=np.load(mp/'arms/oracle'/(key+'.npz'))
            np.testing.assert_array_equal(oracle_record['candidate_final_states'][0],roll['final_states'])
            n=oracle_record['query_counts'][0]
            rv=np.zeros(len(bank))
            rv[~(roll['invalid']|roll['success'])]=oracle_record['query_values'][:n]
            full=roll['costs'].astype(float)+rv+1e6*roll['invalid']
            weights=np.exp(-(full-full.min()))
            uref=np.average(bank[:,0],weights=weights)
            np.testing.assert_allclose(uref,oracle_record['actions'][0,0],atol=1e-6,rtol=0)
            ok=(~roll['invalid']) & (rv<1e6)
            row=dict(model=name,scope='mppi_bank',case=key,region=ep['stratum'],outcome=end,
                     action=u,oracle_action=uref,action_absdiff=abs(u-uref),legal=int((~roll['invalid']).sum()),
                     finite=int(ok.sum()),oracle_success=int(read(mp/'arms/oracle'/(key+'.json'))['summary']['success']))
            if ok.any():
                row.update(ranking(roll['costs'][ok],(q-roll['costs'])[ok],rv[ok]))
            decisions.append(row)
        print(name,'done',flush=True)
    write_rows(out/'metrics.csv',metrics)
    write_rows(out/'paths.csv',paths)
    write_rows(out/'decisions.csv',decisions)
    (out/'inputs.json').write_text(json.dumps(identities,indent=2)+'\n')
    tmp.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();run(args.root.resolve(),args.out)
