"""Model-independent diagnostics copied exactly from the validated E0/E1 tools.

The original tools stay unchanged while E2 uses their frozen source. Fidelity
checks compare these function ASTs and numerical physics outputs to originals.
"""
import copy
from collections import deque
import heapq
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ANCHOR=np.array([10.1156,14.6827])
GOAL=np.array([4.2055,12.0898])
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION
ACTIONS=np.asarray([(x,y) for x in np.linspace(-1,1,11) for y in np.linspace(-1,1,11)],np.float64)

def geometry(env):
    m=env.unwrapped.model
    walls=[]
    for i in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)
        if name and name.startswith('block_'):
            walls.append(np.r_[m.geom_pos[i,:2],m.geom_size[i,:2]])
    radius=float(m.geom_size[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'pointbody'),0])
    return np.asarray(walls),radius

def point_clear(points,walls,radius):
    good=np.ones(len(points),bool)
    for wall in walls:
        diff=np.maximum(np.abs(points-wall[:2])-wall[2:],0)
        good &= (diff*diff).sum(axis=1)>(radius+1e-8)**2
    return good

def segment_clear(a,b,walls,radius):
    """Exact minimum segment-to-axis-aligned-rectangle distance, piecewise quadratic."""
    good=np.ones(len(a),bool)
    direction=b-a
    for wall in walls:
        ids=np.flatnonzero(good)
        if not len(ids): break
        x=a[ids]; v=direction[ids]
        lo=wall[:2]-wall[2:]; hi=wall[:2]+wall[2:]
        with np.errstate(divide='ignore',invalid='ignore'):
            tl=(lo-x)/v; th=(hi-x)/v
        tl=np.where(v==0,0,tl); th=np.where(v==0,0,th)
        breaks=np.sort(np.clip(np.column_stack([np.zeros(len(ids)),np.ones(len(ids)),tl,th]),0,1),axis=1)
        left,right=breaks[:,:-1],breaks[:,1:]
        mid=(left+right)/2
        at_mid=x[:,None,:]+v[:,None,:]*mid[:,:,None]
        active=(at_mid<lo)|(at_mid>hi)
        bound=np.where(at_mid<lo,lo,hi)
        av=np.where(active,v[:,None,:],0)
        bv=np.where(active,x[:,None,:]-bound,0)
        denom=(av*av).sum(axis=-1)
        t=np.clip(-(av*bv).sum(axis=-1)/np.maximum(denom,1e-30),left,right)
        diff=av*t[:,:,None]+bv
        distance2=(diff*diff).sum(axis=-1).min(axis=1)
        good[ids] &= distance2>(radius+1e-8)**2
    return good

def build_graph(walls,radius,spacing=.2):
    axes=np.round(np.arange(-2,22+spacing*.1,spacing),8)
    xx,yy=np.meshgrid(axes,axes,indexing='ij')
    points=np.stack([xx.ravel(),yy.ravel()],axis=1)
    free=point_clear(points,walls,radius)
    ids=np.full(xx.shape,-1,dtype=np.int32)
    ids.flat[np.flatnonzero(free)]=np.arange(free.sum())
    nodes=points[free]
    source=[]; target=[]
    step=int(round(.2/spacing))
    for dx in range(-step,step+1):
        for dy in range(-step,step+1):
            if dx==dy==0:continue
            ix,iy=np.nonzero(ids>=0)
            nx=ix+dx; ny=iy+dy
            keep=(nx>=0)&(ny>=0)&(nx<len(axes))&(ny<len(axes))
            ix,iy,nx,ny=ix[keep],iy[keep],nx[keep],ny[keep]
            keep=ids[nx,ny]>=0
            s,t=ids[ix[keep],iy[keep]],ids[nx[keep],ny[keep]]
            keep=segment_clear(nodes[s],nodes[t],walls,radius)
            source.extend(s[keep]); target.extend(t[keep])
    graph=csr_matrix((np.ones(len(source)),(source,target)),shape=(len(nodes),len(nodes)))
    return nodes,graph

def graph_metrics(nodes,graph,values,goal=GOAL,anchor=ANCHOR,tol=1e-6):
    goals=np.flatnonzero(np.linalg.norm(nodes-goal,axis=1)<=1)
    cost=dijkstra(graph,directed=True,indices=goals,min_only=True,unweighted=True)
    valid=np.isfinite(cost)
    snap_distance,start=cKDTree(nodes).query(anchor)
    assert valid[start]
    # Lowest possible maximum value over ALL graph routes, not just a chosen path.
    peak=np.full(len(nodes),np.inf)
    queue=[]
    for g in goals:
        peak[g]=values[g]; heapq.heappush(queue,(float(values[g]),int(g)))
    while queue:
        value,node=heapq.heappop(queue)
        if value!=peak[node]:continue
        for other in graph.indices[graph.indptr[node]:graph.indptr[node+1]]:
            candidate=max(value,float(values[other]))
            if candidate<peak[other]:
                peak[other]=candidate; heapq.heappush(queue,(candidate,int(other)))
    # Reverse reachability for a strictly descending route to the success region.
    reachable=np.zeros(len(nodes),bool); reachable[goals]=True
    q=deque(goals)
    while q:
        node=q.popleft()
        for other in graph.indices[graph.indptr[node]:graph.indptr[node+1]]:
            if not reachable[other] and values[other]>values[node]+tol:
                reachable[other]=True; q.append(other)
    # Best peak among shortest graph routes, and a concrete route for replay.
    shortest_peak=np.full(len(nodes),np.inf); shortest_peak[goals]=values[goals]
    descending_optimal=np.zeros(len(nodes),bool)
    for node in np.argsort(cost):
        if not valid[node] or cost[node]==0:continue
        ns=graph.indices[graph.indptr[node]:graph.indptr[node+1]]
        ns=ns[cost[ns]==cost[node]-1]
        shortest_peak[node]=max(float(values[node]),float(shortest_peak[ns].min()))
        descending_optimal[node]=np.any(values[ns]<values[node]-tol)
    path=[int(start)]
    while cost[path[-1]]>0:
        node=path[-1]
        ns=graph.indices[graph.indptr[node]:graph.indptr[node+1]]
        ns=ns[cost[ns]==cost[node]-1]
        path.append(int(ns[np.argmin(shortest_peak[ns])]))
    iqr=float(np.subtract(*np.quantile(values[valid],[.75,.25])))
    corridor=valid&(nodes[:,0]>=9.7)&(nodes[:,0]<=12.3)&(nodes[:,1]>=9.2)&(nodes[:,1]<=15)
    stats=dict(nodes=len(nodes),edges=graph.nnz,anchor_node=nodes[start].tolist(),snap_distance=float(snap_distance),
        reference_steps=float(cost[start]),anchor_value=float(values[start]),
        minimum_peak=float(peak[start]),minimum_peak_gap=float(peak[start]-values[start]),
        minimum_peak_gap_over_iqr=float((peak[start]-values[start])/max(iqr,1e-9)),
        shortest_route_peak=float(shortest_peak[start]),strict_descent_from_anchor=bool(reachable[start]),
        strict_descent_reachable_fraction=float(reachable[valid].mean()),
        shortest_route_descent_opportunity_fraction=float(descending_optimal[valid&(cost>0)].mean()),
        distance_iqr=iqr,all_graph_spearman=float(spearmanr(cost[valid],values[valid]).statistic),
        corridor_spearman=float(spearmanr(cost[corridor],values[corridor]).statistic))
    return stats,dict(nodes=nodes,values=values,reference=cost,reachable=reachable,
                      path=nodes[path],path_values=values[path],path_costs=cost[path],walls=None)

def wrappers(env):
    layers=[]
    while True:
        if '_elapsed_steps' in env.__dict__:
            layers.append(env)
        if not hasattr(env,'env'):
            return layers
        env=env.env

def capture(env):
    return copy.copy(env.unwrapped.data),[w._elapsed_steps for w in wrappers(env)]

def restore(env,state):
    mujoco.mj_copyData(env.unwrapped.data,env.unwrapped.model,state[0])
    for w,n in zip(wrappers(env),state[1]):
        w._elapsed_steps=n

def save_integration(env):
    state=np.empty(mujoco.mj_stateSize(env.unwrapped.model,STATE_SPEC))
    mujoco.mj_getState(env.unwrapped.model,env.unwrapped.data,state,STATE_SPEC)
    return state

def load_case(env,case):
    env.reset(seed=int(case['seed']),options=dict(task_id=int(case['task'])))
    env.unwrapped.set_goal(goal_xy=case['goal'])
    mujoco.mj_setState(env.unwrapped.model,env.unwrapped.data,case['integration'],STATE_SPEC)
    mujoco.mj_forward(env.unwrapped.model,env.unwrapped.data)
    for w in wrappers(env): w._elapsed_steps=0
    assert np.max(np.abs(env.unwrapped.get_ob()-case['observation']))<1e-10

def read_bank(path):
    data=np.load(path)
    names=sorted({k.split('__')[0] for k in data.files})
    return {name:{k.split('__')[1]:data[k] for k in data.files if k.startswith(name+'__')} for name in names}

def candidates(env,actions=ACTIONS):
    state=capture(env)
    observations=[]
    for action in actions:
        restore(env,state)
        observations.append(env.step(action)[0])
    restore(env,state)
    return np.array(observations),state


def rollout(env,case,score=None,actor=None):
    """Full official episode; caller supplies only model scoring or actor inference."""
    assert (score is None)!=(actor is None)
    load_case(env,case);goal=case['goal'];positions=[env.unwrapped.get_ob().copy()]
    actions=[];goal_distances=[];success=False;start=time.time()
    for step in range(1000):
        if score is not None:
            next_obs,_=candidates(env)
            values=np.asarray(score(next_obs,np.broadcast_to(goal,next_obs.shape)))
            assert values.shape==(len(ACTIONS),) and np.isfinite(values).all()
            index=int(np.argmin(values));action=ACTIONS[index]
            obs,_,term,trunc,info=env.step(action)
            assert np.array_equal(obs,next_obs[index]),'State restoration changed successor'
            goal_distances.append(float(values[index]))
        else:
            action=np.asarray(actor(positions[-1],goal,step))
            assert action.shape==(2,) and np.isfinite(action).all()
            obs,_,term,trunc,info=env.step(action)
        actions.append(action.copy());positions.append(obs.copy())
        if term or trunc:success=bool(info['success']);break
    assert term or trunc
    return dict(success=success,steps=len(actions),seconds=time.time()-start,
        final_xy=positions[-1].tolist(),max_last100_displacement=float(np.ptp(np.asarray(positions[-100:]),axis=0).max())),dict(
        observations=np.asarray(positions),actions=np.asarray(actions),goal=np.asarray(goal),distances=np.asarray(goal_distances))


def replay_reference(env,path,goal=GOAL,anchor=ANCHOR):
    """Same graph-route simulator verification as the original field tool."""
    np.random.seed(3100);env.action_space.seed(3100);env.reset(seed=3100,options=dict(task_id=3))
    env.unwrapped.set_goal(goal_xy=goal);env.unwrapped.set_xy(anchor)
    done=False;trace=[anchor.copy()];tracking=[]
    for target in path:
        delta=target-env.unwrapped.get_ob()
        if np.max(np.abs(delta))<1e-8:continue
        action=np.clip(delta/.2,-1,1)
        ob,_,term,trunc,info=env.step(action);tracking.append(float(np.max(np.abs(ob-target))));trace.append(ob.copy())
        if term or trunc:done=True;break
    if not done:
        ob,_,term,trunc,info=env.step(np.zeros(2));trace.append(ob.copy());done=term or trunc
    assert done and info['success']
    return dict(reference_replay_success=True,reference_replay_steps=len(trace)-1,
                reference_replay_max_tracking_error=max(tracking,default=0.)),np.asarray(trace)

