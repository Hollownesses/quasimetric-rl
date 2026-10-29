"""Independent collision-checked graph and scale-aware field diagnostics (eval only)."""
import argparse
from collections import deque
import heapq
import json
from pathlib import Path

from common import load_base,load_data,distances
import mujoco
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from utils.flax_utils import restore_agent

ANCHOR=np.array([10.1156,14.6827])
GOAL=np.array([4.2055,12.0898])


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


def graph_metrics(nodes,graph,values,goal=GOAL,anchor=ANCHOR):
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
    tol=1e-6
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


def run(output,checkpoint=None,epoch=None,spacing=.2,radius_delta=0.):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    agent,config=load_base()
    if checkpoint: agent=restore_agent(agent,str(checkpoint),epoch)
    env,_,_=load_data(config)
    walls,radius=geometry(env)
    radius+=radius_delta
    nodes,graph=build_graph(walls,radius,spacing)
    vals=[]
    for chunk in np.array_split(nodes,max(1,int(np.ceil(len(nodes)/1024)))):
        vals.extend(np.asarray(distances(agent,chunk,np.broadcast_to(GOAL,chunk.shape))))
    stats,data=graph_metrics(nodes,graph,np.asarray(vals))
    data['walls']=walls
    np.savez_compressed(output/'field.npz',**data)
    # Replay graph route with actual simulator. Connect actual anchor to snapped start.
    np.random.seed(3100); env.action_space.seed(3100)
    env.reset(seed=3100,options=dict(task_id=3))
    env.unwrapped.set_goal(goal_xy=GOAL)
    env.unwrapped.set_xy(ANCHOR)
    done=False; trace=[ANCHOR.copy()]; tracking=[]
    for target in data['path']:
        delta=target-env.unwrapped.get_ob()
        if np.max(np.abs(delta))<1e-8:continue
        action=np.clip(delta/.2,-1,1)
        ob,_,term,trunc,info=env.step(action)
        tracking.append(float(np.max(np.abs(ob-target))))
        trace.append(ob.copy())
        if term or trunc: done=True;break
    if not done:
        ob,_,term,trunc,info=env.step(np.zeros(2));trace.append(ob.copy());done=term or trunc
    stats.update(spacing=spacing,radius=radius,reference_replay_success=bool(info['success']),
        reference_replay_steps=len(trace)-1,reference_replay_max_tracking_error=max(tracking,default=0.),
        goal=GOAL.tolist(),anchor=ANCHOR.tolist(),checkpoint=str(checkpoint) if checkpoint else 'official100k')
    np.savez_compressed(output/'reference_replay.npz',observations=trace,goal=GOAL)
    (output/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
    env.close()
    print(json.dumps(stats,indent=2),flush=True)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--epoch',type=int)
    p.add_argument('--spacing',type=float,default=.2)
    p.add_argument('--radius-delta',type=float,default=0.)
    a=p.parse_args()
    run(a.output,a.checkpoint,a.epoch,a.spacing,a.radius_delta)
