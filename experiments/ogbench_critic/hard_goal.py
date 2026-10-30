"""Finite max-pool teacher fitting; no gradient through discrete goal selection."""
import jax
import jax.numpy as jnp
import numpy as np
from goal_support import GoalSupport


@jax.jit
def paired_errors(agent,observations,goals,targets):
    value=agent.network.select('value');f=value(observations,goals);r=value(goals,observations)
    return .5*(jnp.square(f-targets)+jnp.square(r-targets))


def select_worst(batch,goals,targets,errors):
    goals=np.asarray(goals);targets=np.asarray(targets);errors=np.asarray(errors)
    assert goals.shape[:2]==targets.shape==errors.shape and goals.shape[2]==2
    assert goals.shape[0]==len(batch['observations'])//2 and np.isfinite(errors).all()
    indices=np.argmax(errors,axis=1);rows=np.arange(len(indices))
    selected=dict(batch);selected['value_goals']=batch['value_goals'].copy()
    selected['value_goals'][1::2]=goals[rows,indices]
    return selected,targets[rows,indices],indices


class HardGoal(GoalSupport):
    def __init__(self,nodes,candidates=8):
        super().__init__(nodes);self.candidates=int(candidates)
        assert 1<=self.candidates<=len(self.nodes)

    def proposals(self,batch,start):
        uniform=super().__call__(batch,start);count=len(batch['observations'])//2
        phase=np.arange(self.candidates)*(len(self.nodes)//self.candidates)
        ids=((np.arange(start,start+count)[:,None]+phase[None])*self.stride)%len(self.nodes)
        goals=self.nodes[ids]+self.offsets[start:start+count,None,2:]
        np.testing.assert_array_equal(goals[:,0],uniform['value_goals'][1::2])
        sources=np.repeat(uniform['observations'][1::2],self.candidates,axis=0)
        return uniform,sources,goals,ids

    def choose(self,agent,batch,start,teacher):
        uniform,sources,goals,ids=self.proposals(batch,start)
        labels=teacher(sources,goals.reshape(-1,2)).astype(np.float32).reshape(goals.shape[:2])
        errors=np.asarray(paired_errors(agent,sources,goals.reshape(-1,2),labels.reshape(-1))).reshape(labels.shape)
        selected,chosen,indices=select_worst(uniform,goals,labels,errors)
        targets=np.empty(len(batch['observations']),np.float32)
        targets[::2]=teacher(uniform['observations'][::2],uniform['value_goals'][::2]).astype(np.float32);targets[1::2]=chosen
        separation=np.max(np.abs(selected['observations'][1::2]-selected['value_goals'][1::2]),axis=1)
        info=dict(pool_mean_squared_error=float(errors.mean()),pool_max_squared_error=float(errors.max(axis=1).mean()),
            selected_column_zero_fraction=float((indices==0).mean()),selected_geometric_cost_mean=float((separation/.2).mean()),
            selected_separation_below_one_fraction=float((separation<1).mean()))
        return selected,targets,info,dict(uniform=uniform,goals=goals,labels=labels,indices=indices,prototype_indices=ids)


def pool_statistics(agent,sources,goals,labels):
    flat_goals=goals.reshape(-1,2);flat_labels=labels.reshape(-1);assert len(sources)==len(flat_goals)
    errors=[]
    for start in range(0,len(sources),1024):
        errors.append(np.asarray(paired_errors(agent,sources[start:start+1024],flat_goals[start:start+1024],flat_labels[start:start+1024])))
    errors=np.concatenate(errors).reshape(labels.shape)
    return dict(pool_mean_squared_error=float(errors.mean()),pool_max_squared_error=float(errors.max(axis=1).mean()),
                pool_max_rmse=float(np.sqrt(errors.max(axis=1).mean())),pool_absolute_max_squared_error=float(errors.max()))
