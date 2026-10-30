"""Fixed data-only distance coordinates; a pullback L-infinity pseudometric."""
import numpy as np


class LandmarkProfile:
    def __init__(self,teacher,indices,lift=None):
        self.teacher=teacher;self.indices=np.asarray(indices,np.int64);self.landmarks=teacher.nodes[self.indices]
        assert self.indices.ndim==1 and len(np.unique(self.indices))==len(self.indices)>0
        if lift is None:
            right,cost,_=teacher.attachments(self.landmarks)
            lift=(teacher.distances[:,right]+cost[None]).min(axis=2)
        self.lift=np.asarray(lift,np.float64)
        assert self.lift.shape==(len(teacher.nodes),len(self.indices)) and np.isfinite(self.lift).all()
        self._goal_cache={};self.calls=0

    @classmethod
    def build(cls,teacher,count=128):
        assert 1<=count<=len(teacher.nodes)
        indices=[0];nearest=np.asarray(teacher.distances[:,0]).copy()
        for _ in range(1,count):
            index=int(np.argmax(nearest));assert index not in indices
            indices.append(index);nearest=np.minimum(nearest,teacher.distances[:,index])
        return cls(teacher,indices)

    def profile(self,observations):
        observations=np.asarray(observations,np.float64);left,cost,_=self.teacher.attachments(observations)
        values=(self.lift[left]+cost[:,:,None]).min(axis=1)
        identical=np.all(observations[:,None]==self.landmarks[None],axis=2)
        values[identical]=0
        assert np.isfinite(values).all() and (values>=0).all()
        return values

    def __call__(self,observations,goals):
        observations,goals=np.asarray(observations,np.float64),np.asarray(goals,np.float64)
        assert observations.shape==goals.shape and observations.ndim==2 and observations.shape[1]==2
        self.calls+=1;left=self.profile(observations)
        if np.array_equal(goals,np.broadcast_to(goals[0],goals.shape)):
            key=goals[0].tobytes()
            if key not in self._goal_cache:self._goal_cache[key]=self.profile(goals[:1])
            right=self._goal_cache[key]
        else:right=self.profile(goals)
        return np.max(np.abs(left-right),axis=1)

    def save(self,path):np.savez_compressed(path,indices=self.indices,landmarks=self.landmarks,lift=self.lift)

    @classmethod
    def load(cls,teacher,path):
        with np.load(path) as f:
            np.testing.assert_array_equal(f['landmarks'],teacher.nodes[f['indices']])
            return cls(teacher,f['indices'],f['lift'])
