"""Transparent environment recording/deadline adapter for official evaluation."""
import numpy as np


class EpisodeRecorder:
    def __init__(self,env,check=lambda:None,integration=None,on_end=lambda count:None):
        self.env=env;self.check=check;self.integration=integration;self.on_end=on_end;self.starts=[];self.ends=[]

    def __getattr__(self,name):return getattr(self.env,name)

    def reset(self,*args,**kwargs):
        self.check();result=self.env.reset(*args,**kwargs);observation,info=result
        self.starts.append(dict(observation=np.asarray(observation).copy(),goal=np.asarray(info['goal']).copy()))
        return result

    def step(self,action):
        self.check();result=self.env.step(action)
        if result[2] or result[3]:
            self.ends.append(None if self.integration is None else self.integration(self.env));self.on_end(len(self.ends))
        return result
