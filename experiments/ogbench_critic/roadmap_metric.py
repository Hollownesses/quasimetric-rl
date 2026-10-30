"""Data-only auxiliary graph and consistent multi-attachment query metric."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra
from scipy.spatial import cKDTree


class RoadmapMetric:
    def __init__(self,nodes,edges,prototype_indices,spacing=.2,connections=8):
        self.nodes=np.asarray(nodes,np.float64);self.edges=np.asarray(edges,np.int64)
        self.prototype_indices=np.asarray(prototype_indices,np.int64)
        self.spacing=float(spacing);self.connections=int(connections);self.distances=None
        assert self.nodes.ndim==2 and self.nodes.shape[1]==2 and np.isfinite(self.nodes).all()
        assert 1<=self.connections<=len(self.nodes)
        assert self.edges.ndim==2 and self.edges.shape[1]==2
        if len(self.edges):
            weights=np.max(np.abs(self.nodes[self.edges[:,0]]-self.nodes[self.edges[:,1]]),axis=1)/self.spacing
            assert (weights>0).all() and np.isfinite(weights).all()
            i,j=self.edges.T
            self.graph=csr_matrix((np.r_[weights,weights],(np.r_[i,j],np.r_[j,i])),shape=(len(self.nodes),len(self.nodes)))
            assert self.graph.nnz==2*len(self.edges)
        else:self.graph=csr_matrix((len(self.nodes),len(self.nodes)))
        assert connected_components(self.graph,directed=False,return_labels=False)==1,'Disconnected support; no subset fallback'
        self.tree=cKDTree(self.nodes);self._goal_cache={};self.calls=0

    @classmethod
    def build(cls,source,successors,connections=8):
        assert source.shape==successors.shape and source.ndim==2 and source.shape[1]==2
        assert source.dtype==successors.dtype==np.float32
        points=np.concatenate([source,successors]);quantized=np.rint(points/.2).astype(np.int32)
        grid,ids=np.unique(quantized,axis=0,return_inverse=True)
        error=np.square(points.astype(np.float64)-grid[ids]*.2).sum(axis=1)
        order=np.lexsort((np.arange(len(points)),error,ids));sorted_ids=ids[order]
        first=np.r_[True,sorted_ids[1:]!=sorted_ids[:-1]];chosen=order[first]
        assert np.array_equal(ids[chosen],np.arange(len(grid)))
        edges=np.sort(np.column_stack([ids[:len(source)],ids[len(source):]]),axis=1)
        edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0)
        return cls(points[chosen],edges,chosen,connections=connections)

    def save(self,path):
        np.savez_compressed(path,nodes=self.nodes,edges=self.edges,prototype_indices=self.prototype_indices,
                            spacing=self.spacing,connections=self.connections)

    @classmethod
    def load(cls,path,distances=None):
        with np.load(path) as data:
            graph=cls(data['nodes'],data['edges'],data['prototype_indices'],float(data['spacing']),int(data['connections']))
        if distances is not None:graph.distances=np.load(distances,mmap_mode='r')
        return graph

    def precompute(self,path,budget=lambda:None,progress=lambda n:None):
        count=len(self.nodes)
        matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(count,count))
        for start in range(0,count,64):
            budget();end=min(start+64,count)
            block=dijkstra(self.graph,directed=False,indices=np.arange(start,end))
            assert np.isfinite(block).all()
            direct=np.max(np.abs(self.nodes[start:end,None]-self.nodes[None]),axis=-1)/self.spacing
            assert np.all(block>=direct-1e-8)
            matrix[start:end]=block;progress(end)
        matrix.flush();del matrix
        self.distances=np.load(path,mmap_mode='r')
        for start in range(0,count,64):
            budget();np.testing.assert_allclose(self.distances[start:start+64],self.distances[:,start:start+64].T,rtol=1e-10,atol=1e-8)
        np.testing.assert_array_equal(np.diag(self.distances),np.zeros(count))

    def attachments(self,observations):
        assert self.distances is not None
        observations=np.asarray(observations,np.float64)
        assert observations.ndim==2 and observations.shape[1]==2 and np.isfinite(observations).all()
        _,ids=self.tree.query(observations,k=self.connections,p=np.inf)
        ids=np.asarray(ids).reshape(len(observations),self.connections)
        raw=np.max(np.abs(observations[:,None]-self.nodes[ids]),axis=-1)/self.spacing
        local=self.distances[ids[:,:,None],ids[:,None,:]]
        correction=.5*np.maximum((local-raw[:,:,None]-raw[:,None,:]).max(axis=(1,2)),0)
        costs=raw+correction[:,None]
        assert np.all(local<=costs[:,:,None]+costs[:,None,:]+1e-8)
        assert np.all(np.abs(costs[:,:,None]-costs[:,None,:])<=local+1e-8)
        return ids,costs,correction

    def __call__(self,observations,goals):
        observations,goals=np.asarray(observations,np.float64),np.asarray(goals,np.float64)
        assert observations.shape==goals.shape and observations.ndim==2 and observations.shape[1]==2
        self.calls+=1;left,cost_left,_=self.attachments(observations)
        if np.array_equal(goals,np.broadcast_to(goals[0],goals.shape)):
            key=goals[0].tobytes()
            if key not in self._goal_cache:self._goal_cache[key]=self.attachments(goals[:1])[:2]
            right,cost_right=self._goal_cache[key]
        else:right,cost_right,_=self.attachments(goals)
        distance=self.distances[left[:,:,None],right[:,None,:]]
        values=(cost_left[:,:,None]+distance+cost_right[:,None,:]).min(axis=(1,2))
        values[np.all(observations==goals,axis=1)]=0
        assert np.isfinite(values).all() and (values>=0).all()
        return values
