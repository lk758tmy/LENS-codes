"""Independent FDM data for fixed-input source inversion on the disk."""
from __future__ import annotations
import json,time
import numpy as np
from scipy.spatial import cKDTree
from lens import regions
from refs.fdm import CurvedFD

R=.5

class InverseSource:
    def __init__(self,name):
        if name not in ('C','A','P','one'):raise ValueError(name)
        self.name=name
    def __call__(self,x):
        x=np.asarray(x,float)
        if self.name=='C':return 1+.3*np.cos(2*np.pi*x)
        if self.name=='A':return 1+.3*np.sin(2*np.pi*x)+.15*np.cos(6*np.pi*x)
        if self.name=='P':return 1+.2*(x-.5)**2
        return np.ones_like(x)

class Control:
    """Known boundary inputs, fixed independently of any unknown source."""
    def __init__(self,index):self.index=int(index)
    def value(self,x,y):
        x=np.asarray(x,float)-.5;y=np.asarray(y,float)-.5
        if self.index==0:return np.zeros_like(x+y)
        if self.index==1:return .15*(x*x-y*y)
        return .30*x*y
    def gradient(self,x,y):
        x=np.asarray(x,float)-.5;y=np.asarray(y,float)-.5
        if self.index==0:return np.zeros_like(x+y),np.zeros_like(x+y)
        if self.index==1:return .30*x,-.30*y
        return .30*y,.30*x
    def tangent(self,angle):
        x,y=boundary(angle);gx,gy=self.gradient(x,y)
        return -np.sin(angle)*gx+np.cos(angle)*gy

def boundary(angle):
    return .5+R*np.cos(angle),.5+R*np.sin(angle)

def observation_angles():
    return 2*np.pi*np.arange(128)/128,2*np.pi*(np.arange(64)+1/3)/64,2*np.pi*(np.arange(257)+.5)/257

def solve_reference(source,control,n,verbose=False):
    inside=lambda x,y:(np.asarray(x)-.5)**2+(np.asarray(y)-.5)**2<=R**2*(1+1e-13)
    start=time.perf_counter()
    p=CurvedFD(0,1,0,1,n,n,inside,control.value,source)
    p.initial();p.build_seconds=time.perf_counter()-start
    start=time.perf_counter();p.solve(tol=2e-8,verbose=verbose)
    p.fit_seconds=time.perf_counter()-start
    return p

class DiskFlux:
    """Boundary-normal derivative from boundary-vanishing local regression.

    Use the KNOWN quadratic control as g_ext throughout the disk. Since
    u-g_ext vanishes on the boundary, fit u-g_ext=d(x)*a(x),
    d=(r^2-R^2)/(2R), locally. Then q=nu.grad(g_ext)+a at the boundary.
    No analytic solution values inside the disk are queried.
    """
    def __init__(self,p,angles,degree=3,radius_h=6.):
        self.x=p.X[p.intm];self.y=p.Y[p.intm];self.h=p.h
        xy=np.column_stack([self.x,self.y]);tree=cKDTree(xy)
        bx,by=boundary(angles);targets=np.column_stack([bx,by])
        self.weights=[];self.indices=[];self.conditions=[]
        powers=[(i,j) for total in range(degree+1) for i in range(total+1) for j in [total-i]]
        for target,indices in zip(targets,tree.query_ball_point(targets,radius_h*p.h)):
            indices=np.asarray(indices,int)
            dx=(self.x[indices]-target[0])/p.h;dy=(self.y[indices]-target[1])/p.h
            normal_factor=((self.x[indices]-.5)**2+(self.y[indices]-.5)**2-R*R)/(2*R*p.h)
            V=np.column_stack([dx**i*dy**j for i,j in powers])
            w=np.exp(-(dx*dx+dy*dy)/(2*3.**2))
            design=normal_factor[:,None]*V*w[:,None]
            if len(indices)<len(powers)+3:raise ValueError('Insufficient interior flux stencil')
            inverse=np.linalg.pinv(design,rcond=1e-12)
            self.weights.append(inverse[0]*w/p.h);self.indices.append(indices)
            self.conditions.append(float(np.linalg.cond(design)))
        self.bx,self.by=bx,by
    def evaluate(self,p,control):
        extension=control.value(self.x,self.y)
        v=p.U[p.intm]-extension
        gx,gy=control.gradient(self.bx,self.by)
        known=((self.bx-.5)*gx+(self.by-.5)*gy)/R
        return known+np.array([w@v[j] for w,j in zip(self.weights,self.indices)])

def reference_record(source,control,n,angles,degree=3,radius_h=6.):
    p=solve_reference(source,control,n)
    stencil=DiskFlux(p,angles,degree,radius_h)
    q=stencil.evaluate(p,control)
    record=dict(source=source.name,control=control.index,n=n,status=p.status,
                residual=p.residual_inf,minimum_discrete_eigenvalue=p.min_eig(),
                iterations=p.iterations,build_seconds=p.build_seconds,fit_seconds=p.fit_seconds,
                flux_degree=degree,flux_radius_h=radius_h,flux_max_condition=max(stencil.conditions))
    return p,q,record
