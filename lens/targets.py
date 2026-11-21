"""Coherent target geometry: signed boundary distance, area and moments.

Polygons retain every vertex. Smooth targets use analytic parameter curves and
Newton closest-point solves; sampling is not a replacement fitted polygon.
"""
from __future__ import annotations
import math
import numpy as np
import torch

class Target:
    def __init__(self,ps,qs,name='polygon',nsub=None):
        P=np.column_stack([np.asarray(ps,float).ravel(),np.asarray(qs,float).ravel()])
        if len(P)>1 and np.linalg.norm(P[0]-P[-1])<1e-14:P=P[:-1]
        P=P[np.linalg.norm(P-np.roll(P,1,axis=0),axis=1)>1e-14]
        assert len(P)>=3
        cross=P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1]
        if cross.sum()<0:P=P[::-1]
        self.name=name;self.ps,self.qs=P.T
        self._bx=torch.tensor(self.ps);self._by=torch.tensor(self.qs)
        self._ex=torch.roll(self._bx,-1)-self._bx;self._ey=torch.roll(self._by,-1)-self._by
        self._l2=self._ex**2+self._ey**2
        self.spec=dict(kind='polygon',name=name,vertices=P.tolist())
    @property
    def area(self):return self.moments()[0]
    @property
    def pmin(self):return float(np.min(self.ps))
    @property
    def pmax(self):return float(np.max(self.ps))
    @property
    def qmin(self):return float(np.min(self.qs))
    @property
    def qmax(self):return float(np.max(self.qs))
    @property
    def per(self):return float(np.hypot(np.roll(self.ps,-1)-self.ps,np.roll(self.qs,-1)-self.qs).sum())
    def center(self):return tuple(self.moments()[1])
    def moments(self):
        x,y=self.ps,self.qs;X,Y=np.roll(x,-1),np.roll(y,-1);c=x*Y-X*y
        A=.5*c.sum();mean=np.array([np.sum((x+X)*c),np.sum((y+Y)*c)])/(6*A)
        xx=np.sum((x*x+x*X+X*X)*c)/12/A;yy=np.sum((y*y+y*Y+Y*Y)*c)/12/A
        xy=np.sum((2*x*y+x*Y+X*y+2*X*Y)*c)/24/A
        return float(A),mean,np.array([[xx,xy],[xy,yy]])-np.outer(mean,mean)
    def at(self,s):
        seg=np.hypot(np.roll(self.ps,-1)-self.ps,np.roll(self.qs,-1)-self.qs)
        cum=np.r_[0,np.cumsum(seg)];s=np.mod(np.asarray(s,float),cum[-1])
        j=np.clip(np.searchsorted(cum,s,side='right')-1,0,len(seg)-1)
        f=(s-cum[j])/seg[j];k=(j+1)%len(seg)
        return self.ps[j]+f*(self.ps[k]-self.ps[j]),self.qs[j]+f*(self.qs[k]-self.qs[j])
    def boundary(self,n,include_vertices=False):
        s=(np.arange(n)+.5)*self.per/n
        if include_vertices:
            seg=np.hypot(np.roll(self.ps,-1)-self.ps,np.roll(self.qs,-1)-self.qs)
            s=np.unique(np.r_[s,np.r_[0,np.cumsum(seg)[:-1]]])
        return self.at(s)
    def _dist(self,px,py):
        s=((px[None]-self._bx[:,None])*self._ex[:,None]+(py[None]-self._by[:,None])*self._ey[:,None])/self._l2[:,None]
        s=s.clamp(0,1);cx=self._bx[:,None]+s*self._ex[:,None];cy=self._by[:,None]+s*self._ey[:,None]
        d2,j=((px[None]-cx)**2+(py[None]-cy)**2).min(0);ar=torch.arange(px.numel())
        dx=px-cx[j,ar];dy=py-cy[j,ar];norm=torch.sqrt(d2).clamp(min=1e-30)
        x1=self._bx[:,None];y1=self._by[:,None];x2=x1+self._ex[:,None];y2=y1+self._ey[:,None]
        crossing=((y1>py[None])!=(y2>py[None])) & (px[None]<(x2-x1)*(py[None]-y1)/(y2-y1+1e-300)+x1)
        inside=(crossing.sum(0)%2)==1;sg=torch.where(inside,-torch.ones_like(px),torch.ones_like(px))
        en=torch.sqrt(self._l2[j]);on=d2<1e-26
        nx=torch.where(on,self._ey[j]/en,sg*dx/norm);ny=torch.where(on,-self._ex[j]/en,sg*dy/norm)
        return sg*torch.sqrt(d2),nx,ny
    def dist_normal(self,px,py):
        px=torch.as_tensor(px);py=torch.as_tensor(py)
        ans=[self._dist(px[j:j+512],py[j:j+512]) for j in range(0,px.numel(),512)]
        return tuple(torch.cat([a[k] for a in ans]) for k in range(3))

class SmoothTarget(Target):
    """Analytic CCW curve, including its first and second derivatives."""
    def __init__(self,curve,name,spec=None):
        self.curve=curve;self.name=name;self.spec=spec or dict(kind='smooth',name=name)
        th=torch.arange(4096)*2*math.pi/4096
        with torch.no_grad():x,y,dx,dy,_,_=curve(th)
        self.ps=x.numpy();self.qs=y.numpy();self._theta=np.linspace(0,2*math.pi,4097)
        self._arc=np.r_[0,np.cumsum(np.hypot(np.roll(self.ps,-1)-self.ps,np.roll(self.qs,-1)-self.qs))]
        dt=2*math.pi/len(th);A=.5*float((x*dy-y*dx).sum())*dt
        mean=np.array([float((x*x*dy).sum()),-float((y*y*dx).sum())])*dt/(2*A)
        xx=float((x**3*dy).sum())*dt/(3*A);yy=-float((y**3*dx).sum())*dt/(3*A)
        xy=float((x*x*y*dy).sum())*dt/(2*A)
        self._moments=A,mean,np.array([[xx,xy],[xy,yy]])-np.outer(mean,mean)
        self._coarse_theta=torch.arange(96)*2*math.pi/96
        self._coarse=curve(self._coarse_theta)[:2]
    def moments(self):return self._moments
    @property
    def per(self):return float(self._arc[-1])
    def at(self,s):
        th=np.interp(np.mod(np.asarray(s,float),self.per),self._arc,self._theta)
        with torch.no_grad():x,y,*_=self.curve(torch.as_tensor(th))
        return x.numpy(),y.numpy()
    def boundary(self,n,include_vertices=False):return self.at((np.arange(n)+.5)*self.per/n)
    def _dist(self,px,py):
        with torch.no_grad():
            bx,by=self._coarse
            j=torch.topk((px[None]-bx[:,None])**2+(py[None]-by[:,None])**2,3,dim=0,largest=False).indices
            th=self._coarse_theta[j];P=px[None];Q=py[None]
            for _ in range(12):
                x,y,dx,dy,ddx,ddy=self.curve(th)
                g=(x-P)*dx+(y-Q)*dy;H=dx*dx+dy*dy+(x-P)*ddx+(y-Q)*ddy
                safe=dx*dx+dy*dy+((x-P)*ddx).abs()+((y-Q)*ddy).abs()
                den=torch.where(H>1e-14,H,safe.clamp(min=1e-14))
                step=(g/den).clamp(-.3,.3);old=(x-P)**2+(y-Q)**2;tn=th-step
                for _ls in range(5):
                    xn,yn,*_=self.curve(tn);bad=((xn-P)**2+(yn-Q)**2)>old+1e-28
                    if not bool(bad.any()):break
                    step=torch.where(bad,step*.5,step);tn=th-step
                th=tn
            x,y,dx,dy,_,_=self.curve(th);k=((x-P)**2+(y-Q)**2).argmin(0);ar=torch.arange(px.numel())
            x,y,dx,dy=x[k,ar],y[k,ar],dx[k,ar],dy[k,ar]
            length=torch.sqrt(dx*dx+dy*dy);nx=dy/length;ny=-dx/length
        # Envelope derivative, also valid on the smooth boundary itself.
        return (px-x)*nx+(py-y)*ny,nx,ny

def target_square(area,center=(0,0),name='square'):
    a=math.sqrt(area)/2;cx,cy=center
    return Target([cx-a,cx+a,cx+a,cx-a],[cy-a,cy-a,cy+a,cy+a],name)
def target_triangle(area,center=(0,0),name='triangle'):
    L=math.sqrt(2*area);cx,cy=center
    return Target([cx-L/3,cx+2*L/3,cx-L/3],[cy-L/3,cy-L/3,cy+2*L/3],name)
def target_ellipse(area,ratio=1.,center=(0,0),n=None,name='ellipse'):
    b=math.sqrt(area/(math.pi*ratio));a=ratio*b;cx,cy=center
    def curve(th):
        c,s=torch.cos(th),torch.sin(th)
        return cx+a*c,cy+b*s,-a*s,b*c,-a*c,-b*s
    return SmoothTarget(curve,name,dict(kind='ellipse',area=area,ratio=ratio,center=list(center)))
def target_from_chart(chart,theta,region,n=2048,name='family'):
    xb,yb=region.boundary(n);fld=chart.field(torch.as_tensor(xb),torch.as_tensor(yb),theta)
    if fld['n_out']:raise ValueError('Cannot manufacture a target from a failed inverse')
    return Target(fld['ux'].numpy(),fld['uy'].numpy(),name)
