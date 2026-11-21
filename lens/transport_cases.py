"""Independent source quadrature, analytic reference maps and balanced design."""
from __future__ import annotations
import math
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from . import regions,sources,targets

class EllipseRegion(regions.ParamRegion):
    def __init__(self,a=.5,b=.35):
        self.a,self.b=a,b;self.name='ellipse';self.bbox=(.5-a,.5+a,.5-b,.5+b)
    def _xy(self,th):return .5+self.a*np.cos(th),.5+self.b*np.sin(th)
    def contains(self,x,y):return ((np.asarray(x)-.5)/self.a)**2+((np.asarray(y)-.5)/self.b)**2<1+1e-14
class DiskRegion(regions.Disk):
    def contains(self,x,y):return (np.asarray(x)-self.cx)**2+(np.asarray(y)-self.cy)**2<self.r**2+1e-14

class SuperellipseRegion(regions.ParamRegion):
    """Legacy exp7 superellipse, smooth polar parameter (no axis singularities)."""
    def __init__(self,a=.5,b=.4,center=(0.,0.)):
        self.a,self.b=a,b;self.cx,self.cy=center;self.name='superellipse4'
        self.bbox=(self.cx-a,self.cx+a,self.cy-b,self.cy+b)
    def _xy(self,th):
        c,s=np.cos(th),np.sin(th);r=(c**4/self.a**4+s**4/self.b**4)**(-.25)
        return self.cx+r*c,self.cy+r*s
    def contains(self,x,y):return ((np.asarray(x)-self.cx)/self.a)**4+((np.asarray(y)-self.cy)/self.b)**4<=1+1e-14
    def curve(self,th):
        c,s=torch.cos(th),torch.sin(th);a,b=self.a**4,self.b**4
        H=c**4/a+s**4/b;Hp=4*(-c**3*s/a+s**3*c/b)
        Hpp=4*((3*c*c*s*s-c**4)/a+(3*s*s*c*c-s**4)/b)
        r=H**(-.25);rp=-r*Hp/(4*H);rpp=r*(5/16*(Hp/H)**2-Hpp/(4*H))
        return self.cx+r*c,self.cy+r*s,rp*c-r*s,rp*s+r*c,(rpp-r)*c-2*rp*s,(rpp-r)*s+2*rp*c
def make_region(code, *, superellipse_center=(0.,0.)):
    if code=='S':r=regions.Square(0,1,0,1)
    elif code=='D':r=DiskRegion(.5,.5,.5)
    elif code=='P':r=regions.PolygonRegion([(0,0),(1,0),(0,1)],'triangle')
    elif code=='E':r=EllipseRegion()
    elif code=='U':r=SuperellipseRegion(center=superellipse_center)
    else:raise ValueError(code)
    r.transport_code=code;return r
def make_source(index):return sources.Source('const',c=1.) if index==0 else sources.Source('cos',a=.3)
def quadrature(region,n=32):
    z,w=leggauss(n);s=(z+1)/2;ws=w/2;code=region.transport_code
    if code in ('S','P'):
        x,t=np.meshgrid(s,s,indexing='ij');W=ws[:,None]*ws[None,:]
        if code=='P':y=(1-x)*t;W=W*(1-x)
        else:y=t
    elif code=='U':
        nr=max(128,4*n);th=np.arange(nr)*2*np.pi/nr
        bx,by=region._xy(th);dx,dy=bx-region.cx,by-region.cy;rad2=dx*dx+dy*dy
        x=region.cx+s[:,None]*dx[None,:];y=region.cy+s[:,None]*dy[None,:]
        W=ws[:,None]*s[:,None]*rad2[None,:]*2*np.pi/nr
    else:
        nr=max(128,4*n);th=np.arange(nr)*2*np.pi/nr
        a,b=(.5,.5) if code=='D' else (region.a,region.b)
        x=.5+a*s[:,None]*np.cos(th)[None,:];y=.5+b*s[:,None]*np.sin(th)[None,:]
        W=np.broadcast_to(ws[:,None]*s[:,None]*a*b*2*np.pi/nr,x.shape)
    return x.ravel(),y.ravel(),W.ravel()
def source_moments(region,source,n=32):
    x,y,w=quadrature(region,n);wf=w*source(x);M=float(wf.sum())
    mean=np.array([np.sum(wf*x),np.sum(wf*y)])/M;z=np.column_stack([x,y])-mean
    return M,mean,(z.T*wf)@z/M
def boundary(region,n,vertices=False):
    if vertices and isinstance(region,regions.PolygonRegion):
        pts=[]
        for i,L in enumerate(region.seg):
            count=max(2,int(np.ceil(n*L/region.per)));a=np.arange(count)/count
            pts.append(region.V[i]+a[:,None]*region._d[i])
        q=np.vstack(pts);return q[:,0],q[:,1]
    return region.boundary(n)
def make_target(code,mass):
    if code=='S':return targets.target_square(mass)
    if code=='D':return targets.target_ellipse(mass,ratio=1.,name='disk')
    if code=='P':return targets.target_triangle(mass)
    if code=='E':return targets.target_ellipse(mass,ratio=2.)
    if code=='U':
        area_unit=4*math.gamma(1.25)**2/math.gamma(1.5)
        b=math.sqrt(mass/(2*area_unit))
        region=SuperellipseRegion(a=2*b,b=b)
        return targets.SmoothTarget(region.curve,'superellipse4',dict(
            kind='superellipse4',exponent=4,a=2*b,b=b,center=[0.,0.]))
    raise ValueError(code)
class Manufactured:
    def __init__(self,source_index,eta=.04):self.index=source_index;self.eta=eta
    def F(self,x):
        if self.index==0:return .5*x*x,x,torch.ones_like(x),torch.zeros_like(x)
        F=.5*x*x+.3*(torch.cos(2*math.pi*x)-1)/(4*math.pi**2)
        F1=x-.3*torch.sin(2*math.pi*x)/(2*math.pi)
        return F,F1,1-.3*torch.cos(2*math.pi*x),.6*math.pi*torch.sin(2*math.pi*x)
    def field_torch(self,x,y):
        xi=x-.5;v=y-.5;F,F1,_,_=self.F(xi);e=self.eta
        A=v-6*e*F;s=torch.sqrt(1-12*e*A);t=2*A/(1+s)
        return F-.5*t*t+e*(t**3-6*F*t)+v*t,F1*s,t
    def __call__(self,x,y):
        with torch.no_grad():r=self.field_torch(torch.as_tensor(np.asarray(x,float)),torch.as_tensor(np.asarray(y,float)))
        return tuple(t.numpy() for t in r)
    def curve(self,th):
        r=.5;e=self.eta;c,s=torch.cos(th),torch.sin(th)
        xi=r*c;v=r*s;dx=-r*s;ddx=-r*c;dv=r*c;ddv=-r*s
        F,F1,f,fp=self.F(xi);A=v-6*e*F;root=torch.sqrt(1-12*e*A)
        q=2*A/(1+root);qp=(dv-6*e*F1*dx)/root
        qpp=(ddv-6*e*(f*dx*dx+F1*ddx)+6*e*qp*qp)/root
        p=F1*root;pp=f*dx*root-6*e*F1*qp
        ppp=(fp*dx*dx+f*ddx)*root-12*e*f*dx*qp-6*e*F1*qpp
        return p,q,pp,qp,ppp,qpp
    def target(self):return targets.SmoothTarget(self.curve,f'manufactured_f{self.index}',dict(kind='manufactured',source=self.index,eta=self.eta))

class SuperellipseManufactured(Manufactured):
    """Independent nonlinear, in-family check on the actual legacy superellipse."""
    def __init__(self,eta=.04):super().__init__(0,eta);self.region=SuperellipseRegion()
    def field_torch(self,x,y):return super().field_torch(x+.5,y+.5)
    def curve(self,th):
        x,v,dx,dv,ddx,ddv=self.region.curve(th);e=self.eta
        F,F1,f,fp=self.F(x);A=v-6*e*F;root=torch.sqrt(1-12*e*A)
        q=2*A/(1+root);qp=(dv-6*e*F1*dx)/root
        qpp=(ddv-6*e*(f*dx*dx+F1*ddx)+6*e*qp*qp)/root
        p=F1*root;pp=f*dx*root-6*e*F1*qp
        ppp=(fp*dx*dx+f*ddx)*root-12*e*f*dx*qp-6*e*F1*qpp
        return p,q,pp,qp,ppp,qpp
    def target(self):return targets.SmoothTarget(self.curve,'superellipse_manufactured',dict(kind='superellipse_manufactured',eta=self.eta,a=.5,b=.4))
def balanced_design(levels='SDPE'):
    """OA(16;4,4,2;2), with interactions not fully identifiable."""
    out=[]
    for i,a in enumerate(levels):
        for j,b in enumerate(levels):
            f=(i+j)%2
            out.append(dict(id=f'B_{a}_{b}_f{f}',group='B',region=a,target=b,source=f,K=16,m=256))
    return out
