"""Fixed-layout Chebyshev generator and exact source-coordinate derivatives.

All polynomial products/integrals have fixed shapes; no candidate-dependent
trimming or rounded cache keys. Mode normalization is fixed from f=1, hence
has zero derivative with respect to the fitted source coefficients.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from numpy.polynomial import chebyshev as C

class SourceGenerator:
    def __init__(self,M,K,tmax,scale_grid=33):
        self.M,self.K=int(M),int(K);self.ns=M+1;self.na=K+1;self.nm=2*self.na
        self.L=max(M+3,(M+2)*(K//2)+2)
        self.tmax=float(tmax);self.xs=.5;self.xo=.5
        size=self.L;integ=[];der=[]
        for j in range(size):
            unit=np.zeros(size);unit[j]=1
            a=C.chebint(unit);a[0]-=C.chebval(-1.,a)
            b=C.chebint(a);b[0]-=C.chebval(-1.,b)
            integ.append(np.pad(b,(0,max(0,size-len(b))))[:size]*self.xs**2)
            d=C.chebder(unit)/self.xs
            der.append(np.pad(d,(0,max(0,size-len(d))))[:size])
        self.I2=torch.tensor(np.array(integ).T);self.D=torch.tensor(np.array(der).T)
        i=torch.arange(self.ns)[:,None];j=torch.arange(size)[None,:]
        self.plus=(i+j).clamp(max=size-1);self.mask=(i+j<size).double();self.minus=(i-j).abs()
        seeds=torch.zeros(self.K+1,self.nm,size)
        for J in range(self.K+1):
            seeds[J,J,0]=1/math.factorial(J)
            seeds[J,self.na+J,1]=1.
        self.seeds=seeds
        self.dF=self.I2[:,:self.ns].T.clone()
        self.g=torch.ones(self.nm)
        eta=torch.zeros(self.ns);eta[0]=1
        st=self.build(eta,False)
        x,t=torch.meshgrid(torch.linspace(0,1,scale_grid),torch.linspace(-tmax,tmax,scale_grid),indexing='ij')
        pp=self.parts(st,x.ravel(),t.ravel())
        self.g=torch.stack([pp[k].abs().amax(1) for k in ('w','wt','wtt')]).amax(0).clamp(min=1e-14)

    def multiply(self,eta,a):
        vals=.5*a.unsqueeze(-2)*eta.reshape((1,)*(a.ndim-1)+(self.ns,1))
        shape=a.shape[:-1]+(self.ns*self.L,)
        vals=vals.reshape(shape)
        ip=self.plus.reshape(-1).expand(shape);im=self.minus.reshape(-1).expand(shape)
        out=torch.zeros_like(a)
        return out.scatter_add(-1,ip,vals*self.mask.reshape(-1))+out.scatter_add(-1,im,vals)

    def multiply_source_units(self,a):
        vals=.5*a[None].expand(self.ns,*a.shape)
        ip=self.plus[:,None,:].expand_as(vals);im=self.minus[:,None,:].expand_as(vals)
        out=torch.zeros_like(vals)
        return out.scatter_add(-1,ip,vals*self.mask[:,None,:])+out.scatter_add(-1,im,vals)

    def build(self,eta,derivatives=True):
        aa=[None]*(self.K+1);dd=[None]*(self.K+1)
        for j in range(self.K,-1,-1):
            aa[j]=self.seeds[j]
            if derivatives:dd[j]=torch.zeros(self.ns,self.nm,self.L)
            if j+2<=self.K:
                c=(j+1)*(j+2)
                aa[j]=aa[j]-c*(self.multiply(eta,aa[j+2])@self.I2.T)
                if derivatives:
                    product=self.multiply(eta,dd[j+2])+self.multiply_source_units(aa[j+2])
                    dd[j]=dd[j]-c*(product@self.I2.T)
        result=dict(A=torch.stack(aa),F=eta@self.dF,eta=eta)
        if derivatives:result['dA']=torch.stack(dd,dim=1)
        return result

    def cheb(self,x):
        z=(x-self.xo)/self.xs
        rows=[torch.ones_like(z),z]
        for j in range(2,self.L):rows.append(2*z*rows[-1]-rows[-2])
        T=torch.stack(rows)
        return T,self.D.T@T,(self.D@self.D).T@T

    def powers(self,t):
        p=[torch.ones_like(t)];d=[torch.zeros_like(t)];dd=[torch.zeros_like(t)]
        for j in range(1,self.K+1):
            p.append(p[-1]*t);d.append(j*p[j-1]);dd.append(j*(j-1)*p[j-2] if j>1 else torch.zeros_like(t))
        return torch.stack(p),torch.stack(d),torch.stack(dd)

    def parts(self,st,x,t,Tcache=None):
        T,Tx,Txx=self.cheb(x) if Tcache is None else Tcache
        p,pt,ptt=self.powers(t)
        v=torch.einsum('jml,ln->jmn',st['A'],T)/self.g[None,:,None]
        vx=torch.einsum('jml,ln->jmn',st['A'],Tx)/self.g[None,:,None]
        return dict(w=torch.einsum('jmn,jn->mn',v,p),wt=torch.einsum('jmn,jn->mn',v,pt),
                    wtt=torch.einsum('jmn,jn->mn',v,ptt),wx=torch.einsum('jmn,jn->mn',vx,p),
                    wxt=torch.einsum('jmn,jn->mn',vx,pt))

    def omega_coefficients(self,st,theta,gamma,derivatives=True):
        a=torch.einsum('jml,m->jl',st['A'],theta/self.g)
        harmonic=a
        a=a+torch.nn.functional.pad((gamma*st['F'])[None],(0,0,0,self.K))
        base=torch.zeros_like(a);base[2,0]=-.5*gamma;a=a+base
        if not derivatives:return a,None
        if 'dA' in st:
            da=torch.einsum('sjml,m->sjl',st['dA'],theta/self.g)
        else:
            # Source differentiation commutes with the field linear combination.
            # Differentiate only this effective field (ns x L), rather than all
            # nm individual modes (ns x nm x L). Fixed theta/g has zero derivative.
            dd=[torch.zeros(self.ns,self.L) for _ in range(self.K+1)]
            for j in range(self.K-2,-1,-1):
                c=(j+1)*(j+2)
                forcing=self.multiply_source_units(harmonic[j+2][None,:])[:,0,:]
                dd[j]=-c*((self.multiply(st['eta'],dd[j+2])+forcing)@self.I2.T)
            da=torch.stack(dd,dim=1)
        extra=torch.zeros_like(da);extra[:,0,:]=gamma*self.dF
        return a,da+extra

    def evaluate(self,a,x,t,Tcache=None):
        T,Tx,Txx=self.cheb(x) if Tcache is None else Tcache
        p,pt,ptt=self.powers(t)
        # a has optional leading source-coordinate dimension.
        values=torch.einsum('...jl,ln->...jn',a,T)
        dx=torch.einsum('...jl,ln->...jn',a,Tx)
        return dict(w=torch.einsum('...jn,jn->...n',values,p),
                    wt=torch.einsum('...jn,jn->...n',values,pt),
                    wtt=torch.einsum('...jn,jn->...n',values,ptt),
                    wx=torch.einsum('...jn,jn->...n',dx,p),
                    wxt=torch.einsum('...jn,jn->...n',dx,pt),
                    wxx=torch.einsum('...jl,ln,jn->...n',a,Txx,p))

    def source_values(self,eta,x):
        T=self.cheb(x)[0][:self.ns]
        return eta@T

    def solve_t(self,a,x,y,tmax=None):
        bound=self.tmax if tmax is None else float(tmax)
        cache=self.cheb(x);lo=torch.full_like(x,-bound);hi=-lo
        left=self.evaluate(a,x,lo,cache)['wt']+y;right=self.evaluate(a,x,hi,cache)['wt']+y
        covered=(left>=-1e-11)&(right<=1e-11)
        t=torch.zeros_like(x)
        for _ in range(18):
            ev=self.evaluate(a,x,t,cache);e=ev['wt']+y
            if float(e.abs().max())<1e-12:break
            lo=torch.where(e>0,t,lo);hi=torch.where(e<=0,t,hi)
            step=t-e/ev['wtt'];good=torch.isfinite(step)&(step>=lo)&(step<=hi)
            t=torch.where(good,step,(lo+hi)/2)
        e=self.evaluate(a,x,t,cache)['wt']+y
        bad=(e.abs()>1e-9)|~covered
        if bool(bad.any()):
            for _ in range(45):
                mid=(lo+hi)/2;pos=self.evaluate(a,x,mid,cache)['wt']+y>0
                lo=torch.where(pos,mid,lo);hi=torch.where(pos,hi,mid)
            t=torch.where(bad,(lo+hi)/2,t)
        return t,int(((self.evaluate(a,x,t,cache)['wt']+y).abs()>1e-9).logical_or(~covered).sum())
