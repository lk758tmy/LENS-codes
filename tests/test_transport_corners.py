"""Independent derivatives, block normal equations, and superellipse geometry."""
import sys, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens
import numpy as np
import torch
from lens import transport_cases as c, transport_fit as tf

def main():
    r=c.make_region('U');s=c.make_source(0)
    mass=c.source_moments(r,s,32)[0]
    exact=4*r.a*r.b*math.gamma(1.25)**2/math.gamma(1.5)
    assert abs(mass-exact)<1e-12
    assert abs(mass-c.source_moments(r,s,48)[0])<1e-12
    tr=c.SuperellipseManufactured();T=tr.target();assert abs(T.area-mass)<1e-12
    for curve in (r.curve,T.curve):
        th=torch.linspace(0,6.1,37).requires_grad_();v=curve(th)
        for j in (0,1):
            d=torch.autograd.grad(v[j].sum(),th,create_graph=True,retain_graph=True)[0]
            dd=torch.autograd.grad(d.sum(),th,retain_graph=True)[0]
            assert float((d-v[j+2]).abs().max())<1e-12
            assert float((dd-v[j+4]).abs().max())<1e-11
    obj=tf.TransportFit(c.make_region('S'),c.make_target('P',1),K=4,n_edge=31,
                        corner_weight=1.7,source_vertices=True).build()
    assert 31<=obj.m<=35
    assert np.max(np.min(np.linalg.norm(obj.region.V[:,None]-np.column_stack([obj.xb,obj.yb])[None],axis=2),axis=1))<1e-13
    p=obj._problem(0);th=obj.theta0.clone();ta=obj.t0.clone()
    th[2]+=.012;ta+=torch.linspace(-.002,.003,len(ta))
    # Repeated active indices exercise the index_add accumulation.
    st=p.eval(th,ta,False)['state'];im=torch.stack([st['wx'],ta],1)
    p.corners=torch.stack([im[3]+torch.tensor([.001,.001]),im[3]+torch.tensor([.002,.001]),im[7]+.001])
    ev=p.eval(th,ta);assert ev['ic'][0]==ev['ic'][1]
    eps=1e-6
    for k,j in enumerate(torch.where(p.free)[0]):
        a=th.clone();b=th.clone();a[j]+=eps;b[j]-=eps
        fd=(p.eval(a,ta,False)['rc']-p.eval(b,ta,False)['rc'])/(2*eps)
        assert float((fd-ev['Jc'][:,k]).abs().max())<1e-8
    for j in range(len(ta)):
        a=ta.clone();b=ta.clone();a[j]+=eps;b[j]-=eps
        fd=(p.eval(th,a,False)['rc']-p.eval(th,b,False)['rc'])/(2*eps)
        col=torch.zeros_like(fd);col.index_add_(0,torch.arange(len(fd)),ev['jc']*(ev['ic']==j))
        assert float((fd-col).abs().max())<1e-8
    m=p.m;n=p.n_par;J=ev['J'];Jc=ev['Jc'];rc=ev['rc'];ic=ev['ic'];jc=ev['jc']
    Jt=torch.cat([torch.diag(ev['j1']),torch.diag(ev['j2'])]);Jtc=torch.zeros(len(rc),m)
    Jtc[torch.arange(len(rc)),ic]=jc
    full=torch.cat([torch.cat([J,Jt],1),torch.cat([Jc,Jtc],1)])
    normal=full.T@full
    A=J.T@J+Jc.T@Jc
    M=(J[:m].T*ev['j1'][None]+J[m:].T*ev['j2'][None]).index_add(1,ic,Jc.T*jc[None])
    D=(ev['j1']**2+ev['j2']**2).index_add(0,ic,jc*jc)
    blocked=torch.cat([torch.cat([A,M],1),torch.cat([M.T,torch.diag(D)],1)])
    assert torch.allclose(blocked,normal,atol=1e-12,rtol=1e-12)
    kappa=.03;rhs=-full.T@torch.cat([ev['r'],rc]);normal+=kappa*torch.eye(n+m)
    direct=torch.linalg.solve(normal,rhs)
    sch=A+kappa*torch.eye(n)-(M/(D+kappa)[None])@M.T
    u=torch.linalg.solve(sch,rhs[:n]-M@(rhs[n:]/(D+kappa)))
    v=(rhs[n:]-M.T@u)/(D+kappa)
    assert torch.allclose(torch.cat([u,v]),direct,atol=1e-10,rtol=1e-9)
    print('PASS: exact superellipse mass/analytic derivatives; corner FD; duplicate-index normal blocks; full vs Schur step')

if __name__=='__main__':main()
