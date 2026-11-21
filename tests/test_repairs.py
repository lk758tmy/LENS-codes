"""Regression checks for coordinate equivalence, missing mode and actual output."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens
import numpy as np
import torch
from lens import basis_fx,basis_const,cert,chart,res,metrics,sources,regions,truth
from refs.fdm import CurvedFD,solve_fdm

def main():
    for b in (basis_const.ConstBasis(K=4,box=(0,1,-1,1)),
              basis_fx.FxBasis(sources.Source('quad'),(0,1,-1,1),K=4)):
        c=chart.Chart(b,cert.Certificate(b.cert_S(21)))
        th=torch.zeros(b.nmodes+1);th[-1]=.8;th[2]=.02
        x=torch.linspace(.1,.9,11);t=torch.linspace(-.7,.7,11)
        h,G=c.equality_coordinates(th)
        assert abs(float(c.gamma(h)-G))<1e-11
        for k in ('w','wt','wx','wtt'):
            assert float((c.eval(x,t,th)[k]-c.eval(x,t,h)[k]).abs().max())<1e-11
        p=res.DirichletBVP(c,x,t*0,torch.zeros_like(x),gamma_mode='gauge')
        assert not p.free[-1]
        if b.kind=='polyt':
            assert p.free[b.n_alpha]
            P=b.parts(x,t)
            assert torch.allclose(P['W'][b.n_alpha],(x-b._xo)/b._xs/b.g[b.n_alpha])
        # All joint coefficient and latent columns, including B_0.
        ev=p.eval(th,t);eps=1e-6
        for jj,j in enumerate(torch.where(p.free)[0]):
            tp=th.clone();tm=th.clone();tp[j]+=eps;tm[j]-=eps
            fd=(p.eval(tp,t,False)['r']-p.eval(tm,t,False)['r'])/(2*eps)
            assert float((fd-ev['J'][:,jj]).abs().max())<2e-8
        for j in range(len(t)):
            tp=t.clone();tm=t.clone();tp[j]+=eps;tm[j]-=eps
            fd=(p.eval(th,tp,False)['r']-p.eval(th,tm,False)['r'])/(2*eps)
            col=torch.zeros_like(fd);col[j]=ev['j1'][j];col[len(t)+j]=ev['j2'][j]
            assert float((fd-col).abs().max())<2e-8
    # An inverse that leaves the box must not acquire a false roundoff PDE audit.
    b=basis_const.ConstBasis(K=2,box=(0,1,-.1,.1));c=chart.Chart(b,cert.Certificate(b.cert_S()))
    th=torch.zeros(b.nmodes+1);th[-1]=1
    assert np.isnan(metrics.audit_pde(c,th,np.array([.3]),np.array([.8]),lambda x:x*0+1))
    # Quadratic exactness on every cut-cell stencil, not just bulk nodes.
    inside=lambda x,y:(np.asarray(x)-.5)**2+(np.asarray(y)-.5)**2<=.25*(1+1e-13)
    q=lambda x,y:2*np.asarray(x)**2+.3*np.asarray(x)*np.asarray(y)+np.asarray(y)**2+.7
    p=CurvedFD(0,1,0,1,32,32,inside,q,lambda x:x*0+7.91)
    p.initial(q);a,b,c=p._split()
    assert np.max(abs(a-4))<2e-9 and np.max(abs(b-2))<2e-9 and np.max(abs(c-.3))<2e-9
    # A callback that refuses any interior truth query proves the cold solver
    # only uses the Dirichlet restriction, including during initialization.
    tr=truth.RadialTruth()
    def boundary_only(x,y):
        assert np.max(abs((np.asarray(x)-.5)**2+(np.asarray(y)-.5)**2-.25))<1e-8
        return tr(x,y)
    guarded=solve_fdm(regions.get_region('disk'),boundary_only,sources.Source('const'),n=20)
    assert guarded.status=='converged'
    e=[]
    for n in (20,40):
        p=solve_fdm(regions.get_region('disk'),truth.RadialTruth(),sources.Source('const'),n=n)
        assert p.status=='converged' and p.min_eig()>.5
        e.append(np.sqrt(np.mean((p.U[p.intm]-truth.RadialTruth()(p.X[p.intm],p.Y[p.intm])[0])**2)))
    assert e[0]/e[1]>3.5
    print('PASS: equivalence, full joint Jacobians, B0, inverse audit, cut-cell quadratics, cold-start convex convergence')

if __name__=='__main__':main()
