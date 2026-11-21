"""Post-fit checks; validation/truth values never enter fitting or rho selection."""
from pathlib import Path
import json
import numpy as np
import torch
from numpy.polynomial import chebyshev as C
from scipy.interpolate import RegularGridInterpolator
from .inverse_data import InverseSource,observation_angles


def discrepancy(problem,z):
    _,sensor,_=observation_angles()
    result=problem.field_metrics(z,sensor)
    q=[];g=[]
    for i,v in enumerate(result):
        observed=problem.q[i,-problem.nsensor:].numpy()
        q.append(float(np.sqrt(np.mean((v['q']-observed)**2))/problem.sq[i]))
        g.append(v['g_rms']/problem.sg[i])
    return dict(q_relative_rms=float(np.sqrt(np.mean(np.array(q)**2))),
                q_each=q,g_scaled_rms=float(np.sqrt(np.mean(np.array(g)**2))),
                g_each=g,n_out=sum(v['n_out'] for v in result))


def audit(problem,z,reference,name,n=65):
    eta,fields=problem.unpack(z);st=problem.basis.build(eta,False)
    axis=np.linspace(0,1,n);X,Y=np.meshgrid(axis,axis,indexing='ij')
    inside=(X-.5)**2+(Y-.5)**2<.25-1e-9
    X,Y=X[inside],Y[inside]
    results=[]
    for i,(theta,gamma) in enumerate(zip(fields,problem.gamma)):
        a,_=problem.basis.omega_coefficients(st,theta,gamma,False)
        x=torch.tensor(X,requires_grad=True);y=torch.tensor(Y,requires_grad=True)
        with torch.no_grad():t,nb=problem.basis.solve_t(a,x,y,problem.tlimits[i])
        cache=problem.basis.cheb(x)
        for _ in range(3):
            v=problem.basis.evaluate(a,x,t,cache);t=t-(v['wt']+y)/v['wtt']
        v=problem.basis.evaluate(a,x,t,cache);u=v['w']+y*t
        ux,uy=torch.autograd.grad(u.sum(),(x,y),create_graph=True)
        uxx,uxy=torch.autograd.grad(ux.sum(),(x,y),create_graph=True)
        uyy=torch.autograd.grad(uy.sum(),y)[0]
        det=uxx*uyy-uxy**2;f=problem.basis.source_values(eta,x)
        ref=np.load(Path(reference)/f'{name}_g{problem.loads[i]}_n640.npz')
        interp=RegularGridInterpolator((ref['x'],ref['y']),ref['u'])
        target=interp(np.array([X,Y]).T);pred=u.detach().numpy()
        # Interior grid is strictly inside the disk; all interpolation cells use
        # available inside/boundary-filled values at n640.
        err=np.isfinite(target)
        results.append(dict(load=problem.loads[i],n_grid=int(len(X)),n_out=nb,
            source_pde_max=float((det-f).abs().max().detach()),
            linear_pde_max=float((v['wxx']+f*v['wtt']).abs().max().detach()),
            field_E2=float(np.linalg.norm(pred[err]-target[err])/np.linalg.norm(target[err])),
            field_Einf=float(np.max(abs(pred[err]-target[err]))/np.max(abs(target[err])))))
    return results


def information(problem,z,tau,rcond=1e-10):
    """Nuisance projection in data space and physical L2 source coordinates.

    Column normalization makes the fixed nuisance cutoff invariant to simple
    coefficient/latent units. No roughness or temporary field ridge is included.
    """
    v=problem.evaluate(z,tau);J=v['J'].numpy();jt=v['jt'].numpy()
    B=np.concatenate([np.diag(row) for row in jt],axis=0)
    Z=np.column_stack([J[:,problem.ns:],B])
    norms=np.linalg.norm(Z,axis=0);nz=norms>1e-30
    U,s,_=np.linalg.svd(Z[:,nz]/norms[nz],full_matrices=False)
    keep=s>rcond*s[0];U=U[:,keep]
    nodes,w=np.polynomial.legendre.leggauss(problem.M+2)
    V=C.chebvander(nodes,problem.M);G=V.T@(w[:,None]*V)/2
    L=np.linalg.cholesky(G);whitener=np.linalg.solve(L.T,np.eye(problem.ns))
    A=J[:,:problem.ns]@whitener;eff=A-U@(U.T@A)
    sv=np.linalg.svd(eff,compute_uv=False)
    return dict(nuisance_rcond=rcond,nuisance_rank=int(keep.sum()),
                source_singular_values=sv.tolist(),source_condition=float(sv[0]/sv[-1]),
                source_l2_whitener=whitener.tolist())


def reference_summary(directory):
    out=[]
    for name in ('C','A'):
        for ell in range(3):
            qs={n:np.load(Path(directory)/f'{name}_g{ell}_n{n}.npz')['q'] for n in (80,160,320,640)}
            differences=[float(np.linalg.norm(qs[2*n]-qs[n])/np.linalg.norm(qs[2*n])) for n in (80,160,320)]
            out.append(dict(source=name,load=ell,relative_flux_differences=differences,
                            observed_orders=np.log2(np.array(differences[:-1])/differences[1:]).tolist()))
    return out
