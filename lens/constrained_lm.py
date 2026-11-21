"""Feasible active-set solve for a damped Schur LM coefficient subproblem.

The sampled physical curvature constraints are linear in stable coefficients.
This changes the step computation, not the PDE family or joint square loss.
"""
import torch

def _working_qp(H,rhs,C,b,maxit=160):
    n=len(rhs);eye=torch.eye(n,dtype=H.dtype)
    L,info=torch.linalg.cholesky_ex(H)
    if int(info):
        v,Q=torch.linalg.eigh(H);H=(Q*v.clamp(min=1e-12))@Q.T
        L=torch.linalg.cholesky(H)
    Hi=torch.cholesky_inverse(L)
    scale=torch.linalg.vector_norm(C,dim=1).clamp(min=1e-30)
    C=C/scale[:,None];b=b.clamp(min=0)/scale
    x=torch.zeros_like(rhs);active=[]
    tol=1e-10
    for it in range(maxit):
        v=Hi@(rhs-H@x)
        if active:
            A=C[active];V=Hi@A.T;K=A@V
            multip=torch.linalg.lstsq(K,(A@v)[:,None],driver='gelsd',rcond=1e-12).solution[:,0]
            p=v-V@multip
        else:p=v;multip=torch.empty(0)
        if float(torch.linalg.vector_norm(p))<=tol*(1+float(torch.linalg.vector_norm(x))):
            if active and float(multip.min())<-1e-10:
                active.pop(int(multip.argmin()));continue
            break
        cp=C@p;slack=b-C@x
        eligible=cp>1e-13*max(1.,float(torch.linalg.vector_norm(p)))
        if active:eligible[active]=False
        ratio=torch.where(eligible,slack.clamp(min=0)/cp.clamp(min=1e-300),torch.full_like(cp,float('inf')))
        alpha,j=ratio.min(0);alpha=min(1.,max(0.,float(alpha)))
        x=x+alpha*p
        if alpha<1-1e-12:
            j=int(j)
            if j not in active:active.append(j)
        if len(active)>n:
            # Dependent active samples are handled by the SVD, but the working
            # set need not retain more than n+1 nearly duplicate grid rows.
            active=active[-n:]
    # Preserve feasibility despite roundoff in near-dependent active samples.
    cx=C@x;bad=cx>b+1e-10
    if bool(bad.any()):
        ratios=torch.where(cx>0,b/cx.clamp(min=1e-300),torch.ones_like(cx))
        x=x*float(ratios.min().clamp(0,1))
    return x,dict(qp_iterations=it+1,qp_active=len(active))


def solve_qp(H,rhs,C,b,maxit=160):
    """Constraint generation keeps the active solve small on a dense check grid."""
    L,info=torch.linalg.cholesky_ex(H)
    if int(info):
        v,Q=torch.linalg.eigh(H);H=(Q*v.clamp(min=1e-12))@Q.T;L=torch.linalg.cholesky(H)
    z=torch.cholesky_solve(rhs[:,None],L)[:,0]
    violation=C@z-b
    if float(violation.max())<=1e-11:return z,dict(qp_iterations=0,qp_active=0,qp_cuts=0)
    work=[];inner=0;rec={}
    for outer in range(16):
        bad=violation.clone()
        if work:bad[work]=-float('inf')
        vals,js=torch.topk(bad,min(6,len(b)))
        add=[int(j) for val,j in zip(vals,js) if float(val)>1e-11]
        if not add:break
        work.extend(add)
        z,rec=_working_qp(H,rhs,C[work],b[work],maxit=maxit)
        inner+=rec['qp_iterations'];violation=C@z-b
        if float(violation.max())<=1e-10:break
    rec.update(qp_iterations=inner,qp_cuts=len(work),qp_violation=float(violation.max()))
    return z,rec
