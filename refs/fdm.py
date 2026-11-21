"""Nine-point finite differences, cut-boundary unequal-step differences.

Every directional second difference uses BOTH boundary distances (Shortley-
Weller formula), so it is exact for quadratics at cut cells as well. The mixed
derivative is (D_(1,1)-D_(1,-1))/4. All Newton trials are checked for a positive
discrete Hessian, including cut cells. Initialization solves Delta u=2 sqrt(f)
with the supplied Dirichlet data, without querying the interior reference.
This central scheme is not claimed to be a monotone MA discretization.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OFFS=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]

class CurvedFD:
    def __init__(self,x0,x1,y0,y1,nx,ny,inside,g,f,eps_b=1e-7):
        self.xs=np.linspace(x0,x1,nx+1);self.ys=np.linspace(y0,y1,ny+1)
        self.h=(x1-x0)/nx
        assert abs(self.h-(y1-y0)/ny)<1e-12
        self.nx,self.ny=nx,ny;self.inside=inside;self.g_fn=g
        self.X,self.Y=np.meshgrid(self.xs,self.ys,indexing='ij')
        self.ins=np.asarray(inside(self.X,self.Y),bool)
        self.theta={d:self._crossing(d) for d in OFFS}
        tmin=np.minimum.reduce(list(self.theta.values()))
        self.bnd=self.ins & (tmin<=eps_b)
        self.intm=self.ins & ~self.bnd
        self.ii,self.jj=np.where(self.intm);self.n=len(self.ii)
        self.idx=np.full(self.X.shape,-1,int);self.idx[self.intm]=np.arange(self.n)
        # Read g only at actual boundary nodes/intersections, never at unknowns.
        self.gv=np.zeros_like(self.X)
        self.gv[self.bnd]=g(self.X[self.bnd],self.Y[self.bnd])
        self.fv=np.asarray(f(self.X),float)
        self.U=self.gv.copy()
        self.c_xx,self.b_xx=self._direction((1,0))
        self.c_yy,self.b_yy=self._direction((0,1))
        dp,bp=self._direction((1,1));dm,bm=self._direction((1,-1))
        self.c_xy=(dp-dm)*0.25;self.b_xy=(bp-bm)*0.25

    def _crossing(self,d):
        di,dj=d;X,Y=self.X,self.Y
        ex=X+di*self.h;ey=Y+dj*self.h
        nb=np.asarray(self.inside(ex,ey),bool)
        need=self.ins & ~nb
        th=np.ones_like(X)
        if np.any(need):
            x,y=X[need],Y[need];lo=np.zeros_like(x);hi=np.ones_like(x)
            for _ in range(48):
                mid=(lo+hi)*0.5
                ok=self.inside(x+mid*di*self.h,y+mid*dj*self.h)
                lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
            th[need]=(lo+hi)*0.5
        return th

    def _direction(self,d):
        i,j=self.ii,self.jj;di,dj=d;n=self.n;h=self.h
        ap=self.theta[d][i,j]*h;am=self.theta[(-di,-dj)][i,j]*h
        cp=2/(ap*(ap+am));cm=2/(am*(ap+am))
        rr=[np.arange(n)];cc=[np.arange(n)];vv=[-cp-cm];b=np.zeros(n)
        for sign,w in ((1,cp),(-1,cm)):
            ds=(sign*di,sign*dj);th=self.theta[ds][i,j]
            ni=np.clip(i+ds[0],0,self.nx);nj=np.clip(j+ds[1],0,self.ny)
            unk=(th>1-1e-10)&self.intm[ni,nj]
            r=np.where(unk)[0];rr.append(r);cc.append(self.idx[ni[r],nj[r]]);vv.append(w[r])
            r=np.where(~unk)[0]
            bx=self.X[i[r],j[r]]+th[r]*ds[0]*h
            by=self.Y[i[r],j[r]]+th[r]*ds[1]*h
            b[r]+=w[r]*self.g_fn(bx,by)
        return sp.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n,n)),b

    def _split(self,U=None):
        U=self.U if U is None else U;u=U[self.intm]
        return self.c_xx@u+self.b_xx,self.c_yy@u+self.b_yy,self.c_xy@u+self.b_xy

    def min_eig(self,U=None):
        a,b,c=self._split(U)
        return float(np.min(0.5*(a+b-np.sqrt((a-b)**2+4*c*c))))

    def is_convex(self,U=None):
        return self.min_eig(U)>0

    def residual(self,U):
        a,b,c=self._split(U)
        return a*b-c*c-self.fv[self.intm]

    def jacobian(self,U):
        a,b,c=self._split(U)
        return (sp.diags(b)@self.c_xx+sp.diags(a)@self.c_yy-2*sp.diags(c)@self.c_xy).tocsc()

    def initial(self,u0=None):
        if u0 is None:
            rhs=2*np.sqrt(self.fv[self.intm])-self.b_xx-self.b_yy
            self.U[self.intm]=spla.spsolve(self.c_xx+self.c_yy,rhs)
        else:
            self.U[self.intm]=u0(self.X[self.intm],self.Y[self.intm])
        self.initial_min_eig=self.min_eig()

    def solve(self,maxit=60,tol=2e-8,verbose=False,keep_convex=True):
        U=self.U.copy();self.iterations=0;self.status='maxit'
        if keep_convex and not self.is_convex(U):
            raise RuntimeError('Poisson initialization is not discretely convex')
        for it in range(maxit):
            R=self.residual(U);rn=float(np.max(np.abs(R)))
            if verbose: print(it,rn,self.min_eig(U),flush=True)
            if rn<tol:
                self.status='converged';break
            d=spla.spsolve(self.jacobian(U),-R)
            accepted=False
            for k in range(40):
                Un=U.copy();Un[self.intm]+=2.**(-k)*d
                if (not keep_convex or self.is_convex(Un)) and np.max(np.abs(self.residual(Un)))<rn:
                    U=Un;accepted=True;self.iterations+=1;break
            if not accepted:
                self.status='stalled';break
        self.U=U;self.residual_inf=float(np.max(np.abs(self.residual(U))))
        if self.residual_inf<tol:self.status='converged'
        return U

def solve_fdm(region,truth,source,n=80,maxit=60,verbose=False,keep_convex=True,u0=None):
    """Only truth's boundary restriction supplies g; errors are measured later."""
    x0,x1,y0,y1=region.bbox;xc=(x0+x1)/2;yc=(y0+y1)/2
    def inside(X,Y):
        if region.name=='disk':
            return (np.asarray(X)-region.cx)**2+(np.asarray(Y)-region.cy)**2<=region.r**2*(1+1e-13)
        return np.asarray(region.contains(xc+(np.asarray(X)-xc)*(1-1e-13),yc+(np.asarray(Y)-yc)*(1-1e-13))).reshape(np.shape(X))
    t0=time.perf_counter()
    p=CurvedFD(x0,x1,y0,y1,n,n,inside,lambda X,Y:truth(X,Y)[0],source)
    p.initial(u0);p.build_seconds=time.perf_counter()-t0
    t0=time.perf_counter();p.solve(maxit=maxit,verbose=verbose,keep_convex=keep_convex)
    p.fit_seconds=time.perf_counter()-t0
    return p

def fdm_field(p,X,Y):
    """Piecewise linear interpolation of unknowns and actual boundary samples.

    Reported formal convergence norms use the unknown grid nodes, avoiding an
    arbitrary outside-domain extension in cells intersecting a curved boundary.
    """
    from scipy.interpolate import LinearNDInterpolator
    bx=[];by=[];bv=[]
    for d in OFFS:
        th=p.theta[d][p.ii,p.jj];cut=th<1-1e-10
        x=p.X[p.ii[cut],p.jj[cut]]+th[cut]*d[0]*p.h
        y=p.Y[p.ii[cut],p.jj[cut]]+th[cut]*d[1]*p.h
        bx.extend(x);by.extend(y);bv.extend(p.g_fn(x,y))
    xy=np.column_stack([np.r_[p.X[p.ins],bx],np.r_[p.Y[p.ins],by]])
    itp=LinearNDInterpolator(xy,np.r_[p.U[p.ins],bv])
    return np.asarray(itp(np.asarray(X),np.asarray(Y)))
