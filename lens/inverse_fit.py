"""Shared-source, multi-input joint Schur-LM; latent t remains an unknown."""
from __future__ import annotations
import math,time
import numpy as np
import torch
from numpy.polynomial import chebyshev as C
from .inverse_basis import SourceGenerator
from .inverse_data import Control,boundary,observation_angles


def input_matrix(ell):
    v=np.sqrt(1+.3**2)
    return np.eye(2) if ell==0 else np.diag([v+.3,v-.3]) if ell==1 else np.array([[v,.3],[.3,v]])

class InverseProblem:
    def __init__(self,q,loads,M,K,lam=.4,rho=0.,mu=0.,fixed_source=None,tmax=None,grid=61,exact_weight=1.):
        self.loads=tuple(loads);self.nload=len(loads);self.M,self.K=M,K
        self.lam,self.rho,self.mu=float(lam),float(rho),float(mu)
        self.exact_weight=float(exact_weight)
        self.curvature_floor=.102
        gd,qs,_=observation_angles();angles=np.r_[gd,qs];xx,yy=boundary(angles)
        self.N=len(angles);self.nsensor=len(qs);self.m=self.N*self.nload
        self.x=torch.tensor(xx);self.y=torch.tensor(yy);self.angles=angles
        self.nx=torch.tensor(np.cos(angles));self.ny=torch.tensor(np.sin(angles))
        self.mask=torch.zeros(self.N);self.mask[len(gd):]=1
        self.q=torch.zeros(self.nload,self.N)
        self.g=torch.tensor(np.array([Control(ell).value(xx,yy) for ell in loads]))
        sg=[];sq=[];limits=[]
        for i,ell in enumerate(loads):
            observed=np.asarray(q[ell],float);self.q[i,len(gd):]=torch.tensor(observed)
            sq.append(max(float(np.sqrt(np.mean(observed**2))),1e-6))
            sg.append(max(float(np.std(self.g[i].numpy())),.5*sq[-1],1e-6))
            # Measured normal response + known tangential derivative estimates u_y.
            # Periodic interpolation is used for chart extent only, not labels.
            order=np.argsort(qs);ap=np.r_[qs[order]-2*np.pi,qs[order],qs[order]+2*np.pi]
            qp=np.r_[observed[order],observed[order],observed[order]]
            qi=np.interp(angles,ap,qp)
            tang=Control(ell).tangent(angles)
            extent=float(np.max(np.abs(np.cos(angles)*tang+np.sin(angles)*qi)))
            initial_extent=.5*float(np.linalg.norm(input_matrix(ell)[1]))
            limits.append(max(1.1*extent+.01,1.05*initial_extent+.01))
        if tmax is not None:limits=([float(tmax)]*self.nload if np.isscalar(tmax) else list(map(float,tmax)))
        self.tlimits=limits;self.tmax=max(limits)
        self.tupper=torch.repeat_interleave(torch.tensor(limits),self.N)
        self.tlower=-self.tupper
        self.basis=SourceGenerator(M,K,self.tmax)
        self.nm=self.basis.nm;self.ns=self.basis.ns
        self.fixed_source=None if fixed_source is None else torch.tensor(np.asarray(fixed_source,float))
        self.neta=self.ns if fixed_source is None else 0
        self.p=self.neta+self.nload*self.nm
        scale=math.sqrt(self.nload)
        self.wg=self.exact_weight*torch.tensor([1/(s*math.sqrt(self.N)*scale) for s in sg])[:,None]
        self.wq=torch.tensor([1/(s*math.sqrt(self.nsensor)*scale) for s in sq])[:,None]*self.mask
        self.we=self.exact_weight*math.sqrt(lam)/(math.sqrt(self.N)*scale)
        self.sq=np.asarray(sq);self.sg=np.asarray(sg)
        self.gamma=[1/float(input_matrix(ell)[1,1]) for ell in loads]
        self.cache=self.basis.cheb(self.x)
        gx=torch.linspace(0,1,grid)
        self.Tgrid=self.basis.cheb(gx)[0]
        self.Pttgrids=[self.basis.powers(torch.linspace(-b,b,grid))[2] for b in limits]
        cover=torch.linspace(0,1,129)
        self.Tcover=self.basis.cheb(cover)[0]
        self.cover_lo=.5-torch.sqrt((.25-(cover-.5)**2).clamp(min=0))
        self.cover_hi=.5+torch.sqrt((.25-(cover-.5)**2).clamp(min=0))
        self.ptlos=[self.basis.powers(torch.tensor([-b]))[1][:,0] for b in limits]
        self.pthis=[self.basis.powers(torch.tensor([b]))[1][:,0] for b in limits]
        self.Tsource=torch.tensor(C.chebvander(np.linspace(-1,1,257),M).T.copy())
        gauss,weight=np.polynomial.legendre.leggauss(M+3)
        RR=[]
        for j in range(M+1):
            c=np.zeros(M+1);c[j]=1
            RR.append(4*C.chebval(gauss,C.chebder(c,2)))
        self.R=torch.tensor(np.sqrt(weight/2)[:,None]*np.array(RR).T)

    def unpack(self,z):
        eta=z[:self.ns] if self.fixed_source is None else self.fixed_source
        fields=z[self.neta:].reshape(self.nload,self.nm)
        return eta,fields

    def initial(self):
        eta=torch.zeros(self.ns);eta[0]=1
        st=self.basis.build(eta,False)
        x,t=torch.meshgrid(torch.linspace(0,1,9),torch.linspace(-self.tmax,self.tmax,9),indexing='ij')
        x,t=x.ravel(),t.ravel();pp=self.basis.parts(st,x,t)
        low=[0,1,2,self.basis.na,self.basis.na+1]
        A=torch.cat([pp[k][low].T for k in ('w','wx','wt')])
        fields=[];taus=[]
        for ell,gamma in zip(self.loads,self.gamma):
            H=torch.tensor(input_matrix(ell));xi=x-.5;zeta=(t-H[0,1]*xi)/H[1,1]
            y=zeta+.5
            u=.5*(H[0,0]*xi*xi+2*H[0,1]*xi*zeta+H[1,1]*zeta*zeta)-.25*torch.trace(H)*.25
            target=torch.cat([u-y*t-gamma*(.5*x*x-.5*t*t),
                              H[0,0]*xi+H[0,1]*zeta-gamma*x,
                              -y+gamma*t])
            coef=torch.linalg.lstsq(A,target,driver='gelsd').solution
            theta=torch.zeros(self.nm);theta[low]=coef;fields.append(theta)
            taus.append(H[0,1]*(self.x-.5)+H[1,1]*(self.y-.5))
        z=torch.cat(([eta] if self.neta else [])+[torch.cat(fields)])
        return z,torch.cat(taus)

    def feasible(self,z):
        eta,fields=self.unpack(z)
        f=eta@self.Tsource
        if float(f.min())<.1-1e-10 or float(f.max())>3+1e-10 or not bool(torch.isfinite(z).all()):return False
        st=self.basis.build(eta,False)
        for ell,(theta,gamma) in enumerate(zip(fields,self.gamma)):
            a,_=self.basis.omega_coefficients(st,theta,gamma,False)
            v=(a@self.Tgrid).T@self.Pttgrids[ell]
            if float(v.max())>-self.curvature_floor+1e-12:return False
            d=a@self.Tcover
            if float((self.ptlos[ell]@d+self.cover_lo).min())<-1e-10:return False
            if float((self.pthis[ell]@d+self.cover_hi).max())>1e-10:return False
        return True

    def linear_constraints(self,z):
        eta,fields=self.unpack(z);st=self.basis.build(eta,False)
        C=[];b=[]
        if self.neta:
            F=torch.zeros(self.Tsource.shape[1],self.p);F[:,:self.ns]=self.Tsource.T
            values=eta@self.Tsource
            C.extend([F,-F]);b.extend([3-values,values-.1])
        mode_grid=torch.einsum('jml,lx->jmx',st['A'],self.Tgrid)/self.basis.g[None,:,None]
        mode_cover=torch.einsum('jml,lx->jmx',st['A'],self.Tcover)/self.basis.g[None,:,None]
        for i,(theta,gamma) in enumerate(zip(fields,self.gamma)):
            a,da=self.basis.omega_coefficients(st,theta,gamma,bool(self.neta))
            vals=(a@self.Tgrid).T@self.Pttgrids[i]
            Q=torch.zeros(vals.numel(),self.p)
            if self.neta:
                Q[:,:self.ns]=torch.einsum('sjl,lx,jt->xts',da,self.Tgrid,self.Pttgrids[i]).reshape(-1,self.ns)
            cols=slice(self.neta+i*self.nm,self.neta+(i+1)*self.nm)
            Q[:,cols]=torch.einsum('jmx,jt->xtm',mode_grid,self.Pttgrids[i]).reshape(-1,self.nm)
            C.append(Q);b.append(-self.curvature_floor-vals.flatten())
            value=a@self.Tcover
            for sign,power,y in ((-1.,self.ptlos[i],self.cover_lo),(1.,self.pthis[i],self.cover_hi)):
                Q=torch.zeros(self.Tcover.shape[1],self.p)
                if self.neta:Q[:,:self.ns]=sign*torch.einsum('sjl,lx,j->xs',da,self.Tcover,power)
                Q[:,cols]=sign*torch.einsum('jmx,j->xm',mode_cover,power)
                C.append(Q);b.append(-sign*(power@value+y))
        return torch.cat(C),torch.cat(b)

    def project_field_feasibility(self,z):
        z=z.clone();eta,fields=self.unpack(z)
        f=eta@self.Tsource
        if float(f.min())<.1-1e-10 or float(f.max())>3+1e-10:return None
        st=self.basis.build(eta,False)
        for i,(theta,gamma) in enumerate(zip(fields,self.gamma)):
            a,_=self.basis.omega_coefficients(st,theta,gamma,False)
            short=self.curvature_floor+float(((a@self.Tgrid).T@self.Pttgrids[i]).max())
            cover=a@self.Tcover
            short=max(short,-float((self.ptlos[i]@cover+self.cover_lo).min())/self.tlimits[i],
                      float((self.pthis[i]@cover+self.cover_hi).max())/self.tlimits[i])
            if short>0:fields[i,2]-=(short+1e-11)*self.basis.g[2]
        return z

    def evaluate(self,z,tau,jac=True):
        eta,fields=self.unpack(z);st=self.basis.build(eta,False)
        tt=tau.reshape(self.nload,self.N)
        rg=[];rq=[];re=[];jg=[];jq=[];je=[];tg=[];tq=[];te=[]
        for ell,(theta,gamma) in enumerate(zip(fields,self.gamma)):
            a,da=self.basis.omega_coefficients(st,theta,gamma,jac and bool(self.neta))
            v=self.basis.evaluate(a,self.x,tt[ell],self.cache)
            e=v['wt']+self.y
            rg.append((v['w']+self.y*tt[ell]-self.g[ell])*self.wg[ell])
            rq.append((self.nx*v['wx']+self.ny*tt[ell]-self.q[ell])*self.wq[ell])
            re.append(e*self.we)
            if jac:
                pp=self.basis.parts(st,self.x,tt[ell],self.cache)
                G=torch.zeros(self.N,self.p);Q=torch.zeros_like(G);E=torch.zeros_like(G)
                if self.neta:
                    dv=self.basis.evaluate(da,self.x,tt[ell],self.cache)
                    G[:,:self.ns]=dv['w'].T;Q[:,:self.ns]=self.nx[:,None]*dv['wx'].T;E[:,:self.ns]=dv['wt'].T
                col=slice(self.neta+ell*self.nm,self.neta+(ell+1)*self.nm)
                G[:,col]=pp['w'].T;Q[:,col]=self.nx[:,None]*pp['wx'].T;E[:,col]=pp['wt'].T
                jg.append(G*self.wg[ell]);jq.append(Q*self.wq[ell,:,None]);je.append(E*self.we)
                tg.append(e*self.wg[ell]);tq.append((self.nx*v['wxt']+self.ny)*self.wq[ell]);te.append(v['wtt']*self.we)
        r=torch.cat([torch.cat(rg),torch.cat(rq),torch.cat(re)])
        out=dict(r=r,loss=.5*float(r@r))
        if self.neta and self.rho:
            rr=math.sqrt(self.rho)*(self.R@eta);out['loss']+=.5*float(rr@rr)
        if self.mu:out['loss']+=.5*self.mu*float(fields.flatten()@fields.flatten())
        if jac:
            out.update(J=torch.cat([torch.cat(jg),torch.cat(jq),torch.cat(je)]),
                       jt=torch.stack([torch.cat(tg),torch.cat(tq),torch.cat(te)]))
        return out

    def field_metrics(self,z,angles):
        eta,fields=self.unpack(z);st=self.basis.build(eta,False)
        x,y=map(torch.tensor,boundary(angles));nx=torch.tensor(np.cos(angles));ny=torch.tensor(np.sin(angles))
        ans=[]
        for ell,(theta,gamma) in enumerate(zip(fields,self.gamma)):
            a,_=self.basis.omega_coefficients(st,theta,gamma,False)
            t,nb=self.basis.solve_t(a,x,y,self.tlimits[ell]);v=self.basis.evaluate(a,x,t)
            pred=nx*v['wx']+ny*t
            error=v['w']+y*t-torch.tensor(Control(self.loads[ell]).value(x.numpy(),y.numpy()))
            ans.append(dict(q=pred.numpy(),g_rms=float(torch.mean(error**2).sqrt()),g_max=float(error.abs().max()),n_out=nb))
        return ans


def solve_joint(problem,z,tau,rounds=80,kappa=1e-3,callback=None,ftol=1e-4,
                deadline=None,state_callback=None):
    if not problem.feasible(z):raise ValueError('Initial source/field/coverage is infeasible')
    z=z.clone();tau=tau.clone();ev=problem.evaluate(z,tau);history=[]
    if state_callback:state_callback(problem,z,tau,dict(iteration=0,loss=ev['loss']))
    for iteration in range(rounds):
        if deadline is not None and time.perf_counter()>=deadline:break
        J,r,jt=ev['J'],ev['r'],ev['jt'];m=problem.m
        A=J.T@J;g=J.T@r;M=torch.zeros(problem.p,m);D=(jt*jt).sum(0);gt=torch.zeros(m)
        for channel in range(3):
            Ji=J[channel*m:(channel+1)*m];ri=r[channel*m:(channel+1)*m]
            M+=Ji.T*jt[channel];gt+=jt[channel]*ri
        if problem.neta and problem.rho:
            gram=problem.rho*(problem.R.T@problem.R)
            A[:problem.ns,:problem.ns]+=gram;g[:problem.ns]+=gram@z[:problem.ns]
        if problem.mu:
            A[problem.neta:,problem.neta:]+=problem.mu*torch.eye(problem.p-problem.neta)
            g[problem.neta:]+=problem.mu*z[problem.neta:]
        scale=A.diagonal().clamp(min=1e-30).sqrt();si=1/scale
        As=A*si[:,None]*si[None,:];gs=g*si;Ms=M*si[:,None]
        accepted=False;constraints=slack=None;qp={}
        for trial in range(18):
            if deadline is not None and time.perf_counter()>=deadline:break
            inv=1/(D+kappa)
            S=As+kappa*torch.eye(problem.p)-(Ms*inv)@Ms.T
            rhs=-gs+Ms@(gt*inv)
            L,info=torch.linalg.cholesky_ex(S)
            ds=(torch.cholesky_solve(rhs[:,None],L)[:,0] if int(info)==0 else torch.linalg.lstsq(S,rhs[:,None]).solution[:,0])*si
            znew=z+ds
            if not problem.feasible(znew):
                from .constrained_lm import solve_qp
                if constraints is None:constraints,slack=problem.linear_constraints(z)
                scaled,qp=solve_qp(S,rhs,constraints*si[None,:],slack)
                ds=scaled*si
                znew=problem.project_field_feasibility(z+ds)
                if znew is None or not problem.feasible(znew):kappa=min(kappa*4,1e14);continue
                ds=znew-z
            tnew=torch.maximum(problem.tlower,torch.minimum(problem.tupper,tau-(gt+M.T@ds)*inv))
            dt=tnew-tau
            pred=-float(g@ds+gt@dt)-.5*float(ds@(A@ds)+2*ds@(M@dt)+dt@(D*dt))
            vn=problem.evaluate(znew,tnew,False)
            gain=(ev['loss']-vn['loss'])/pred if pred>0 else -1
            if gain>1e-4 and vn['loss']<ev['loss']:
                next_ev=problem.evaluate(znew,tnew)
                if deadline is not None and time.perf_counter()>=deadline:break
                z,tau,ev=znew,tnew,next_ev;kappa=max(kappa/4,1e-14);accepted=True;break
            kappa=min(kappa*4,1e14)
        rec=dict(iteration=iteration+1,loss=ev['loss'],kappa=kappa,accepted=accepted,trial=trial+1,
                 qp_active=qp.get('qp_active',0),
                 qp_violation=qp.get('qp_violation',0.),
                 g_rms=float(torch.mean(ev['r'][:m]**2).sqrt()),
                 q_rms=float(torch.mean(ev['r'][m:2*m]**2).sqrt()),
                 position_rms=float(torch.mean(ev['r'][2*m:]**2).sqrt()))
        history.append(rec)
        if state_callback:state_callback(problem,z,tau,rec)
        if deadline is not None and time.perf_counter()>=deadline:
            rec['stop_reason']='time_limit';break
        if callback and (iteration==0 or (iteration+1)%10==0):callback(rec)
        if not accepted and kappa>=1e14:break
        if ev['loss']<1e-22:break
        if ftol is not None and len(history)>=21 and history[-21]['loss']-history[-1]['loss']<=ftol*max(history[-21]['loss'],1e-30):
            history[-1]['stop_reason']='relative_objective_change_over_20_steps';break
    return z,tau,history

def prolong(old,new,z,tau):
    eta,fields=old.unpack(z);nf=torch.zeros(new.nload,new.nm)
    for i in range(new.nload):
        for j in range(old.basis.na):
            nf[i,j]=fields[i,j]*new.basis.g[j]/old.basis.g[j]
            nf[i,new.basis.na+j]=fields[i,old.basis.na+j]*new.basis.g[new.basis.na+j]/old.basis.g[old.basis.na+j]
    eta_new=torch.zeros(new.ns);eta_new[:len(eta)]=eta
    carried=torch.cat(([eta_new] if new.neta else [])+[nf.flatten()])
    source,f=new.unpack(carried);st=new.basis.build(source,False)
    for i,(theta,gamma) in enumerate(zip(f,new.gamma)):
        a,_=new.basis.omega_coefficients(st,theta,gamma,False)
        shortfall=new.curvature_floor+float(((a@new.Tgrid).T@new.Pttgrids[i]).max())
        if 0<shortfall<1e-10:
            f[i,2]-=(shortfall+1e-11)*new.basis.g[2]
    return carried,tau.clone()
