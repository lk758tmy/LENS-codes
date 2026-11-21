"""Unified joint Schur-LM transport solver with resumable stage checkpoints."""
from __future__ import annotations
import json,os,time
from pathlib import Path
import numpy as np
import torch
from . import basis_const,basis_fx,cert,chart,res,lm,metrics
from . import transport_cases as cases

def _sqrt(A):
    v,Q=np.linalg.eigh(A)
    if np.min(v)<=0:raise ValueError('Nonpositive covariance')
    return (Q*np.sqrt(v))@Q.T

def distance_polyline(px,py,bx,by):
    p=torch.as_tensor(np.column_stack([px,py]));b=torch.as_tensor(np.column_stack([bx,by]))
    e=torch.roll(b,-1,0)-b;length=(e*e).sum(1).clamp(min=1e-30);out=[]
    for q in p.split(128):
        d=q[:,None,:]-b[None,:,:];s=(d*e[None]).sum(2)/length[None]
        v=d-s.clamp(0,1)[:,:,None]*e[None]
        out.append((v*v).sum(2).min(1).values.sqrt())
    return torch.cat(out).numpy()

def self_crosses(x,y):
    p=np.column_stack([x,y]);p=p[::max(1,len(p)//512)];q=np.roll(p,-1,axis=0);n=len(p)
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for i in range(n):
        js=np.arange(i+2,n)
        if i==0:js=js[js<n-1]
        if not len(js):continue
        a=cross(q[i]-p[i],p[js]-p[i]);b=cross(q[i]-p[i],q[js]-p[i])
        c=cross(q[js]-p[js],p[i]-p[js]);d=cross(q[js]-p[js],q[i]-p[js])
        if np.any((a*b < -1e-22)&(c*d < -1e-22)):return True
    return False

class TransportFit:
    def __init__(self,region,target,K=16,n_edge=256,source=None,truth=None,
                 delta=.1,iota=.02,lam=.4,mu=1e-8,rounds1=60,rounds=200,
                 t_pad=.05,grid=65,audit=65,checkpoint=None,fingerprint='',
                 progress=None,gamma_mode='gauge',gamma_policy='project',
                 loss_norm='sq',tau_mode='joint',corner_weight=0.,source_vertices=False,**unsupported):
        if unsupported:raise TypeError(f'Unsupported transport options: {list(unsupported)}')
        if (gamma_mode,loss_norm,tau_mode)!=('gauge','sq','joint'):
            raise ValueError('Formal transport uses stable coordinates and joint square residuals')
        self.region,self.T=region,target;self.source=source or cases.make_source(0);self.truth=truth
        self.K,self.m=int(K),int(n_edge);self.delta,self.iota,self.lam=delta,iota,lam
        self.mu,self.rounds1,self.rounds=mu,rounds1,rounds;self.t_pad=t_pad
        self.corner_weight=float(corner_weight)
        self.source_vertices=bool(source_vertices)
        self.grid,self.audit=grid,audit;self.gamma_policy=gamma_policy
        self.checkpoint=Path(checkpoint) if checkpoint else None
        self.fingerprint=fingerprint;self.progress=progress
        self.hist=[];self.stage_iters=[0,0,0];self.timings={};self._fit_seconds=0.

    def _basis(self,K):
        if self.source.kind=='const':return basis_const.ConstBasis(c=float(self.source.kw.get('c',1)),K=K,box=self.box)
        return basis_fx.FxBasis(self.source,self.box,K=K,seed='paper')
    def _set_chart(self,K):
        self.basis=self._basis(K)
        # Same sampled max, with dense boundary sampling to resolve active peaks.
        x0,x1,t0,t1=self.box
        a=torch.linspace(x0,x1,1025);b=torch.linspace(t0,t1,1025)
        ex=torch.cat([a,a,torch.full_like(b,x0),torch.full_like(b,x1)])
        et=torch.cat([torch.full_like(a,t0),torch.full_like(a,t1),b,b])
        edge=self.basis.parts(ex,et)['WTT']
        self.cert=cert.Certificate(torch.cat([self.basis.cert_S(61),edge],dim=1),
                                  'box',self.delta,self.iota)
        self.chart=chart.Chart(self.basis,self.cert)
    def _problem(self,mu):
        p=res.TransportBVP(self.chart,self.x,self.y,self.T.dist_normal,
                          lam=self.lam,mu=mu,tau_mode='joint',gamma_mode='gauge',
                          gamma_policy=self.gamma_policy,loss_norm='sq',
                          corners=(None if isinstance(self.T,__import__('lens.targets',fromlist=['SmoothTarget']).SmoothTarget)
                                   else np.column_stack([self.T.ps,self.T.qs])),
                          corner_weight=self.corner_weight)
        p.curvature_floor=self.delta*(1+self.iota)
        p.enforce_linear_curvature=True
        p.t_lo,p.t_hi=self.T.qmin,self.T.qmax
        return p

    def build(self):
        start=time.perf_counter()
        self.mass,self.center,self.cov=cases.source_moments(self.region,self.source,32)
        M2,*_=cases.source_moments(self.region,self.source,48)
        self.mass_quad_error=abs(M2-self.mass)/self.mass
        self.input_mass_error=abs(self.T.area-self.mass)/self.mass
        if self.input_mass_error>1e-10 or self.mass_quad_error>1e-11:
            raise ValueError(f'Mass incompatibility: {self.mass}, target {self.T.area}')
        _,ct,Ct=self.T.moments();S=_sqrt(self.cov);Si=np.linalg.inv(S)
        A=Si@_sqrt(S@Ct@S)@Si
        self.reference_A=A;self.G_ref=1/float(A[1,1])
        if self.G_ref<self.delta:raise ValueError('Reference curvature below floor')
        x0,x1,_,_=self.region.bbox;span=self.T.qmax-self.T.qmin
        pad=self.t_pad*span+.001
        self.box=(x0,x1,self.T.qmin-pad,self.T.qmax+pad)
        self._set_chart(min(4,self.K))
        xb,yb=self.region.boundary(self.m)
        if self.source_vertices and hasattr(self.region,'V'):
            pts=np.column_stack([xb,yb]);v=self.region.V
            v=v[np.min(np.linalg.norm(v[:,None]-pts[None],axis=2),axis=1)>1e-13]
            pts=np.vstack([pts,v]);xb,yb=pts.T
        self.m=len(xb)
        self.x,self.y=torch.as_tensor(xb),torch.as_tensor(yb)
        self.xb,self.yb=xb,yb
        # A single PDE-exact low-order reference, not a boundary correspondence fit.
        xc,yc=self.center;pc,qc=ct;d=float(A[0,1]/A[1,1])
        ph,phx,pht=self.basis.phi(torch.tensor([xc]),torch.tensor([qc]))
        bx=float(pc-self.G_ref*float(phx[0])-d*qc)
        bt=float(-yc-self.G_ref*float(pht[0])-d*xc)
        xx,tt=torch.meshgrid(torch.linspace(x0,x1,13),torch.linspace(self.box[2],self.box[3],13),indexing='ij')
        xx,tt=xx.flatten(),tt.flatten();P=self.basis.parts(xx,tt)
        pr=self._problem(0);keep=pr.free[:-1]
        AA=torch.cat([P['WX'][keep].T,P['WT'][keep].T])
        rhs=torch.cat([d*tt+bx,d*xx+bt])
        co=torch.linalg.lstsq(AA,rhs,driver='gelsd').solution
        self.theta0=torch.zeros(self.basis.nmodes+1);self.theta0[-1]=self.G_ref;self.theta0[:-1][keep]=co
        self.t0,nb=self.chart.solve_t(self.x,self.y,self.theta0)
        self.initial_inverse_failures=nb
        self.t0=self.t0.clamp(self.T.qmin,self.T.qmax)
        ev=pr.eval(self.theta0,self.t0)
        self.initial_residuals=(ev['r1'],ev['r2'])
        self.timings['build']=time.perf_counter()-start
        return self

    def _save(self,stage,iteration,theta,t,kappa):
        if not self.checkpoint:return
        meta=dict(fingerprint=self.fingerprint,stage=stage,iteration=iteration,kappa=kappa,
                  hist=self.hist,stage_iters=self.stage_iters,fit_seconds=self._fit_seconds)
        self.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.checkpoint.with_suffix('.tmp.npz')
        np.savez(tmp,theta=theta.detach().numpy(),t=t.detach().numpy(),meta=json.dumps(meta))
        os.replace(tmp,self.checkpoint)

    def _stage(self,stage,theta,t,rounds,mu,state=None):
        offset=0;kappa=1e-3
        if state is not None and state['stage']==stage:
            offset=state['iteration'];kappa=state['kappa']
        if offset>=rounds:return theta,t
        p=self._problem(mu);start=time.perf_counter();prior=self._fit_seconds
        def cb(it,th,ta,ev):
            self.stage_iters[stage]=offset+it
            rec=dict(stage=stage,iteration=offset+it,r1=ev['r1'],r2=ev['r2'])
            # The callback is before the solver appends its own history.
            rec.update(getattr(p,'qp_info',{}))
            self._pending=rec
            if self.progress and (it==1 or (offset+it)%10==0):self.progress(rec)
            return {}
        def observe(it,th,ta,rec):
            self.hist.append(dict(rec,stage=stage,iteration=offset+it,
                                  r1=self._pending['r1'],r2=self._pending['r2']))
            self._fit_seconds=prior+time.perf_counter()-start
            if (offset+it)%10==0:self._save(stage,offset+it,th,ta,rec['kappa'])
        o=lm.solve(p,theta,t,rounds=rounds-offset,kappa0=kappa,
                   ftol=1e-26*self.m if mu==0 else 0,callback=cb,observer=observe)
        self._fit_seconds=prior+time.perf_counter()-start
        self.stage_iters[stage]=offset+len(o['hist'])
        return o['theta'],o['t']

    def fit(self):
        state=None;th=self.theta0.clone();t=self.t0.clone()
        if self.checkpoint and self.checkpoint.exists():
            with np.load(self.checkpoint,allow_pickle=False) as a:
                state=json.loads(str(a['meta']))
                if state['fingerprint']!=self.fingerprint:raise ValueError('Checkpoint code/config mismatch')
                th=torch.tensor(a['theta']);t=torch.tensor(a['t'])
            self.hist=state['hist'];self.stage_iters=state['stage_iters'];self._fit_seconds=state['fit_seconds']
        phase=state['stage'] if state else 0
        if phase==0:
            if state is None: self._save(0,0,th,t,1e-3)
            th,t=self._stage(0,th,t,self.rounds1,0,state)
            old=self.basis;oldth=th;self._set_chart(self.K)
            th=torch.zeros(self.basis.nmodes+1);th[-1]=self.G_ref
            n=old.n_alpha;nn=self.basis.n_alpha
            th[:n]=oldth[:n]*self.basis.g[:n]/old.g[:n]
            th[nn:nn+n]=oldth[n:2*n]*self.basis.g[nn:nn+n]/old.g[n:2*n]
            self._save(1,0,th,t,1e-3);phase=1;state=None
        else:self._set_chart(self.K)
        if phase==1:
            th,t=self._stage(1,th,t,self.rounds1,self.mu,state)
            self._save(2,0,th,t,1e-3);phase=2;state=None
        if phase==2:
            th,t=self._stage(2,th,t,self.rounds,0,state)
            self._save(3,0,th,t,1e-3)
        self.theta,self.t=th,t;self.iters=tuple(self.stage_iters)
        self.timings['fit']=self._fit_seconds;self.prob=self._problem(0)
        return self

    def query(self,n=10000,reps=5):
        rng=np.random.default_rng(12345);x0,x1,y0,y1=self.region.bbox;pts=[];total=0
        while total<n:
            x=rng.uniform(x0,x1,3*n);y=rng.uniform(y0,y1,3*n);m=self.region.contains(x,y)
            pts.append((x[m],y[m]));total+=int(m.sum())
        x=torch.tensor(np.concatenate([p[0] for p in pts])[:n]);y=torch.tensor(np.concatenate([p[1] for p in pts])[:n])
        ts=[]
        for _ in range(reps):
            start=time.perf_counter();self.chart.field(x,y,self.theta);ts.append(time.perf_counter()-start)
        self.timings['query']=float(np.median(ts));return self

    def measure(self,boundary_n=2048):
        ch,T=self.chart,self.T;ell=np.sqrt(T.area);x,y=cases.boundary(self.region,boundary_n,True)
        fld=metrics.field_on_grid(ch,self.theta,x,y);px,py=fld['ux'],fld['uy']
        d=T.dist_normal(torch.tensor(px),torch.tensor(py))[0].numpy()
        tx,ty=T.boundary(boundary_n,include_vertices=True);back=distance_polyline(tx,ty,px,py)
        X,Y=self.region.interior_grid(self.grid);fi=metrics.field_on_grid(ch,self.theta,X,Y)
        Xa,Ya=self.region.interior_grid(self.audit)
        rp=metrics.audit_pde(ch,self.theta,Xa,Ya,metrics.f_torch(self.source))
        ev=self.prob.eval(self.theta,self.t,False)
        x0,x1,t0,t1=self.box
        a=torch.linspace(x0,x1,4097);b=torch.linspace(t0,t1,4097)
        ex=torch.cat([a,a,torch.full_like(b,x0),torch.full_like(b,x1)])
        et=torch.cat([torch.full_like(a,t0),torch.full_like(a,t1),b,b])
        dense=torch.cat([self.basis.cert_S(65),self.basis.parts(ex,et)['WTT']],dim=1)
        margin=float(self.theta[-1]-(self.theta[:-1]@dense).max())
        area=.5*np.sum(px*np.roll(py,-1)-np.roll(px,-1)*py)
        xx,yy=cases.boundary(self.region,2*boundary_n,True);ff=metrics.field_on_grid(ch,self.theta,xx,yy)
        area2=.5*np.sum(ff['ux']*np.roll(ff['uy'],-1)-np.roll(ff['ux'],-1)*ff['uy'])
        crossed=self_crosses(px,py);cent=T.center();vx,vy=px-cent[0],py-cent[1]
        winding=np.arctan2(vx*np.roll(vy,-1)-vy*np.roll(vx,-1),vx*np.roll(vx,-1)+vy*np.roll(vy,-1)).sum()/(2*np.pi)
        interior_d=T.dist_normal(torch.tensor(fi['ux']),torch.tensor(fi['uy']))[0].numpy()
        qx,qy,qw=cases.quadrature(self.region,32);qf=metrics.field_on_grid(ch,self.theta,qx,qy)
        w=qw*self.source(qx)/self.mass;g=np.column_stack([qf['ux'],qf['uy']]);mean=(w[:,None]*g).sum(0)
        second=(g.T*w)@g;_,mt,ct=T.moments();target2=ct+np.outer(mt,mt)
        nout=fld['n_out']+fi['n_out']+ff['n_out']+qf['n_out']
        mass_error=abs(area2-self.mass)/self.mass if not crossed and not nout else float('nan')
        h=max(float(np.max(np.abs(d))),float(back.max()))/ell
        self.metrics=dict(dH=h,forward_rms=float(np.sqrt(np.mean(d*d)))/ell,
            forward_max=float(np.abs(d).max())/ell,backward_max=float(back.max())/ell,
            r_inv=ev['r2']/(self.region.bbox[3]-self.region.bbox[2]),r1_opt=ev['r1'],
            input_mass_error=self.input_mass_error,mass_quad_error=self.mass_quad_error,
            mass_error=mass_error,area_signed=float(area2),area_refinement=abs(area2-area)/self.mass,
            dense_margin=margin,margin=ch.margin(self.theta),Rpde=rp,n_out=nout,
            winding=float(winding),sampled_self_crossing=crossed,
            interior_out_fraction=float(np.mean(interior_d>1e-8*ell)),
            centroid_error=float(np.linalg.norm(mean-mt)/ell),
            second_moment_error=float(np.linalg.norm(second-target2)/T.area),
            q_coverage=float((py.max()-py.min())/(T.qmax-T.qmin)),
            gamma_equality=float(ch.equality_coordinates(self.theta)[1]),G_ref=self.G_ref)
        if not isinstance(T, __import__('lens.targets',fromlist=['SmoothTarget']).SmoothTarget):
            self.metrics['corner_max']=float(distance_polyline(T.ps,T.qs,px,py).max()/ell)
            self.metrics['corner_joint_sample_max']=float(torch.cdist(
                torch.tensor(np.column_stack([T.ps,T.qs])),
                torch.stack([ev['state']['wx'],self.t],dim=1)).min(1).values.max())/ell
        if self.truth:
            ut,ux,uy=self.truth(X,Y)
            self.metrics.update(metrics.errors(fi['u'],ut,center_reference=True,align_mean=True))
            self.metrics['E_grad']=float(np.sqrt(np.mean((fi['ux']-ux)**2+(fi['uy']-uy)**2))/np.sqrt(np.mean(ux*ux+uy*uy)))
        admissible=(nout==0 and margin>=self.delta-1e-10 and np.isfinite(rp) and not crossed)
        fit_ok=h<=1e-3 and self.metrics['r_inv']<=1e-6 and mass_error<=1e-4
        self.status='fit_converged' if admissible and fit_ok else 'poor_fit' if admissible else 'invalid'
        self.audit_field=dict(x=x,y=y,px=px,py=py,tx=tx,ty=ty,ix=X,iy=Y,ip=fi['ux'],iq=fi['uy'])
        return self.metrics
    def run(self):self.build().fit().query();self.measure();return self
