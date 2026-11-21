"""Dirichlet driver: boundary-only reference, joint Schur LM, inversion query.

The fixed representative gamma removes the exact Phi/A2 redundancy. Original
sampled-max equality coordinates are recovered by Chart.equality_coordinates;
the PDE family and finite joint residual are unchanged. No latent elimination,
alternating linear solves or source-specific initialization stages are used.
The common low-order LM selects the box; ridge warm-up and zero-ridge polish
then use the requested order. All stages are timed as fitting.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import basis_const, cert as cert_mod, chart as chart_mod, init as init_mod
from . import lm, metrics, res

lm_mod, metrics_mod, res_mod = lm, metrics, res


class DirichletFit:
    def __init__(self, region, source, K: int, m: int, truth=None,
                 delta: float = 0.1, iota: float = 0.02, cert_mode: str = 'box',
                 margin: float = 0.35, tau_mode: str = 'joint', lam: float = 0.4,
                 mu1: float = 1e-8, rounds1: int = 60, rounds2: int = 200,
                 t_box=None, seed: int = 0, noise: float = 0.0, grid: int = 65,
                 audit: int = 65, basis_kind: str = 'const', fx_seed: str = 'paper',
                 fx_refine: int = 4, gamma_policy: str = 'reject'):
        self.region = region
        self.source = source
        self.K = int(K)
        self.m = int(m)
        self.truth = truth
        self.delta, self.iota, self.cert_mode = float(delta), float(iota), cert_mode
        self.margin = float(margin)
        self.tau_mode, self.lam = tau_mode, float(lam)
        self.mu1, self.rounds1, self.rounds2 = float(mu1), int(rounds1), int(rounds2)
        self.t_box = t_box
        self.seed = int(seed)
        self.noise = float(noise)
        self.grid = int(grid)
        self.audit = int(audit)
        self.basis_kind = basis_kind
        if basis_kind == 'fx' and fx_seed != 'paper':
            raise ValueError('Formal FxBasis uses paper seeds; select alternatives directly for ablations')
        self.fx_seed = fx_seed
        self.fx_refine = int(fx_refine)
        self.gamma_policy = gamma_policy
        self.timings: dict[str, float] = {}

    # ---------------------------------------------------------------- build
    def build(self):
        t0 = time.perf_counter()
        xb, yb = self.region.boundary(self.m, seed=self.seed if self.noise else None)
        g = self.truth(xb, yb)[0] if self.truth is not None else None
        if g is None:
            raise ValueError('DirichletFit needs a truth to define the data')
        self.noise_sigma = 0.0
        if self.noise > 0:
            rng = np.random.default_rng(self.seed)
            g = g + self.noise * float(np.std(g)) * rng.standard_normal(g.shape)
            self.noise_sigma = self.noise * float(np.std(g))
        self.xb, self.yb, self.g = xb, yb, g
        ini = init_mod.boundary_reference(
            self.region, xb, yb, g, self.source, K=self.K,
            basis_kind=self.basis_kind, t_box=self.t_box, margin=self.margin,
            delta=self.delta)
        self.basis = ini['basis']
        self.G = ini['G']
        self.S = self.basis.cert_S(61)
        self.cert = cert_mod.Certificate(self.S, self.cert_mode, self.delta,
                                         self.iota, W_disc=self.basis.W_disc())
        self.chart = chart_mod.Chart(self.basis, self.cert)
        self.theta0 = ini['theta0']
        self.t0, self.n_out0 = self.chart.solve_t(
            torch.as_tensor(xb), torch.as_tensor(yb), self.theta0)
        self.timings['build'] = time.perf_counter() - t0
        return self

    # ------------------------------------------------------------------ fit
    def fit(self):
        t0 = time.perf_counter()
        xt, yt = torch.as_tensor(self.xb), torch.as_tensor(self.yb)
        # One low-order joint LM warm start for EVERY Dirichlet case.
        ini = init_mod.boundary_reference(
            self.region,self.xb,self.yb,self.g,self.source,K=min(4,self.K),
            basis_kind=self.basis_kind,t_box=self.t_box,margin=self.margin,
            delta=self.delta)
        bc=ini['basis']
        cc=chart_mod.Chart(bc,cert_mod.Certificate(bc.cert_S(61),self.cert_mode,
                                self.delta,self.iota,W_disc=bc.W_disc()))
        pc=res.DirichletBVP(cc,xt,yt,self.g,lam=self.lam,mu=0,
                          tau_mode=self.tau_mode,gamma_mode='gauge')
        tc,_=cc.solve_t(xt,yt,ini['theta0'])
        oc=lm.solve(pc,ini['theta0'],tc,rounds=self.rounds1,ftol=1e-26*self.m)
        # Fixed box for the requested-order solve, inferred from the coarse fit.
        # Only the low-order LM uses the broad reference box.
        xbd,ybd=self.region.boundary(max(512,self.m))
        tbd,nb=cc.solve_t(torch.as_tensor(xbd),torch.as_tensor(ybd),oc['theta'])
        if nb: raise RuntimeError('Coarse LM does not cover the boundary')
        span=float(tbd.max()-tbd.min())
        tb=(float(tbd.min())-0.10*span-1e-3,float(tbd.max())+0.10*span+1e-3)
        if self.t_box is not None:tb=self.t_box
        ni=init_mod.boundary_reference(self.region,self.xb,self.yb,self.g,self.source,
                            K=self.K,basis_kind=self.basis_kind,t_box=tb,
                            delta=self.delta)
        self.basis=ni['basis']
        self.S=self.basis.cert_S(61)
        self.cert=cert_mod.Certificate(self.S,self.cert_mode,self.delta,self.iota,
                                      W_disc=self.basis.W_disc())
        self.chart=chart_mod.Chart(self.basis,self.cert)
        # Exact polynomial prolongation, assembled using value/first-derivative
        # samples of the LOW modes only; no boundary-data solve is hidden here.
        xx,tt=torch.meshgrid(torch.linspace(self.basis.box[0],self.basis.box[1],13),
                            torch.linspace(tb[0],tb[1],13),indexing='ij')
        xx,tt=xx.flatten(),tt.flatten()
        st=cc.eval(xx,tt,oc['theta']);P=self.basis.parts(xx,tt)
        phi,phix,phit=self.basis.phi(xx,tt)
        low=list(range(bc.n_alpha))+list(range(self.basis.n_alpha,self.basis.n_alpha+bc.n_alpha))
        low=[j for j in low if not(self.basis.kind=='complex' and j==self.basis.n_alpha)]
        AA=torch.cat([P[k][low].t() for k in ('W','WT','WX')])
        G=float(oc['theta'][-1])
        vv=torch.cat([st['w']-G*phi,st['wt']-G*phit,st['wx']-G*phix])
        co=torch.linalg.lstsq(AA,vv,driver='gelsd').solution
        th=torch.zeros(self.basis.nmodes+1);th[-1]=G;th[low]=co
        # At an active reference floor, least-squares prolongation can lose a
        # few ulps of curvature. Restore only this roundoff-sized shortfall;
        # a genuinely infeasible coarse field still fails the reject policy.
        shortfall=self.delta-self.chart.margin(th)
        roundoff=64*torch.finfo(th.dtype).eps*max(1.,abs(G),self.delta)
        if G == self.delta and 0 < shortfall <= roundoff:
            th[:-1]+=(shortfall+roundoff)*self.chart.phi_coefficients()
        self.prolongation_error=float((AA@th[low]-vv).abs().max())
        self.theta0_fine=th.clone()
        tf,_=self.chart.solve_t(xt,yt,th,t0=oc['t'])
        pf=res.DirichletBVP(self.chart,xt,yt,self.g,lam=self.lam,mu=self.mu1,
              tau_mode=self.tau_mode,gamma_mode='gauge',gamma_policy=self.gamma_policy)
        ow=lm.solve(pf,th,tf,rounds=self.rounds1,ftol=1e-26*self.m)
        pf.mu=0.0
        of=lm.solve(pf,ow['theta'],ow['t'],rounds=self.rounds2,ftol=1e-26*self.m)
        self.theta,self.t=of['theta'],of['t']
        self.hist=oc['hist']+ow['hist']+of['hist']
        self.iters=(len(oc['hist']),len(ow['hist']),len(of['hist']))
        self.timings['fit'] = time.perf_counter() - t0
        return self

    # ---------------------------------------------------------------- query
    def query(self, n: int = 10000, reps: int = 5):
        rng = np.random.default_rng(12345)
        x0, x1, y0, y1 = self.region.bbox
        pts = []
        while sum(len(p[0]) for p in pts) < n:
            X = x0 + (x1 - x0) * rng.random(4 * n)
            Y = y0 + (y1 - y0) * rng.random(4 * n)
            msk = self.region.contains(X, Y)
            pts.append((torch.as_tensor(X[msk]), torch.as_tensor(Y[msk])))
        X = torch.cat([p[0] for p in pts])[:n]
        Y = torch.cat([p[1] for p in pts])[:n]
        f = lambda: self.chart.field(X, Y, self.theta)      # noqa: E731
        dt, _ = _median(f, reps)
        self.timings['query'] = dt
        return self

    # -------------------------------------------------------------- measure
    def measure(self) -> dict:
        if self.truth is None:
            return {}
        X, Y = self.region.interior_grid(self.grid)
        fld = metrics.field_on_grid(self.chart, self.theta, X, Y)
        u_t, ux_t, uy_t = self.truth(X, Y)
        e = metrics.errors(fld['u'], u_t)
        h1 = metrics.h1_semi(fld['ux'], fld['uy'], ux_t, uy_t)
        Xa, Ya = self.region.interior_grid(self.audit)
        rp = metrics.audit_pde(self.chart, self.theta, Xa, Ya,
                               metrics.f_torch(self.source))
        xt, yt = torch.as_tensor(self.xb), torch.as_tensor(self.yb)
        prob = res.DirichletBVP(self.chart, xt, yt, self.g, lam=self.lam,
                                tau_mode=self.tau_mode, gamma_mode='gauge')
        ev = prob.eval(self.theta, self.t)
        self.metrics = dict(
            E2=e['E2'], Einf=e['Einf'], H1=h1, Rpde=rp,
            rms=ev['r1'], res1=ev['r1'], res2=ev['r2'],
            gamma=float(self.theta[-1]), gamma_equality=float(self.chart.equality_coordinates(self.theta)[1]) if self.cert_mode=='box' else None, margin=self.chart.margin(self.theta),
            certified=bool(self.chart.margin(self.theta) >= self.delta - 1e-12),
            n_out=fld['n_out'], G_init=self.G,
            t_min=float(fld['t'].min()), t_max=float(fld['t'].max()),
        )
        return self.metrics

    # ------------------------------------------------------------------ all
    def run(self):
        self.build().fit().query()
        self.measure()
        return self


def _median(fn, reps: int):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], out


# The transport implementation shares the joint LM and stable chart modules.
from .transport_fit import TransportFit
