"""Chart assembly, safeguarded inversion and equality-coordinate recovery.

theta with nmodes entries is the original equality chart, gamma=cert.bound(theta).
theta with nmodes+1 entries evaluates gamma*Phi+sum theta_j*w_j. Joint Dirichlet
fits fix this redundant representative gamma and optimize all nonzero modes.
Chart.equality_coordinates recovers the equivalent original h,gamma pair. The
sampled curvature margin and inverse failures are audited independently.
"""
from __future__ import annotations

import torch


class Chart:
    def __init__(self, basis, cert):
        self.basis = basis
        self.cert = cert
        self.n_alpha = basis.n_alpha
        self.nmodes = basis.nmodes
        self.n_par = basis.nmodes + 1              # + gamma
        self.t_lo, self.t_hi = basis.box[2], basis.box[3]

    # ------------------------------------------------------------------ gamma
    def split(self, theta):
        return theta[:self.nmodes], theta[self.nmodes]

    def gamma(self, theta) -> torch.Tensor:
        """gamma of a parameter vector (last entry) or of a coefficient vector."""
        if theta.numel() == self.nmodes + 1:
            return theta[self.nmodes]
        return self.cert.bound(theta)

    def cert_bound(self, theta) -> float:
        """delta + (1+iota) max_box H_tt   -- the right-hand side of (2.5)."""
        th = theta[:self.nmodes] if theta.numel() == self.nmodes + 1 else theta
        return float(self.cert.bound(th))

    def margin(self, theta) -> float:
        """gamma - max_box H_tt, which (2.5) requires to be >= delta."""
        if theta.numel() == self.nmodes + 1:
            th, g = theta[:self.nmodes], theta[self.nmodes]
        else:
            th, g = theta, self.cert.bound(theta)
        return float(g) - float(self.cert.max_htt(th))

    def phi_coefficients(self):
        """q with Phi=sum q_j w_j (K>=2)."""
        if self.basis.K < 2:
            raise ValueError('Stable gauge requires K >= 2')
        q = torch.zeros(self.nmodes)
        q[2] = (0.5*self.basis.R**2*self.basis.g[2]
                if self.basis.kind == 'complex' else -self.basis.g[2])
        return q

    def equality_coordinates(self, theta):
        """Equivalent original equality-gamma coefficients; evaluate in stable gauge.

        With c=theta_coef+gamma_ref*q, S=max_grid(c@WTT),
        Gamma=-((1+iota)*S+delta)/iota and h=c-Gamma*q.
        Then Gamma=delta+(1+iota)*max_grid(h@WTT), exactly in real arithmetic.
        This is a coordinate change for the sampled-max equality, not l1.
        """
        if self.cert.mode != 'box':
            raise ValueError('Equality remapping applies to sampled max only')
        q = self.phi_coefficients()
        c = theta[:self.nmodes] + self.gamma(theta)*q
        S = torch.max(c @ self.cert.S)
        G = -((1+self.cert.iota)*S+self.cert.delta)/self.cert.iota
        return c-G*q, G

    # -------------------------------------------------------------- assembly
    def eval(self, x, t, theta, gamma=None, parts=None) -> dict:
        """omega and its derivatives at (x,t), plus the pieces the Jacobian needs."""
        P = self.basis.parts(x, t) if parts is None else parts
        Phi, Phi_x, Phi_t = self.basis.phi(x, t)
        if gamma is None:
            gamma = self.gamma(theta)
        a = theta[:self.n_alpha]
        b = theta[self.n_alpha:2 * self.n_alpha]
        na = self.n_alpha
        lin = lambda A: a @ A[:na] + b @ A[na:]                      # noqa: E731
        return dict(
            w=lin(P['W']) + gamma * Phi,
            wx=lin(P['WX']) + gamma * Phi_x,
            wt=lin(P['WT']) + gamma * Phi_t,
            wtt=lin(P['WTT']) - gamma,
            wxt=lin(P['WXT']),
            Phi=Phi, Phi_x=Phi_x, Phi_t=Phi_t, gamma=gamma, parts=P,
        )

    # ------------------------------------------------------------- inversion
    def _bisect(self, x, y, theta, g, nbis: int = 60):
        lo = torch.full_like(x, self.t_lo)
        hi = torch.full_like(x, self.t_hi)
        for _ in range(nbis):
            mid = 0.5 * (lo + hi)
            pos = self.eval(x, mid, theta, g)['wt'] + y > 0
            lo = torch.where(pos, mid, lo)
            hi = torch.where(pos, hi, mid)
        return 0.5 * (lo + hi)

    def bracket_ok(self, x, y, theta, gamma=None):
        """True where the root lies inside the prior t-box (i.e. the chart covers y)."""
        g = self.gamma(theta) if gamma is None else gamma
        with torch.no_grad():
            glo = self.eval(x, torch.full_like(x, self.t_lo), theta, g)['wt'] + y
            ghi = self.eval(x, torch.full_like(x, self.t_hi), theta, g)['wt'] + y
            return (glo >= -1e-12) & (ghi <= 1e-12)

    def solve_t(self, x, y, theta, gamma=None, t0=None, nnewt: int = 15,
                nbis: int = 60, tol: float = 1e-13):
        """Solve omega_t(x,t) + y = 0.  Returns (t, n_residual_above_1e-9)."""
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        g = self.gamma(theta) if gamma is None else gamma
        with torch.no_grad():
            t = (torch.full_like(x,0.5*(self.t_lo+self.t_hi)) if t0 is None
                 else torch.clamp(t0.clone(),self.t_lo,self.t_hi))
            lo=torch.full_like(x,self.t_lo);hi=torch.full_like(x,self.t_hi)
            for _ in range(nnewt):
                st=self.eval(x,t,theta,g);e=st['wt']+y
                if float(e.abs().max())<tol:break
                lo=torch.where(e>0,t,lo);hi=torch.where(e<=0,t,hi)
                tn=t-e/st['wtt']
                ok=torch.isfinite(tn)&(tn>=lo)&(tn<=hi)
                t=torch.where(ok,tn,0.5*(lo+hi))
            bad=(self.eval(x,t,theta,g)['wt']+y).abs()>1e-9
            if bool(bad.any()):
                tb=self._bisect(x[bad],y[bad],theta,g,nbis)
                t=t.clone();t[bad]=tb
            res = (self.eval(x, t, theta, g)['wt'] + y).abs()
        valid = self.bracket_ok(x, y, theta, g)
        return t, int(((res > 1e-9) | ~valid | ~torch.isfinite(res)).sum())

    def solve_t_diff(self, x, y, theta, gamma=None, t0=None, nnewt: int = 3):
        """Differentiable inversion: Newton steps kept in the autograd graph.

        At a fixed point the composite map has exactly the implicit-function
        derivative, so differentiating through it is exact; this is what the
        PDE-residual audit uses.
        """
        g = self.gamma(theta) if gamma is None else gamma
        if t0 is None:
            t0, _ = self.solve_t(x, y, theta, g)
        t = t0.detach()
        for _ in range(nnewt):
            st = self.eval(x, t, theta, g)
            t = t - (st['wt'] + y) / st['wtt']
        return t

    # ------------------------------------------------------------------- u
    def field(self, x, y, theta, gamma=None, t0=None):
        """u, u_x = omega_x, u_y = t on given points (no autograd)."""
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
        g = self.gamma(theta) if gamma is None else gamma
        t, n_out = self.solve_t(x, y, theta, g, t0=t0)
        with torch.no_grad():
            st = self.eval(x, t, theta, g)
            return dict(u=st['w'] + y * t, ux=st['wx'], uy=t, t=t,
                        n_out=n_out, gamma=float(g))
