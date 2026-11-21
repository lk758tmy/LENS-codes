"""Residuals and analytic Jacobians for the two boundary conditions.

Unified notation (paper eq. 3.12):

    r_{1,i} = u_hat_i - (data)                     weight 1
    r_{2,i} = sqrt(lam) * (omega_t(x_i,t_i) + y_i) weight sqrt(lam)

  (D) Dirichlet   u_hat_i = omega(x_i,t_i) + y_i t_i,  data = g_i
  (T) transport   u_hat_i = d_T(omega_x(x_i,t_i), t_i)  (distance to target curve)

``gamma_mode``
    'param' : gamma is the last entry of theta (default).  The certificate (2.5)
              becomes the one-sided constraint enforced by ``project``.
    'gauge' : fixed representative gamma, all nonzero modes free (same span).
    'cert'  : gamma = (2.5) evaluated at theta; its derivative then enters the
              Jacobian (the paper's literal statement, kept for comparison).

``tau_mode``
    'joint'   : latent coordinates tau are optimisation variables (paper 3.11).
    'reduced' : tau is eliminated pointwise; only the coefficients are optimised
                (optional comparison, a different finite-residual objective).
"""
from __future__ import annotations

import math

import torch


class BVP:
    bc = 'abstract'
    with_tau = True

    def __init__(self, chart, lam: float = 0.4, mu: float = 0.0,
                 tau_mode: str = 'joint', gamma_mode: str = 'param',
                 gamma_policy: str = 'none'):
        self.chart = chart
        self.lam = float(lam)
        self.sqrt_lam = math.sqrt(self.lam)
        self.mu = float(mu)
        self.tau_mode = tau_mode
        self.with_tau = (tau_mode == 'joint')
        self.gamma_mode = gamma_mode
        self.gamma_policy = gamma_policy
        n = chart.nmodes
        if gamma_mode in ('param', 'gauge'):
            free = torch.ones(n + 1, dtype=torch.bool)
            free[chart.n_alpha] = False             # const B0; corrected below for fx
        else:
            free = torch.ones(n, dtype=torch.bool)
            free[chart.n_alpha] = False
        # B_0=x is NONZERO for the paper's f(x) basis.
        if chart.basis.kind == 'polyt' and chart.basis.seed_name not in ('std', 'raw'):
            free[chart.n_alpha] = True
        if gamma_mode == 'gauge':
            free[-1] = False  # Phi is already in the span: remove exact redundancy.
        self.free = free
        self.n_par = int(free.sum())
        # ridge mask: never regularise gamma (it is a curvature scale, not noise)
        rm = torch.ones(self.n_par, dtype=torch.bool)
        if gamma_mode == 'param':
            rm[-1] = False
        self.reg_mask = rm
        self.t_lo, self.t_hi = chart.t_lo, chart.t_hi

    # ------------------------------------------------------------- helpers
    lagged_gamma = False

    def _design(self, st, which: str) -> torch.Tensor:
        """Design block for the coefficient part, restricted to the free parameters.

        gamma_mode='param'  -> an extra column gamma, filled with Phi (or its t/x
                               derivative); gamma is optimised.
        gamma_mode='cert'    -> gamma = c(theta) is slaved; its derivative enters
                               through the Phi * dgamma term (paper Remark 1/2).
        lagged_gamma=True    -> gamma = c(theta) is still slaved, but its derivative
                               is dropped from the Jacobian (it is updated between
                               Gauss-Newton steps).  Removes the positive feedback
                               gamma <-> sum|theta| that makes the strict l1 bound
                               diverge; the returned point is still certified.
        """
        P = st['parts']
        n = self.chart.nmodes
        phi = {'W': st['Phi'], 'WT': st['Phi_t'], 'WX': st['Phi_x']}[which]
        if self.gamma_mode in ('param', 'gauge'):
            A = torch.empty(n + 1, P[which].shape[1])
            A[:n] = P[which]
            A[n] = phi
            return A[self.free]
        if self.lagged_gamma:
            return P[which][self.free]
        dg = self.chart.cert.grad(st['theta_coef'])
        return (P[which] + dg[:, None] * phi[None, :])[self.free]

    def project(self, theta, t):
        """Clamp the latent coordinates and enforce the one-sided certificate.

        gamma_policy:
          'none' (default) no enforcement; (2.5) is checked on the returned chart.
          'reject'   return None for an infeasible trial (the LM then
                              shrinks the step and retries);
          'none'              no enforcement (used to measure the raw data fit).
        """
        theta = theta.clone()
        t = torch.clamp(t, self.t_lo, self.t_hi)
        if self.gamma_mode == 'gauge' and self.gamma_policy != 'none':
            shortfall = getattr(self,'curvature_floor',self.chart.cert.delta) - self.chart.margin(theta)
            if shortfall > 0:
                if self.gamma_policy == 'reject': return None, t
                # Project the PHYSICAL chart along Phi, whose tt derivative is -1.
                theta[:self.chart.nmodes] += (shortfall+1e-12)*self.chart.phi_coefficients()
            return theta, t
        if self.gamma_mode != 'param' or self.gamma_policy == 'none':
            return theta, t
        lb = self.chart.cert_bound(theta)
        if float(theta[-1]) < lb:
            if self.gamma_policy == 'reject':
                return None, t
            theta[-1] = lb
        return theta, t

    def loss(self, theta, t):
        ev = self.eval(theta, t, need_jac=False)
        r = ev['r']
        return 0.5 * float(r @ r) + (0.5 * float(ev['rc'] @ ev['rc']) if 'rc' in ev else 0.)

    def eval(self, theta, t, need_jac: bool = True) -> dict:
        raise NotImplementedError


def _solve_state(prob, theta, t):
    c = prob.chart
    te = t if prob.with_tau else c.solve_t(prob.x, prob.y, theta, t0=t)[0]
    st = c.eval(prob.x, te, theta)
    st['theta_coef'] = theta[:c.nmodes] if prob.gamma_mode in ('param', 'gauge') else theta
    return st, te


# ------------------------------------------------------------------ Dirichlet
class DirichletBVP(BVP):
    bc = 'dirichlet'

    def __init__(self, chart, x, y, g, lam: float = 0.4, mu: float = 0.0,
                 tau_mode: str = 'joint', gamma_mode: str = 'param',
                 gamma_policy: str = 'none'):
        super().__init__(chart, lam=lam, mu=mu, tau_mode=tau_mode,
                         gamma_mode=gamma_mode, gamma_policy=gamma_policy)
        self.x = torch.as_tensor(x)
        self.y = torch.as_tensor(y)
        self.g = torch.as_tensor(g)
        self.m = int(self.x.numel())

    def eval(self, theta, t, need_jac: bool = True) -> dict:
        st, te = _solve_state(self, theta, t)
        e = st['wt'] + self.y
        r1 = st['w'] + self.y * te - self.g
        out = dict(state=st, gamma=float(st['gamma']), t=te,
                   r1=float((r1 ** 2).mean().sqrt()),
                   r2=float((e ** 2).mean().sqrt()))
        if self.with_tau:
            out['r'] = torch.cat([r1, self.sqrt_lam * e])
            out['j1'] = e
            out['j2'] = self.sqrt_lam * st['wtt']
        else:
            out['r'] = r1
            out['j1'] = torch.zeros(self.m)
            out['j2'] = torch.zeros(self.m)
        if need_jac:
            A = self._design(st, 'W')
            if self.with_tau:
                B = self._design(st, 'WT')
                J = torch.empty(2 * self.m, self.n_par)
                J[:self.m] = A.t()
                J[self.m:] = self.sqrt_lam * B.t()
            else:
                J = A.t()
            out['J'] = J
        return out


# ------------------------------------------------------------------ transport
class TransportBVP(BVP):
    bc = 'transport'

    def __init__(self, chart, x, y, dist_fn, lam: float = 0.4, mu: float = 0.0,
                 tau_mode: str = 'joint', gamma_mode: str = 'param',
                 gamma_policy: str = 'none', loss_norm: str = 'sq',
                 eps: float = 1e-8, corners=None, corner_weight: float = 0.):
        """dist_fn(px, py) -> (d, nx, ny): distance to the target boundary curve and
        the unit normal of the closest point (differentiable in px, py)."""
        super().__init__(chart, lam=lam, mu=mu, tau_mode=tau_mode,
                         gamma_mode=gamma_mode, gamma_policy=gamma_policy)
        self.x = torch.as_tensor(x)
        self.y = torch.as_tensor(y)
        # The additive potential constant has no transport data.
        self.free[0] = False
        self.n_par = int(self.free.sum())
        self.reg_mask = torch.ones(self.n_par,dtype=torch.bool)
        if gamma_mode == 'param': self.reg_mask[-1] = False
        self.dist = dist_fn
        self.m = int(self.x.numel())
        self.loss_norm = loss_norm
        self.eps = float(eps)
        self.corner_weight = float(corner_weight)
        if self.corner_weight < 0: raise ValueError('Negative corner weight')
        self.corners = None if corners is None else torch.as_tensor(corners)
        if self.corner_weight and (not self.with_tau or loss_norm != 'sq'):
            raise ValueError('Corner residuals currently require joint square loss')

    def eval(self, theta, t, need_jac: bool = True) -> dict:
        st, te = _solve_state(self, theta, t)
        e = st['wt'] + self.y
        d, nx, ny = self.dist(st['wx'], te)
        out = dict(state=st, gamma=float(st['gamma']), t=te, d=d,
                   r1=float((d ** 2).mean().sqrt()),
                   r2=float((e ** 2).mean().sqrt()))
        j1 = nx * st['wxt'] + ny
        j2 = self.sqrt_lam * st['wtt']
        r1 = d
        r2 = self.sqrt_lam * e
        if self.loss_norm == 'sumnorm':
            # smooth-L1 loss  sum_i sqrt(d_i^2 + lam e_i^2 + eps^2)  as a sum of
            # squares of  q_i^{-1/4}(d_i, sqrt(lam) e_i),  q_i = d_i^2+lam e_i^2+eps^2
            q = (d * d + self.lam * e * e + self.eps ** 2)
            w = q.pow(-0.25)
            r1 = w * d
            r2 = w * self.sqrt_lam * e
            j1 = w * j1
            j2 = w * j2
        else:
            w = None
        if self.with_tau:
            out['r'] = torch.cat([r1, r2])
            out['j1'] = j1
            out['j2'] = j2
        else:
            out['r'] = r1
            out['j1'] = torch.zeros(self.m)
            out['j2'] = torch.zeros(self.m)
        if need_jac:
            A = self._design(st, 'WX')
            B = self._design(st, 'WT')
            if self.with_tau:
                J = torch.empty(2 * self.m, self.n_par)
                J[:self.m] = ((w[:, None] if w is not None else 1.0)
                              * nx[:, None] * A.t())
                J[self.m:] = ((w[:, None] if w is not None else 1.0)
                              * self.sqrt_lam * B.t())
            else:
                # t = t(theta) through omega_t + y = 0: extra -j1 * w_tt^{-1} B term
                J = (nx[None, :] * A - (j1 / st['wtt'])[None, :] * B).t()
                if w is not None:
                    J = w[:, None] * J
            out['J'] = J
        if self.corner_weight and self.corners is not None:
            # Piecewise smooth min over EXISTING joint image samples, paper 3.3.
            # Indices are frozen in this Jacobian, recomputed for each trial loss.
            image = torch.stack([st['wx'], te], dim=1)
            k = ((image[None] - self.corners[:, None]) ** 2).sum(2).argmin(1)
            sw = math.sqrt(self.corner_weight)
            out['rc'] = sw * torch.cat([st['wx'][k]-self.corners[:,0], te[k]-self.corners[:,1]])
            out['ic'] = torch.cat([k,k])
            out['jc'] = sw * torch.cat([st['wxt'][k], torch.ones_like(te[k])])
            if need_jac:
                Ac = self._design(st, 'WX')[:,k].t()
                out['Jc'] = sw * torch.cat([Ac, torch.zeros_like(Ac)])
        return out
