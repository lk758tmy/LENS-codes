"""Schur-complement Levenberg-Marquardt (paper eqs. 3.13-3.14).

The latent coordinates tau in R^m are eliminated blockwise: J_tau^T J_tau is
diagonal, so each trial costs one P x P solve (P = 2n - 1 unknowns) plus
O(P m) products.  Acceptance uses the standard gain ratio with threshold 1e-4.
"""
from __future__ import annotations

import torch


def _solve_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(b[:, None], L)[:, 0]
    except Exception:                       # pragma: no cover - numerical fallback
        return torch.linalg.lstsq(A, b[:, None]).solution[:, 0]


def solve(prob, theta0, t0, rounds: int = 200, kappa0: float = 1e-3,
          kappa_max: float = 1e12, ftol: float = 0.0, trial_limit: int = 16,
          log_every: int = 0, callback=None, verbose: bool = False, observer=None) -> dict:
    """Minimise 0.5||r(theta,tau)||^2 + mu/2 ||theta||^2  over (theta, tau)."""
    theta, t = prob.project(theta0, t0)
    if theta is None:
        raise ValueError('Infeasible initial chart for reject policy')
    kappa = float(kappa0)
    ev = prob.eval(theta, t)
    hist = []
    free = prob.free
    mu = prob.mu
    reg = getattr(prob, 'reg_mask', torch.ones(prob.n_par, dtype=torch.bool))
    eye = torch.eye(prob.n_par)

    def loss_of(ev_, th):
        L = 0.5 * float(ev_['r'] @ ev_['r'])
        if 'rc' in ev_: L += 0.5 * float(ev_['rc'] @ ev_['rc'])
        if mu > 0:
            tf = th[free][reg]
            L += 0.5 * mu * float(tf @ tf)
        return L

    L0 = loss_of(ev, theta)
    for it in range(1, rounds + 1):
        r = ev['r']
        J = ev['J']
        m = prob.m
        A = J.t() @ J
        g = J.t() @ r
        if mu > 0:
            A = A + mu * torch.diag(reg.to(A.dtype))
            g = g + mu * (theta[free] * reg)
        if prob.with_tau:
            j1, j2 = ev['j1'], ev['j2']
            M = J[:m].t() * j1[None, :] + J[m:].t() * j2[None, :]      # (P, m)
            D = j1 * j1 + j2 * j2
            gt = j1 * r[:m] + j2 * r[m:]
        else:                                   # tau eliminated: no latent block
            M = torch.zeros(prob.n_par, m)
            D = torch.zeros(m)
            gt = torch.zeros(m)

        if 'rc' in ev:
            # Each extra row depends on one existing latent coordinate. Repeated
            # nearest indices must ADD, so the latent normal block stays diagonal.
            Jc, rc, jc, ic = ev['Jc'], ev['rc'], ev['jc'], ev['ic']
            A = A + Jc.t() @ Jc
            g = g + Jc.t() @ rc
            M = M.index_add(1, ic, Jc.t() * jc[None,:])
            D = D.index_add(0, ic, jc*jc)
            gt = gt.index_add(0, ic, jc*rc)

        accepted = False
        # Marquardt column scaling: the monomial modes are nearly dependent on the
        # boundary curve (cond(J) ~ 1e10 at K = 12), so the damping must be applied
        # in units of each column norm, not in raw parameter units.
        S = torch.sqrt(torch.diagonal(A).clamp(min=1e-30))
        Si = 1.0 / S
        A_s = A * Si[:, None] * Si[None, :]
        g_s = g * Si
        M_s = M * Si[:, None]
        for _trial in range(trial_limit):
            inv = 1.0 / (D + kappa)
            Sch = A_s + kappa * eye - (M_s * inv[None, :]) @ M_s.t()
            rhs = -g_s + M_s @ (gt * inv)
            if getattr(prob,'enforce_linear_curvature',False):
                from .constrained_lm import solve_qp
                C=prob.chart.cert.S[free[:-1]].t()
                b=theta[-1]-getattr(prob,'curvature_floor',prob.chart.cert.delta)-theta[:-1]@prob.chart.cert.S
                zz,prob.qp_info=solve_qp(Sch,rhs,C*Si[None,:],b)
                dth=zz*Si
            else:
                dth = _solve_spd(Sch, rhs) * Si
            t_new = torch.clamp(t + (-(gt + M.t() @ dth) * inv), prob.t_lo, prob.t_hi)
            dtau = t_new - t
            if not (torch.isfinite(dth).all() and torch.isfinite(t_new).all()):
                kappa = min(kappa * 4.0, kappa_max)
                continue
            th_new = theta.clone()
            th_new[free] = theta[free] + dth
            th_new, t_new = prob.project(th_new, t_new)
            if th_new is None:                  # infeasible trial ('reject' policy)
                kappa = min(kappa * 4.0, kappa_max)
                continue
            # The model must use the ACTUAL projected step.
            dth = th_new[free] - theta[free]
            dtau = t_new - t
            pred = -(float(g @ dth) + float(gt @ dtau)) \
                - 0.5 * (float(dth @ (A @ dth)) + 2.0 * float(dth @ (M @ dtau))
                         + float(dtau @ (D * dtau)))
            ev_new = prob.eval(th_new, t_new, need_jac=False)
            L1 = loss_of(ev_new, th_new)
            rho = (L0 - L1) / pred if pred > 0 else -1.0
            if L1 < L0 and rho > 1e-4:
                theta, t = th_new, t_new
                ev = prob.eval(theta, t)
                kappa = max(kappa / 4.0, 1e-14)
                accepted = True
                break
            kappa = min(kappa * 4.0, kappa_max)
        else:
            if kappa >= kappa_max:             # no descent direction left
                break

        L0 = loss_of(ev, theta)
        rec = dict(rnd=it, loss=L0, kappa=kappa, accepted=accepted)
        if callback is not None:
            rec.update(callback(it, theta, t, ev) or {})
        hist.append(rec)
        if observer is not None: observer(it, theta, t, rec)
        if verbose and (log_every and it % log_every == 0):
            print(f'    it {it:4d}  L={L0:.3e}  kappa={kappa:.1e}  {rec.get("info", "")}')
        if ftol > 0 and L0 < ftol:
            break
    return dict(theta=theta, t=t, kappa=kappa, hist=hist, iters=len(hist))
