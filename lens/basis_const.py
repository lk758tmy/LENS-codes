"""Generator (3.1) for a constant source term f == c.

        omega(x,t) = (gamma/2)(c (x-xc)^2 - (t-tc)^2)
                     + Re sum_{k=0}^{K} (alpha_k + i beta_k) phi_k(zeta),
        zeta        = (sqrt(c) x + i t - z0) / R .

Two re-parametrisations are used, both *diagonal* changes of basis (identical span):

* zeta = (z - z0)/R with z0 at the box centre keeps |zeta| <= 1 on the prior box;
* each mode is divided by  sup_box max(|d_t phi_k|, |d^2_t phi_k|) ~ k^2/R^2
  ('d2').  Without it the certificate sensitivity d gamma / d alpha_k grows like
  k^2 while the mode value stays O(1), which de-conditions the LM for K > 6.

The base quadratic is expanded at the same centre (xc,tc) as the polynomial part;
both are annihilated by L_c = d_xx + c d_tt, so the identity L_c omega = 0 holds.

Mode ordering (must match theta = [alpha | beta]):
    w_k       = Re zeta^k     (k = 0..K)      <- alpha_k
    w_{K+1+k} = -Im zeta^k    (k = 0..K)      <- beta_k
so that  Re sum (alpha_k + i beta_k) zeta^k = sum_k alpha_k w_k + beta_k w_{K+1+k}.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .basis import Basis


def _shift(A: torch.Tensor, m: int) -> torch.Tensor:
    """out[k] = A[k-m] for k >= m, else 0  (A is (K+1, N))."""
    out = torch.zeros_like(A)
    if m == 0:
        out[:] = A
    elif m < A.shape[0]:
        out[m:] = A[:-m]
    return out


class ConstBasis(Basis):
    kind = 'complex'

    def __init__(self, c: float = 1.0, K: int = 12, box=None,
                 z0=None, R: float | None = None, ngrid: int = 61,
                 mode_norm: str = 'd2'):
        """
        box : (x_lo, x_hi, t_lo, t_hi)  prior box holding Omega' in (x,t) space.
        z0  : (Re, Im) of the expansion centre in the z = sqrt(c) x + i t plane
              (default: centre of the box).
        R   : radius used to normalise zeta (default: 1.02 * box half-diagonal).
        mode_norm : 'd2' (default) or 'none'; see the module docstring.
        """
        assert box is not None, 'ConstBasis needs the (x,t) prior box'
        self.c = float(c)
        self.K = int(K)
        self.n_alpha = self.K + 1
        self.nmodes = 2 * self.n_alpha
        self.box = tuple(float(v) for v in box)
        self.ngrid = int(ngrid)
        self.sq = math.sqrt(self.c)

        x_lo, x_hi, t_lo, t_hi = self.box
        if z0 is None:
            z0 = (self.sq * 0.5 * (x_lo + x_hi), 0.5 * (t_lo + t_hi))
        self.z0 = (float(z0[0]), float(z0[1]))
        self.xc = self.z0[0] / self.sq
        self.tc = self.z0[1]
        if R is None:
            R = 1.02 * math.hypot(self.sq * 0.5 * (x_hi - x_lo),
                                  0.5 * (t_hi - t_lo))
        self.R = float(R)
        self.mode_norm = mode_norm
        self.g = torch.ones(self.nmodes)
        if mode_norm == 'd2':
            self.g = self._mode_scales(ngrid)

    # ---------------------------------------------------------------- powers
    def _powers(self, x: torch.Tensor, t: torch.Tensor):
        """Re/Im of zeta^k by the recurrence (no in-place ops: the graph must stay
        differentiable for the PDE audit)."""
        xi = (self.sq * x.reshape(-1) - self.z0[0]) / self.R
        et = (t.reshape(-1) - self.z0[1]) / self.R
        K = self.K
        U = [torch.ones_like(xi)]
        V = [torch.zeros_like(et)]
        if K >= 1:
            U.append(xi)
            V.append(et)
        for k in range(1, K):
            U.append(xi * U[k] - et * V[k])
            V.append(xi * V[k] + et * U[k])
        return torch.stack(U), torch.stack(V)

    def _parts_raw(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        shape = x.shape
        U, V = self._powers(x, t)
        K, R, sq = self.K, self.R, self.sq
        kk = torch.arange(K + 1, dtype=x.dtype).reshape(-1, 1)

        dUdt = -(kk / R) * _shift(V, 1)
        dVdt = (kk / R) * _shift(U, 1)
        d2U = -(kk * (kk - 1) / R ** 2) * _shift(U, 2)
        d2V = -(kk * (kk - 1) / R ** 2) * _shift(V, 2)
        dUdx = (sq / R) * kk * _shift(U, 1)
        dVdx = (sq / R) * kk * _shift(V, 1)
        d2Uxt = -(sq / R ** 2) * kk * (kk - 1) * _shift(V, 2)
        d2Vxt = (sq / R ** 2) * kk * (kk - 1) * _shift(U, 2)
        return dict(W=torch.cat([U, -V]),
                    WX=torch.cat([dUdx, -dVdx]),
                    WT=torch.cat([dUdt, -dVdt]),
                    WTT=torch.cat([d2U, -d2V]),
                    WXT=torch.cat([d2Uxt, -d2Vxt]))

    def parts(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        shape = x.shape
        raw = self._parts_raw(x, t)
        inv = (1.0 / self.g).reshape(-1, 1)
        return {k: (v * inv).reshape((self.nmodes,) + shape) for k, v in raw.items()}

    def phi(self, x: torch.Tensor, t: torch.Tensor):
        """Base term (gamma/2)(c (x-xc)^2 - (t-tc)^2), expanded at the *same* centre
        as the polynomial part (a translation only moves a polynomial inside the
        span of degree <= K for K >= 2, so the family is unchanged)."""
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
        xc, tc = self.xc, self.tc
        return (0.5 * (self.c * (x - xc) ** 2 - (t - tc) ** 2),
                self.c * (x - xc), -(t - tc))

    # ---------------------------------------------------------- certificate
    def _mode_scales(self, ngrid: int):
        """Per-mode scale  max(1, sup_box max(|phi|, |d_t phi|, |d^2_t phi|) ).

        The '1' floor matters: the modes Re zeta and 1 have no t-dependence at all,
        so their t-derivatives vanish and a pure t-derivative scale would blow them
        up.  With the floor, low modes keep unit scale and high modes are divided by
        their k^2/R^2 growth.
        """
        X, T = self.cert_grid(ngrid)
        with torch.no_grad():
            P = self._parts_raw(X, T)
            s = torch.maximum(P['W'].abs().amax(dim=1),
                              torch.maximum(P['WT'].abs().amax(dim=1),
                                            P['WTT'].abs().amax(dim=1)))
        return s.clamp(min=1.0)

    def W_disc(self) -> torch.Tensor:
        """Analytic sup over |zeta|<=1 of |d^2_t w_m| (mode-normalised)."""
        k = torch.arange(self.n_alpha, dtype=torch.float64)
        w = k * (k - 1) / self.R ** 2
        w = torch.cat([w, w])
        return w / self.g

    def seed(self, gamma_target: float, delta: float, iota: float = 0.02):
        """theta0 with only alpha_2 switched on, chosen so that gamma(theta0) =
        gamma_target under the 'box' certificate."""
        assert self.K >= 2, 'K >= 2 required to seed the reference curvature'
        H_target = (float(gamma_target) - float(delta)) / (1.0 + float(iota))
        S = self.cert_S(self.ngrid)
        col = S[2]
        theta = torch.zeros(self.nmodes)
        theta[2] = H_target / float(col.max())
        return theta

    def describe(self) -> dict:
        return dict(kind=self.kind, K=self.K, c=self.c, R=self.R,
                    z0=list(self.z0), box=list(self.box), nmodes=self.nmodes,
                    mode_norm=self.mode_norm)


def default_box(region, t_pad: float = 0.5, x_pad: float = 0.0):
    """Naive prior box; prefer :func:`lens.init.dirichlet_init` (data driven)."""
    x0, x1, y0, y1 = region.bbox
    span = max(x1 - x0, y1 - y0)
    return (x0 - x_pad, x1 + x_pad, y0 - t_pad * span, y1 + t_pad * span)
