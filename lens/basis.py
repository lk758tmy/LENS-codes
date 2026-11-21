"""Abstract generator protocol shared by the f == const and f = f(x) charts.

A basis provides, for the coefficient vector

        theta = [alpha_0 .. alpha_K | beta_0 .. beta_K]          (nmodes = 2(K+1))

the coefficient part of the generator

        omega(x,t) = gamma * Phi(x,t) + sum_m theta_m w_m(x,t)

with  Phi_tt == -1  and  L_f Phi = 0  (so that  L_f omega = 0  for every theta),
together with its first and second derivatives.  The convexity constant gamma is
supplied by :class:`lens.cert.Certificate` and is *not* part of theta.

The partial-Legendre transform then reads

        y = -omega_t(x,t),     u = omega(x,t) + y t,

and `res` only ever needs (w, w_x, w_t, w_tt, w_xt) plus Phi.
"""
from __future__ import annotations

import torch


class Basis:
    kind = 'abstract'
    nmodes: int
    n_alpha: int

    # ---- coefficient functions ------------------------------------------
    def parts(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """Return dict of (nmodes, N) tensors W, WX, WT, WTT, WXT."""
        raise NotImplementedError

    def phi(self, x: torch.Tensor, t: torch.Tensor):
        """Base term Phi and its derivatives (Phi, Phi_x, Phi_t); Phi_tt == -1."""
        raise NotImplementedError

    # ---- certificate ------------------------------------------------------
    def cert_grid(self, ngrid: int = 61):
        """(x_m, t_m) of the prior box, as meshgrid-flattened torch tensors."""
        x_lo, x_hi, t_lo, t_hi = self.box
        gx = torch.linspace(x_lo, x_hi, ngrid)
        gt = torch.linspace(t_lo, t_hi, ngrid)
        X, T = torch.meshgrid(gx, gt, indexing='ij')
        return X.reshape(-1), T.reshape(-1)

    def cert_S(self, ngrid: int = 61) -> torch.Tensor:
        """d^2_t w_m on the certificate grid, shape (nmodes, ngrid^2)."""
        X, T = self.cert_grid(ngrid)
        with torch.no_grad():
            return self.parts(X, T)['WTT'].clone()

    def seed(self, gamma_target: float, delta: float, iota: float = 0.02):
        """theta0 whose certificate equals ``gamma_target`` (zeros elsewhere)."""
        raise NotImplementedError

    # ---- misc -------------------------------------------------------------
    def describe(self) -> dict:
        return dict(kind=self.kind, K=getattr(self, 'K', None),
                    nmodes=int(self.nmodes))


def as_tensor(x, dtype=torch.float64):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype)
