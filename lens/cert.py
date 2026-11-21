"""Convexity certificate  (paper eq. 2.5).

The chart requires  omega_tt = H_tt - gamma <= -delta < 0, i.e.

        gamma >= delta + max_box H_tt .

Three admissible bounds for max H_tt (they differ only in how the supremum over the
prior box B = [x_lo,x_hi] x [t_lo,t_hi] is taken):

  'box'   (1+iota) * max_{grid in B} H_tt + delta     <- paper (2.5), default
  'sum'   delta + sum_m |theta_m| sup_{grid} |d^2_t w_m|      (l^1, conservative)
  'disc'  delta + sum_m |theta_m| * (k(k-1)/R^2)              (analytic, |zeta|<=1)

Usage
    bound(theta)   right-hand side of (2.5)
    max_htt(theta) sampled sup of H_tt (diagnostics)
"""
from __future__ import annotations

import torch


class Certificate:
    def __init__(self, S: torch.Tensor, mode: str = 'box', delta: float = 0.1,
                 iota: float = 0.02, W_disc: torch.Tensor | None = None):
        """
        S      : (nmodes, ngrid)  d^2_t w_m at the fixed certificate grid points.
        mode   : 'box' | 'sum' | 'disc'
        delta  : margin, the certificate guarantees omega_tt <= -delta
        iota   : inflation factor of (2.5)
        W_disc : (nmodes,) analytic per-mode bound used by mode 'disc'.
        """
        if mode not in ('box', 'sum', 'disc'):
            raise KeyError(mode)
        self.S = S
        self.mode = mode
        self.delta = float(delta)
        self.iota = float(iota)
        self.W = S.abs().amax(dim=1)
        self.W_disc = torch.ones_like(self.W) if W_disc is None else W_disc

    # ---------------------------------------------------------------- bound
    def bound(self, theta: torch.Tensor) -> torch.Tensor:
        """Right-hand side of (2.5) for the coefficient vector theta."""
        if self.mode == 'box':
            return self.delta + (1.0 + self.iota) * (theta @ self.S).max()
        if self.mode == 'sum':
            return self.delta + (theta.abs() * self.W).sum()
        return self.delta + (theta.abs() * self.W_disc).sum()

    # ------------------------------------------------------------ gradient
    # Only used by the literal ('cert') variant of the chart, in which gamma is a
    # function of theta; see lens.chart for why 'param' is the default.
    def grad(self, theta: torch.Tensor) -> torch.Tensor:
        if self.mode == 'box':
            j = int(torch.argmax(theta @ self.S))
            return (1.0 + self.iota) * self.S[:, j]
        if self.mode == 'sum':
            return torch.sign(theta) * self.W
        return torch.sign(theta) * self.W_disc

    # ------------------------------------------------------------ diagnostics
    def max_htt(self, theta: torch.Tensor) -> float:
        return float((theta @ self.S).max())
