"""Separable generator for a source that depends on x only:  det D^2 u = f(x).

Construction (the f(x) analogue of (3.1); see EXPERIMENTS_DESIGN and the old
exp10 notes):

        omega(x,t) = gamma * Phi(x,t) + sum_m theta_m w_m(x,t),
        Phi(x,t)   = F(x) - (t-tc)^2/2,      F'' = f,   F(x_lo) = F'(x_lo) = 0,

        w_m(x,t)   = sum_{j=0}^{J} a_j(x) (t-tc)^j ,
        a_j''      = -(j+2)(j+1) f(x) a_{j+2}                (j = J-2 .. 0),

i.e. L_f w = w_xx + f w_tt = 0 term by term, so L_f omega = 0 and
det D^2 u = -omega_xx/omega_tt = f(x) identically, as in the constant-source case.

Only the two *top* coefficients a_J, a_{J-1} are free (they are the seeds); the
rest follow from the recursion.  Everything is done in Chebyshev coefficients on
the prior box, so that f is multiplied and integrated exactly (up to the
truncation of its own Chebyshev interpolant).

Seeds (the 'seed' argument; exp10, Table 4.4):

  'std'    A: (1/J!, 0)      B: (0, 1/(J-1)!)          -- original, poor
  'paper'  A: (1/J!, 0)      B: (xi, 0)              -- independent families, default
  'onex'   A: (1/J!, xi/J!)  B: (xi/(J-1)!, 1/(J-1)!)  -- mixing xi
  'raw'    A: (1, 0)         B: (0, 1)                 -- no factorial scaling
  'rawx'   A: (1, xi)        B: (xi, 1)

The factorial normalisation must be kept: without it the certificate weights grow
by three orders of magnitude and the fit degrades (exp10, Table 4.4).

Mode ordering (must match theta = [alpha | beta]):
    w_m (m = k)          seeds 'A' of order k        <- alpha_k
    w_{K+1+k}            seeds 'B' of order k        <- beta_k
For the paper seeds, B_0=xi is nonzero and must be optimized. There are 2K+2
coefficients. The std/raw seeds have a zero B_0 and duplicate lower A modes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from numpy.polynomial import chebyshev as C

from .basis import Basis

_TOL = 1e-16


# ------------------------------------------------------------------ helpers
def _trim(c, tol: float = _TOL) -> np.ndarray:
    c = np.atleast_1d(np.asarray(c, float)).ravel()
    n = len(c)
    while n > 1 and abs(c[n - 1]) < tol:
        n -= 1
    return c[:n].copy()


def _trimrel(c, tol: float) -> np.ndarray:
    """Drop the tail below ``tol`` times the polynomial's OWN max (relative).

    An absolute threshold must not be used on the generated modes: with the
    factorial seeds of the recursion the coefficients of high-order modes are
    O(1/J!), and absolute truncation would zero whole polynomials (which silently
    breaks the identity w_xx + f w_tt = 0 that the recursion is built on).
    """
    c = np.atleast_1d(np.asarray(c, float)).ravel()
    if c.size == 0:
        return np.zeros(1)
    s = float(np.abs(c).max())
    if s == 0.0:
        return np.zeros(1)
    idx = np.where(np.abs(c) > tol * s)[0]
    if idx.size == 0:
        return np.zeros(1)
    return c[:idx[-1] + 1].copy()


def _antider(c) -> np.ndarray:
    """Antiderivative with zero constant term, gauged at xi = -1 (i.e. x = x_lo).

    Gauging at x_lo matters: with the gauge at the box centre the base term Phi
    differs from any analytic F = int int f by an affine function of x, which is
    representable but *badly scaled* (it is the near-null direction spanned by
    {Phi, A_2}).  Gauging at x_lo makes Phi equal to the F of ``lens.sources``.
    """
    out = np.atleast_1d(C.chebint(np.asarray(c, float).ravel()))
    out[0] -= float(C.chebval(-1.0, out))
    return out


_ONE = np.array([1.0])
_XI = np.array([0.0, 1.0])


def _seeds(J: int, seed: str):
    """(A seed, B seed) of order J, each a pair (top, next) in Chebyshev xi."""
    fJ = 1.0 / math.factorial(J)
    fJm = (1.0 / math.factorial(J - 1)) if J >= 1 else 0.0
    z = np.zeros(1)
    if seed == 'std':
        return ((fJ * _ONE, z), (z, fJm * _ONE))
    if seed == 'paper':        # the two seeds of the paper: (1,0) and (x,0)
        return ((fJ * _ONE, z), (_XI, z))
    if seed == 'onex':
        return ((fJ * _ONE, fJ * _XI), (fJm * _XI, fJm * _ONE))
    if seed == 'raw':
        return ((_ONE, z), (z, _ONE))
    if seed == 'rawx':
        return ((_ONE, _XI), (_XI, _ONE))
    raise ValueError(f'unknown seed {seed!r}')


class FxBasis(Basis):
    kind = 'polyt'

    def __init__(self, source, box, K: int = 16, seed: str = 'paper',
                 t_center: float | None = None, nfit: int = 80,
                 tol: float = 1e-15, mode_norm: str = 'd2'):
        """
        source : callable f(x) (vectorised, positive)
        box    : (x_lo, x_hi, t_lo, t_hi); the certificate box
        K      : highest t-degree of a mode (nmodes = 2(K+1))
        seed   : see the module docstring
        nfit   : degree of the Chebyshev interpolant of f
        """
        self.source = source
        self.K = int(K)
        self.n_alpha = self.K + 1
        self.nmodes = 2 * self.n_alpha
        self.box = tuple(float(v) for v in box)
        self.seed_name = seed
        self.mode_norm = mode_norm
        x_lo, x_hi, t_lo, t_hi = self.box
        self.x_lo, self.x_hi = x_lo, x_hi
        self._xo = 0.5 * (x_lo + x_hi)
        self._xs = 0.5 * (x_hi - x_lo)
        self.tc = 0.5 * (t_lo + t_hi) if t_center is None else float(t_center)
        self.tol = float(tol)

        self.f_coef = self._fit_f(source, nfit, tol)
        self._build_modes()
        # Phi = F(x) - (t-tc)^2/2, with F'' = f (Chebyshev-truncated f)
        xs = self._xs
        self.cF1 = _trimrel(xs * _antider(self.f_coef), tol)          # F'
        self.cF = _trimrel(xs ** 2 * _antider(_antider(self.f_coef)), tol)   # F

        self.g = torch.ones(self.nmodes, dtype=torch.float64)
        if mode_norm == 'd2':
            self.g = self._mode_scales()

    # ------------------------------------------------------------- fitting f
    def _fit_f(self, source, nfit: int, tol: float) -> np.ndarray:
        """Chebyshev interpolant of f on the box, trimmed to ``tol`` relative."""
        N = int(nfit) + 1
        xi = np.cos(np.pi * (np.arange(N) + 0.5) / N)              # chebpts1
        x = self._xo + self._xs * xi
        fx = np.asarray(source(x), float)
        c = C.chebfit(xi, fx, N - 1)
        scale = max(np.abs(c).max(), 1e-300)
        c = np.where(np.abs(c) < tol * scale, 0.0, c)
        return _trim(c, 0.0)

    # ------------------------------------------------------------ mode build
    def _family(self, J: int, sJ, sJm):
        """Coefficient arrays a_j (j = 0..J) of one mode, from its two seeds."""
        a = [np.zeros(1)] * (J + 1)
        a[J] = np.atleast_1d(np.asarray(sJ, float)).ravel()
        if J >= 1:
            a[J - 1] = np.atleast_1d(np.asarray(sJm, float)).ravel()
        for j in range(J - 2, -1, -1):
            g = C.chebmul(self.f_coef, a[j + 2])
            a[j] = _trimrel(-(j + 2) * (j + 1) * self._xs ** 2
                            * _antider(_antider(g)), self.tol)
        return a

    def _build_modes(self):
        M, xs, tol = self.K, self._xs, self.tol
        rows_a, rows_b = [], []
        for fam in ('A', 'B'):
            for J in range(M + 1):
                sA, sB = _seeds(J, self.seed_name)
                st = sA if fam == 'A' else sB
                if fam == 'B' and J < 1 and self.seed_name in ('std', 'raw'):
                    rows_a.append([np.zeros(1)])
                    rows_b.append([np.zeros(1)])
                    continue
                a = self._family(J, st[0], st[1])
                b = [np.zeros(1)] * (J + 1)
                b[J] = (_trimrel(C.chebder(a[J]) / xs, tol)
                        if len(a[J]) > 1 else np.zeros(1))
                if J >= 1:
                    b[J - 1] = (_trimrel(C.chebder(a[J - 1]) / xs, tol)
                                if len(a[J - 1]) > 1 else np.zeros(1))
                for j in range(J - 2, -1, -1):
                    g = C.chebmul(self.f_coef, a[j + 2])
                    b[j] = _trimrel(-(j + 2) * (j + 1) * xs * _antider(g), tol)
                rows_a.append(a)
                rows_b.append(b)

        L = 2
        for a, b in zip(rows_a, rows_b):
            for c in a + b:
                L = max(L, len(c))
        self.L = L
        # A3[m, j, k] = Chebyshev coefficient of xi^k in a_j(x) of mode m.
        # A single row per mode is NOT enough: w_m = sum_j a_j(x) tau^j is a
        # two-index tensor (degree in tau AND degree in xi).
        self.A3 = np.zeros((self.nmodes, self.K + 1, L))
        self.B3 = np.zeros((self.nmodes, self.K + 1, L))
        for m, (a, b) in enumerate(zip(rows_a, rows_b)):
            for j, c in enumerate(a):
                self.A3[m, j, :len(c)] = c
            for j, c in enumerate(b):
                self.B3[m, j, :len(c)] = c
        self.A3_t = torch.as_tensor(self.A3)
        self.B3_t = torch.as_tensor(self.B3)
        self._jf = torch.arange(self.K + 1, dtype=torch.float64)

    # ----------------------------------------------------------- evaluation
    def _chebmat(self, xi: torch.Tensor) -> torch.Tensor:
        """(L, N) Chebyshev polynomials T_j(xi) -- built functionally (no in-place
        writes: the chart is differentiated through by the PDE audit)."""
        L = self.L
        rows = [torch.ones_like(xi)]
        if L > 1:
            rows.append(xi)
        for j in range(1, L - 1):
            rows.append(2.0 * xi * rows[j] - rows[j - 1])
        return torch.stack(rows[:L], 0)

    def _raw(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        xi = (x.reshape(-1) - self._xo) / self._xs
        tau = t.reshape(-1) - self.tc
        T = self._chebmat(xi)                                     # (L, N)
        # powers of tau and their t-derivatives, shape (K+1, N); built with
        # out-of-place ops only, because the audit differentiates through this
        K1 = self.K + 1
        pw = [torch.ones_like(tau)]
        d1 = [torch.zeros_like(tau)]
        d2 = [torch.zeros_like(tau)]
        for j in range(1, K1):
            pw.append(pw[j - 1] * tau)
            d1.append(j * pw[j - 1])
            d2.append(j * (j - 1) * pw[j - 2] if j >= 2 else torch.zeros_like(tau))
        P, D1, D2 = torch.stack(pw, 0), torch.stack(d1, 0), torch.stack(d2, 0)
        Va = torch.einsum('mjk,kn->mjn', self.A3_t, T)             # (m, K+1, N)
        Vb = torch.einsum('mjk,kn->mjn', self.B3_t, T)
        return dict(W=torch.einsum('mjn,jn->mn', Va, P),
                    WX=torch.einsum('mjn,jn->mn', Vb, P),
                    WT=torch.einsum('mjn,jn->mn', Va, D1),
                    WTT=torch.einsum('mjn,jn->mn', Va, D2),
                    WXT=torch.einsum('mjn,jn->mn', Vb, D1))

    def parts(self, x, t) -> dict:
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
        t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)
        x = x.to(torch.float64)
        t = t.to(torch.float64)
        shape = x.shape
        raw = self._raw(x, t)
        inv = (1.0 / self.g).reshape(-1, 1)
        return {k: (v * inv).reshape((self.nmodes,) + shape) for k, v in raw.items()}

    def _chebval(self, coef: np.ndarray, x: torch.Tensor) -> torch.Tensor:
        xi = (x.to(torch.float64) - self._xo) / self._xs
        T = self._chebmat(xi)
        c = torch.zeros(self.L, dtype=torch.float64)
        c[:len(coef)] = torch.as_tensor(coef)
        return c @ T

    def phi(self, x, t):
        """Phi = F(x) - (t-tc)^2/2, with Phi_tt == -1."""
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float64)
        t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=torch.float64)
        F = self._chebval(self.cF, x).reshape(x.shape)
        F1 = self._chebval(self.cF1, x).reshape(x.shape)
        return F - 0.5 * (t - self.tc) ** 2, F1, -(t - self.tc)

    # ---------------------------------------------------------- certificate
    def _mode_scales(self) -> torch.Tensor:
        """Per-mode scale max(1, sup_box max(|w|, |w_t|, |w_tt|)) (pre-conditioner)."""
        X, T = self.cert_grid(61)
        with torch.no_grad():
            P = self._raw(X, T)
            s = torch.maximum(P['W'].abs().amax(dim=1),
                              torch.maximum(P['WT'].abs().amax(dim=1),
                                            P['WTT'].abs().amax(dim=1)))
        return s.clamp(min=1.0)

    def W_disc(self) -> torch.Tensor:
        """Sampled per-mode sup of |d^2_t w_m| (only used by the 'disc' mode)."""
        with torch.no_grad():
            X, T = self.cert_grid(81)
            return (self._raw(X, T)['WTT'].abs().amax(dim=1)) / self.g

    def seed(self, gamma_target: float, delta: float, iota: float = 0.02):
        """theta0 with a single mode switched on so that the 'box' bound is hit."""
        H_target = (float(gamma_target) - float(delta)) / (1.0 + float(iota))
        S = self.cert_S(61)
        j = int(torch.argmax(S.abs().amax(dim=1)))
        theta = torch.zeros(self.nmodes, dtype=torch.float64)
        theta[j] = H_target / float(S[j].max())
        return theta

    def describe(self) -> dict:
        return dict(kind=self.kind, K=self.K, nmodes=self.nmodes,
                    seed=self.seed_name, box=list(self.box), tc=self.tc,
                    L=int(self.L), mode_norm=self.mode_norm)
