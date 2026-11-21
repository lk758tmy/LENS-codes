"""Exact solutions used as manufactured data.

Two families:

RadialTruth      det D^2 u = 1, non-polynomial (branch points), NOT in the span of
                 any of our generators.  This is the reference solution of S4.2.

SeparableTruth   det D^2 u = f(x) for an arbitrary analytic source, of the form
                 u = G(x)/alpha + (alpha/2)(y-y0)^2.  This is the general
                 "translation invariant in y" solution and is used for the f(x) experiments of S4.3. Other, coupled
                 solution families exist; this benchmark is in-span.

Both return numpy arrays via ``u, ux, uy = truth(x, y)``.
"""
from __future__ import annotations

import math

import numpy as np


class RadialTruth:
    """u = 1/2 [ r sqrt(r^2+A) + A asinh(r/sqrt(A)) ],  r = |(x,y)-(cx,cy)|.

    det D^2 u == 1 and u is strictly convex for every centre outside the domain;
    the centre (0,-1) puts the branch point outside [0,1]^2.
    """

    name = 'radial'

    def __init__(self, cx: float = 0.0, cy: float = -1.0, A: float = 1.0):
        self.cx, self.cy, self.A = float(cx), float(cy), float(A)

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        dx = x - self.cx
        dy = y - self.cy
        r = np.sqrt(dx * dx + dy * dy + 1e-300)
        s = np.sqrt(r * r + self.A)
        u = 0.5 * (r * s + self.A * np.arcsinh(r / math.sqrt(self.A)))
        ux = s * dx / r
        uy = s * dy / r
        return u, ux, uy

    def label(self) -> str:
        return f'radial(c=({self.cx:g},{self.cy:g}),A={self.A:g})'


class ParaboloidTruth:
    """u = (sqrt(c)/2)((x-y0)^2 + (y-y0)^2): det D^2 u = c, u_yy == sqrt(c) constant.

    This is the truth used by the old experiment 3 ("analytic paraboloid"); with the
    base quadratic of (3.1) and delta = 1/sqrt(c) it is an EXACT member of the chart
    with theta = 0, which is why the strictly certified runs of exp3 converged.
    """

    name = 'paraboloid'

    def __init__(self, c: float = 1.0, x0: float = 0.0, y0: float = 0.0):
        self.c, self.x0, self.y0 = float(c), float(x0), float(y0)

    def __call__(self, x, y):
        s = math.sqrt(self.c)
        u = 0.5 * s * ((np.asarray(x, float) - self.x0) ** 2
                       + (np.asarray(y, float) - self.y0) ** 2)
        return (u, s * (np.asarray(x, float) - self.x0),
                s * (np.asarray(y, float) - self.y0))

    def label(self) -> str:
        return f'paraboloid(c={self.c:g})'


class SeparableTruth:
    """u = G(x)/alpha + (alpha/2) (y - y0)^2  ==>  det D^2 u = f(x).

    u_xx = f/alpha, u_yy = alpha, u_xy = 0.  alpha sets the anisotropy
    ("det = 4" or "det = 100" style contrasts are alpha^2 = 4 or 100).
    """

    name = 'separable'

    def __init__(self, source, alpha: float = 1.0, y0: float = 0.0):
        self.src = source
        self.alpha = float(alpha)
        self.y0 = float(y0)

    def __call__(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        u = self.src.F(x) / self.alpha + 0.5 * self.alpha * (y - self.y0) ** 2
        ux = self.src.F1(x) / self.alpha
        uy = self.alpha * (y - self.y0)
        return u, ux, uy

    def label(self) -> str:
        return f'separable({self.src.label()},alpha={self.alpha:g})'
