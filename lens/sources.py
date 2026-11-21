"""Source terms f(x) with their exact primitives F'' = f, F' = F1.

Only analytic sources are used as *manufactured* data; the sign/step source is
only used to exhibit the Gibbs limit of a global polynomial representation
(never as an exact-convergence example).
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


class Source:
    """f(x) > 0 with F'' = f, F' = F1 (any additive constants are irrelevant)."""

    def __init__(self, kind: str, **kw):
        self.kind = kind
        self.kw = kw

    # ---- f ---------------------------------------------------------------
    def __call__(self, x):
        x = np.asarray(x, float)
        k = self.kind
        if k == 'const':
            return np.full_like(x, float(self.kw.get('c', 1.0)))
        if k == 'lin':                                  # 1 + a x
            return 1.0 + float(self.kw.get('a', 1.0)) * x
        if k == 'quad':                                 # 1 + a x^2
            return 1.0 + float(self.kw.get('a', 2.0)) * x * x
        if k == 'cos':                                  # 1 + a cos(2 pi x)
            return 1.0 + float(self.kw.get('a', 0.3)) * np.cos(TWO_PI * x)
        if k == 'step':                                 # 1 +- a
            a = float(self.kw.get('a', 0.9))
            return 1.0 + a * np.sign(x - float(self.kw.get('x0', 0.5)))
        if k == 'abs':                                  # 1 + a |x - x0|  (C^0, kink)
            a = float(self.kw.get('a', 0.9))
            return 1.0 + a * np.abs(x - float(self.kw.get('x0', 0.5)))
        raise KeyError(k)

    # ---- F' and F --------------------------------------------------------
    def F1(self, x):
        x = np.asarray(x, float)
        k = self.kind
        if k == 'const':
            return float(self.kw.get('c', 1.0)) * x
        if k == 'lin':
            return x + 0.5 * float(self.kw.get('a', 1.0)) * x * x
        if k == 'quad':
            return x + float(self.kw.get('a', 2.0)) * x ** 3 / 3.0
        if k == 'cos':
            a = float(self.kw.get('a', 0.3))
            return x + a * np.sin(TWO_PI * x) / TWO_PI
        if k == 'step':
            a = float(self.kw.get('a', 0.9))
            return x + a * np.abs(x - float(self.kw.get('x0', 0.5)))
        if k == 'abs':                              # d/dx |u|^3 / 6 = u|u| / 2
            a = float(self.kw.get('a', 0.9))
            u = x - float(self.kw.get('x0', 0.5))
            return x + 0.5 * a * u * np.abs(u)
        raise KeyError(k)

    def F(self, x):
        x = np.asarray(x, float)
        k = self.kind
        if k == 'const':
            return 0.5 * float(self.kw.get('c', 1.0)) * x * x
        if k == 'lin':
            return 0.5 * x * x + float(self.kw.get('a', 1.0)) * x ** 3 / 6.0
        if k == 'quad':
            return 0.5 * x * x + float(self.kw.get('a', 2.0)) * x ** 4 / 12.0
        if k == 'cos':                          # int int cos(2 pi x) = -cos(2 pi x)/(4 pi^2)
            a = float(self.kw.get('a', 0.3))
            return 0.5 * x * x - a * np.cos(TWO_PI * x) / (TWO_PI ** 2)
        if k == 'step':
            a = float(self.kw.get('a', 0.9))
            return 0.5 * x * x + a * 0.5 * (x - float(self.kw.get('x0', 0.5))) \
                * np.abs(x - float(self.kw.get('x0', 0.5)))
        if k == 'abs':                          # int int |u| = |u|^3 / 6
            a = float(self.kw.get('a', 0.9))
            u = x - float(self.kw.get('x0', 0.5))
            return 0.5 * x * x + a * np.abs(u) ** 3 / 6.0
        raise KeyError(k)

    def label(self) -> str:
        if self.kind == 'const':
            return f"const(c={self.kw.get('c', 1.0):g})"
        return self.kind + '(' + ','.join(f'{k}={v:g}' for k, v in self.kw.items()) + ')'
