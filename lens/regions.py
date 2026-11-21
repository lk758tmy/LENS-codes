"""Source domains Omega and arc-length boundary sampling.

All regions expose the same three services:

    region.boundary(n)        -> (x, y)  n points, uniform in arc length
    region.contains(x, y)     -> bool array (crossing-number test on a dense
                                 polygonal rendering, valid for non-convex shapes)
    region.area()             -> shoelace area of the same rendering

Two implementations cover every domain used in the paper:

    PolygonRegion   [Square, LShape, IsoTriangle]        exact vertices
    ParamRegion     [Disk, Superellipse, Flower]         r(theta) / implicit curve,
                                                         re-parametrised by arc
                                                         length

Design note (S4.2/S4.4): the chart only requires Omega to be *vertically convex*
(every line x = const meets Omega in one interval).  The axis-aligned L-shape
[-1,1]^2 minus the first quadrant satisfies this; a 45-degree rotation would not.
"""
from __future__ import annotations

import math

import numpy as np


class Region:
    name = 'region'
    bbox = (-1.0, 1.0, -1.0, 1.0)       # xmin, xmax, ymin, ymax

    # -- to be provided by subclasses -------------------------------------
    def boundary(self, n: int, seed: int | None = None):  # pragma: no cover
        raise NotImplementedError

    def _poly(self, n: int = 2000) -> np.ndarray:         # pragma: no cover
        x, y = self.boundary(n)
        return np.column_stack([x, y])

    # -- generic services --------------------------------------------------
    def contains(self, x, y):
        """Crossing-number test against a dense polygonal rendering."""
        V = getattr(self, '_poly_cache', None)
        if V is None:
            V = self._poly(4000)
            self._poly_cache = V
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        inside = np.zeros(x.shape, dtype=bool)
        n = len(V)
        for i in range(n):
            x1, y1 = V[i]
            x2, y2 = V[(i + 1) % n]
            cond = (y1 > y) != (y2 > y)
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
            inside ^= cond & (x < xin)
        return inside

    def area(self) -> float:
        V = getattr(self, '_poly_cache', None)
        if V is None:
            V = self._poly(4000)
            self._poly_cache = V
        x, y = V[:, 0], V[:, 1]
        return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    def interior_grid(self, n: int = 65, pad: float = 1e-9):
        """Uniform n x n grid over bbox, restricted to Omega (for error norms)."""
        x0, x1, y0, y1 = self.bbox
        gx = np.linspace(x0 + pad, x1 - pad, n)
        gy = np.linspace(y0 + pad, y1 - pad, n)
        X, Y = np.meshgrid(gx, gy, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        m = self.contains(X, Y)
        return X[m], Y[m]

    def span(self) -> float:
        x0, x1, y0, y1 = self.bbox
        return max(x1 - x0, y1 - y0)

    def center(self):
        x0, x1, y0, y1 = self.bbox
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


# ---------------------------------------------------------------- polygons
class PolygonRegion(Region):
    """Closed polygon given by its vertices (any orientation, not repeated)."""

    def __init__(self, verts, name: str = 'polygon'):
        self.V = np.asarray(verts, float)
        self.name = name
        x0, y0 = self.V.min(axis=0)
        x1, y1 = self.V.max(axis=0)
        self.bbox = (float(x0), float(x1), float(y0), float(y1))
        Vc = np.vstack([self.V, self.V[:1]])
        d = np.diff(Vc, axis=0)
        self._d = d
        self.seg = np.hypot(d[:, 0], d[:, 1])
        self.per = float(self.seg.sum())
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg)])

    def at(self, s):
        """Points at arc length ``s`` (array, periodic)."""
        s = np.mod(np.asarray(s, float), self.per)
        i = np.clip(np.searchsorted(self.cum, s, side='right') - 1, 0, len(self.seg) - 1)
        f = (s - self.cum[i]) / self.seg[i]
        return self.V[i] + f[:, None] * self._d[i]

    def boundary(self, n: int, seed: int | None = None):
        """Midpoint rule in arc length (deterministic) or jittered strata."""
        s = (np.arange(n) + 0.5) * (self.per / n)
        if seed is not None:
            rng = np.random.default_rng(seed)
            s = (np.arange(n) + rng.random(n)) * (self.per / n)
        P = self.at(s)
        return P[:, 0], P[:, 1]

    def _poly(self, n: int = 2000) -> np.ndarray:
        return self.V


class Square(PolygonRegion):
    def __init__(self, x0: float = 0.0, x1: float = 1.0,
                 y0: float = 0.0, y1: float = 1.0):
        super().__init__([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], 'square')


def peanut_smooth(a: float = 1.0, b: float = 0.85, depth: float = 0.55,
                  width: float = 0.35, n: float = 6.0, m: int = 1600,
                  name: str = 'peanut_smooth') -> 'PolygonRegion':
    """Waist domain with *smooth* corners:  |y| <= h(x),

        h(x) = b (1 - depth exp(-(x/width)^2)) (1 - (|x|/a)^n)^(1/n).

    Unlike :func:`peanut` (flat-topped profile closed by vertical edges, i.e. four
    corners), the superellipse factor ends the profile with a vertical tangent, so
    the boundary is C^1 everywhere and still vertically convex --- the only
    hypothesis the chart needs.  This isolates the effect of a waist from the
    effect of a corner.
    """
    xs = np.linspace(-a, a, m // 2)
    u = np.clip(np.abs(xs) / a, 0.0, 1.0)
    cap = np.clip(1.0 - u ** n, 0.0, None) ** (1.0 / n)
    h = b * (1.0 - depth * np.exp(-(xs / width) ** 2)) * cap
    V = np.vstack([np.column_stack([xs, -h]),
                   np.column_stack([xs[::-1], h[::-1]])])
    return PolygonRegion(V, name=name)


class LShape(PolygonRegion):
    """[-1,1]^2 minus the first quadrant (vertically convex, one re-entrant corner)."""

    def __init__(self):
        super().__init__([(-1, -1), (1, -1), (1, 0), (0, 0), (0, 1), (-1, 1)], 'lshape')


class IsoTriangle(PolygonRegion):
    """Right isosceles triangle, legs along +x/+y, area = leg^2/2."""

    def __init__(self, leg: float = 1.0, origin=(0.0, 0.0), centroid_at=None):
        ox, oy = origin
        if centroid_at is not None:            # place centroid at a given point
            cx, cy = centroid_at
            ox, oy = cx - leg / 3.0, cy - leg / 3.0
        super().__init__([(ox, oy), (ox + leg, oy), (ox, oy + leg)], 'triangle')


# ---------------------------------------------------------------- parametric
class ParamRegion(Region):
    """Region described by a parametric boundary; sampled uniformly in arc length.

    Subclasses implement ``_xy(theta)``; the re-parametrisation table is built
    once on 4097 nodes and interpolated.
    """

    def _xy(self, th):                       # pragma: no cover
        raise NotImplementedError

    def _table(self):
        t = getattr(self, '_tab', None)
        if t is None:
            th = np.linspace(0.0, 2.0 * math.pi, 4097)
            x, y = self._xy(th)
            ds = np.hypot(np.diff(x), np.diff(y))
            s = np.concatenate([[0.0], np.cumsum(ds)])
            self._tab = (th, s, s[-1])
            t = self._tab
        return t

    def boundary(self, n: int, seed: int | None = None):
        th, s, tot = self._table()
        u = (np.arange(n) + 0.5) / n
        if seed is not None:
            rng = np.random.default_rng(seed)
            u = (np.arange(n) + rng.random(n)) / n
        thn = np.interp(u * tot, s, th)
        x, y = self._xy(thn)
        return x, y

    def _poly(self, n: int = 2000) -> np.ndarray:
        x, y = self.boundary(n)
        return np.column_stack([x, y])


class Disk(ParamRegion):
    def __init__(self, cx: float = 0.5, cy: float = 0.5, r: float = 0.5):
        self.cx, self.cy, self.r = float(cx), float(cy), float(r)
        self.name = 'disk'
        self.bbox = (cx - r, cx + r, cy - r, cy + r)

    def _xy(self, th):
        return self.cx + self.r * np.cos(th), self.cy + self.r * np.sin(th)


class Superellipse(ParamRegion):
    """|x/a|^n + |y/b|^n = 1, centred at the origin (n = 4 gives a rounded box)."""

    def __init__(self, a: float = 0.5, b: float = 0.4, n: float = 4.0):
        self.a, self.b, self.n = float(a), float(b), float(n)
        self.name = f'superellipse{n:g}'
        self.bbox = (-a, a, -b, b)

    def _xy(self, th):
        p = 2.0 / self.n
        c, s = np.cos(th), np.sin(th)
        return (self.a * np.abs(c) ** p * np.sign(c),
                self.b * np.abs(s) ** p * np.sign(s))


class Polar(ParamRegion):
    """r(theta) = base * (1 + sum_k amp_k cos(k theta + ph_k)); strictly convex iff
    sum |amp_k| * (k^2 + 1) < 1."""

    def __init__(self, base: float = 0.75, modes=((4, 0.05, 0.0),), name='polar'):
        self.base = float(base)
        self.modes = tuple(modes)
        self.name = name
        rmax = base * (1.0 + sum(abs(a) for _k, a, _p in self.modes))
        self.bbox = (-rmax, rmax, -rmax, rmax)

    def _r(self, th):
        r = np.full_like(np.asarray(th, float), self.base)
        for k, amp, ph in self.modes:
            r = r + self.base * amp * np.cos(k * th + ph)
        return r

    def _xy(self, th):
        r = self._r(th)
        return r * np.cos(th), r * np.sin(th)


# ---------------------------------------------------------------- registry
REGISTRY = {
    'square':       lambda: Square(0.0, 1.0, 0.0, 1.0),
    'disk':         lambda: Disk(0.5, 0.5, 0.5),
    'superellipse': lambda: Superellipse(0.5, 0.4, 4.0),
    'flower':       lambda: Polar(0.75, ((4, 0.05, 0.0),), 'flower'),
    'lshape':       lambda: LShape(),
    'peanut':       lambda: peanut(),
    'peanut_smooth': lambda: peanut_smooth(),
}


def get_region(name: str) -> Region:
    if name not in REGISTRY:
        raise KeyError(f'unknown region {name!r}; have {sorted(REGISTRY)}')
    return REGISTRY[name]()

def peanut(a: float = 1.0, b: float = 0.85, depth: float = 0.55, width: float = 0.35,
           n: int = 1600, name: str = 'peanut') -> 'PolygonRegion':
    """Smooth waist domain  {(x,y): |x|<=a, |y| <= b(1-depth*exp(-(x/width)^2))}.

    Non-convex but vertically convex, so the chart applies; unlike an L-shape it has
    no re-entrant corner, which keeps the initialisation well conditioned.
    """
    xs = np.linspace(-a, a, n // 2)
    h = b * (1.0 - depth * np.exp(-(xs / width) ** 2))
    V = np.vstack([np.column_stack([xs, -h]), np.column_stack([xs[::-1], h[::-1]])])
    return PolygonRegion(V, name=name)
