"""Error norms and the PDE-residual audit.

E2      ||u-u*||_{L2(Omega)} / ||u*||_{L2(Omega)} for Dirichlet
Einf    same, Linf
OT      align both potentials to zero mean and use the centered reference scale
H1semi  RMS |grad u - grad u*|, no gauge
Rpde    max |det D^2 u_theta - f| on an independent audit grid, computed by
        *autograd through the actual forward map* (differentiable Newton inversion)

R_pde is deliberately not inferred from the algebraic identity: it measures the
realised residual of the returned object, which is what the paper claims.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------- evaluation
def field_on_grid(chart, theta, X, Y, t0=None):
    X = torch.as_tensor(np.asarray(X, float))
    Y = torch.as_tensor(np.asarray(Y, float))
    out = chart.field(X, Y, theta, t0=t0)
    return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in out.items()}


def errors(u, u_t, *, center_reference: bool = False, align_mean: bool = False) -> dict:
    u = np.asarray(u, float)
    u_t = np.asarray(u_t, float)
    if align_mean:
        u = u - float(np.mean(u - u_t))
    reference = u_t - u_t.mean() if center_reference else u_t
    scale = float(np.sqrt(np.mean(reference ** 2)))
    du = u - u_t
    return dict(E2=float(np.sqrt(np.mean(du ** 2)) / max(scale, 1e-300)),
                Einf=float(np.abs(du).max() / max(float(np.abs(reference).max()), 1e-300)),
                u_scale=scale, gauge=float(np.mean(u - u_t)))


def h1_semi(ux, uy, ux_t, uy_t) -> float:
    return float(np.sqrt(np.mean((ux - ux_t) ** 2 + (uy - uy_t) ** 2)))


# ------------------------------------------------------------ PDE audit (AD)
def audit_pde(chart, theta, X, Y, f_fn, nnewt: int = 3) -> float:
    """max |det D^2 u_theta(x,y) - f| by autograd through the returned map."""
    X = torch.as_tensor(np.asarray(X, float)).clone().requires_grad_(True)
    Y = torch.as_tensor(np.asarray(Y, float)).clone().requires_grad_(True)
    with torch.no_grad():
        t0, nbad = chart.solve_t(X, Y, theta)
        if nbad:
            return float('nan')  # Do not audit an unclamped, different inverse.
    t = chart.solve_t_diff(X, Y, theta, t0=t0, nnewt=nnewt)
    st = chart.eval(X, t, theta)
    u = st['w'] + Y * t
    ux = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
    uy = torch.autograd.grad(u.sum(), Y, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), X, create_graph=True)[0]
    uxy = torch.autograd.grad(ux.sum(), Y, create_graph=True)[0]
    uyy = torch.autograd.grad(uy.sum(), Y, create_graph=True)[0]
    det = uxx * uyy - uxy * uxy
    return float((det - f_fn(X)).detach().abs().max())


def f_torch(source):
    """Wrap a :class:`lens.sources.Source` as a torch callable."""
    def f(x):
        if isinstance(x, torch.Tensor):
            return torch.as_tensor(source(x.detach().cpu().numpy()))
        return source(x)
    return f


# --------------------------------------------------------------- diagnostics
def grid_in_region(region, n: int = 65, pad: float = 1e-9):
    return region.interior_grid(n, pad=pad)


def summarize(hist) -> dict:
    """Compact description of an LM history."""
    if not hist:
        return {}
    L = [h['loss'] for h in hist]
    nacc = sum(1 for h in hist if h['accepted'])
    return dict(iters=len(hist), accepted=nacc, L0=L[0], L1=L[-1],
                drop=math.log10(L[0] / L[-1]) if L[-1] > 0 else float('inf'))
