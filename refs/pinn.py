"""Physics-informed MLP baseline for the Monge-Ampere Dirichlet problem.

Soft-constraint formulation (the standard PINN setting for this equation):

    L = 100 * MSE_{dOmega}(u - g)  +  MSE_Omega( det D^2 u - f )

with an MLP  (x,y) -> u  (tanh, 3 hidden layers), a fixed budget of Adam steps
followed by L-BFGS, and the same boundary points as the LENS runs.  Ten seeds are
used per configuration and mean +- std are reported.

This is deliberately the *strong* version of the baseline: the boundary term is
weighted by 100, the interior quadrature is resampled every epoch, and L-BFGS is
warm started from the Adam iterate.
"""
from __future__ import annotations

import time

import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self, width: int = 48, depth: int = 3):
        super().__init__()
        layers = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy).squeeze(-1)


def _grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def det_hessian(u, xy):
    g = _grad(u, xy)
    ux, uy = g[:, 0], g[:, 1]
    uxx = _grad(ux, xy)[:, 0]
    uxy = _grad(ux, xy)[:, 1]
    uyy = _grad(uy, xy)[:, 1]
    return uxx * uyy - uxy * uxy


def solve_pinn(region, truth, source, n_int: int = 1024, n_bnd: int = 256,
               width: int = 48, depth: int = 3, adam: int = 1500, lbfgs: int = 500,
               seed: int = 0, w_bnd: float = 100.0, verbose: bool = False):
    """Train one PINN and return (field callable, metrics dict)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    x0, x1, y0, y1 = region.bbox
    # interior quadrature: uniform in the bounding box, resampled each step
    # one rejection pass builds a large pool; batches are then drawn from it
    pool = []
    need = 60000
    while need > 0:
        cand = np.column_stack([rng.uniform(x0, x1, 3 * need),
                                rng.uniform(y0, y1, 3 * need)])
        ok = region.contains(cand[:, 0], cand[:, 1])
        pool.append(cand[ok][:need])
        need -= pool[-1].shape[0]
    pool = torch.as_tensor(np.vstack(pool))

    def sample_int(n):
        idx = torch.randint(0, pool.shape[0], (n,))
        return pool[idx]

    xb, yb = region.boundary(n_bnd)
    tim = 0.0
    t0 = time.perf_counter()
    xy_b = torch.as_tensor(np.column_stack([xb, yb]))
    g_b = torch.as_tensor(truth(xb, yb)[0])

    def src_t(X):
        return torch.as_tensor(np.asarray(source(X.detach().cpu().numpy()), dtype=float))

    net = MLP(width, depth)
    n_par = sum(p.numel() for p in net.parameters())

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for it in range(adam):
        xy_i = sample_int(n_int)
        xy_i.requires_grad_(True)
        u_i = net(xy_i)
        r_i = det_hessian(u_i, xy_i) - src_t(xy_i[:, 0])
        loss = (r_i ** 2).mean() + w_bnd * ((net(xy_b) - g_b) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    tim += time.perf_counter() - t0
    t0 = time.perf_counter()

    # L-BFGS on a frozen quadrature (standard practice, avoids noisy gradients)
    xy_i = sample_int(max(n_int, 2048))
    xy_i.requires_grad_(True)
    opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs, history_size=50,
                             tolerance_grad=1e-12, tolerance_change=1e-14,
                             line_search_fn='strong_wolfe')

    def closure():
        opt2.zero_grad()
        u_i = net(xy_i)
        r_i = det_hessian(u_i, xy_i) - src_t(xy_i[:, 0])
        loss = (r_i ** 2).mean() + w_bnd * ((net(xy_b) - g_b) ** 2).mean()
        loss.backward()
        return loss

    opt2.step(closure)
    tim += time.perf_counter() - t0

    with torch.no_grad():
        field = lambda X, Y: net(torch.as_tensor(
            np.column_stack([np.asarray(X, float).ravel(),
                             np.asarray(Y, float).ravel()]))).detach().numpy().reshape(np.shape(X))
        res_b = float(torch.sqrt(((net(xy_b) - g_b) ** 2).mean()))
    xy_i = sample_int(4096)
    xy_i.requires_grad_(True)
    res_i = float(((det_hessian(net(xy_i), xy_i)
                              - src_t(xy_i[:, 0])) ** 2).mean().sqrt().detach())
    # convexity of the returned field: min eigenvalue of D^2 u on an interior grid
    xy_c = sample_int(2500)
    xy_c.requires_grad_(True)
    g1 = _grad(net(xy_c), xy_c)
    uxx = _grad(g1[:, 0], xy_c)[:, 0]
    uxy = _grad(g1[:, 0], xy_c)[:, 1]
    uyy = _grad(g1[:, 1], xy_c)[:, 1]
    ev = 0.5 * ((uxx + uyy) - torch.sqrt((uxx - uyy) ** 2 + 4 * uxy * uxy))
    min_eig = float(ev.detach().min())
    frac_neg = float((ev.detach() < 0).double().mean())
    return field, dict(params=n_par, wall=tim, rms_bnd=res_b, rms_pde=res_i,
                       min_eig=min_eig, frac_neg=frac_neg,
                       width=width, depth=depth, seed=seed)


def field_on_grid(field, X, Y):
    return field(X, Y)
