"""Initialisation of the chart from the Dirichlet data.

Why not simply  theta0 = 0 ?
---------------------------
With theta = 0 and gamma = delta the initial chart maps  y -> t = delta*y, usually a
factor ten off the real latent scale, so the latent coordinates leave the prior box
before the first step.

What is used instead (the analogue of the covariance-matching init of the transport
case, which reads the scale off the target T):

1. fit the boundary data by a 6-term quadratic  u ~ a x^2 + b xy + d y^2 + ...  (a
   linear least-squares problem on the m boundary points);
2. read the reference curvature  G = 1/d  and the latent range  t ~= u_y|_dOmega
   (the normal derivative of the surrogate on the boundary is exactly t there);
3. prior box from that range; gamma0 = G so that the base quadratic alone already
   has the right curvature (omega_tt = -gamma = -G);
4. shift the linear-in-t coefficient beta_1 so the chart covers the data's y-range
   (exact in one Newton step: y is affine in beta_1).

Everything is data driven; nothing is fitted to the reference solution.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .basis_const import ConstBasis


def quad_surrogate(xb, yb, g):
    """Least-squares  u ~ c0 + c1 x + c2 y + c3 x^2 + c4 xy + c5 y^2."""
    A = np.column_stack([np.ones_like(xb), xb, yb, xb * xb, xb * yb, yb * yb])
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    return coef, float(np.sqrt(np.mean((A @ coef - g) ** 2)))


def curvature_from_gamma(gamma: float, delta: float, iota: float) -> float:
    """G = gamma - max H_tt  given  gamma = (1+iota) max H_tt + delta."""
    return (gamma * iota + delta) / (1.0 + iota)


def gamma_for_curvature(G: float, delta: float, iota: float) -> float:
    """Inverse of :func:`curvature_from_gamma` (only used by the 'cert' variant)."""
    return ((1.0 + iota) * G - delta) / iota


def dirichlet_init(region, xb, yb, g, c: float = 1.0, K: int = 12,
                   delta: float = 0.1, iota: float = 0.02,
                   margin: float = 0.35, t_box=None, gamma_init=None,
                   gamma_mode: str = 'param'):
    """Return dict(basis, theta0, box, G, t_range, quad_rms)."""
    x0, x1, y0, y1 = region.bbox
    sq = math.sqrt(c)
    coef, qres = quad_surrogate(np.asarray(xb, float), np.asarray(yb, float),
                                np.asarray(g, float))
    c0, c1, c2, c3, c4, c5 = coef
    d = 2.0 * c5                                  # u_yy of the surrogate
    G = 1.0 / d if d > 1e-12 else 1.0
    uy = c2 + c4 * np.asarray(xb, float) + 2.0 * c5 * np.asarray(yb, float)
    t_lo_s, t_hi_s = float(uy.min()), float(uy.max())
    if t_box is not None:
        t_lo, t_hi = float(t_box[0]), float(t_box[1])
    else:
        mid = 0.5 * (t_lo_s + t_hi_s)
        half = 0.5 * (t_hi_s - t_lo_s) * (1.0 + margin) + 1e-3
        t_lo, t_hi = mid - half, mid + half
    box = (x0, x1, t_lo, t_hi)
    z0 = (sq * 0.5 * (x0 + x1), 0.5 * (t_lo + t_hi))
    basis = ConstBasis(c=c, K=K, box=box, z0=z0)
    n = basis.nmodes
    na = basis.n_alpha

    if gamma_mode == 'param':
        theta0 = torch.zeros(n + 1)
        gamma0 = G if gamma_init is None else float(gamma_init)
        theta0[-1] = gamma0
    else:
        gam_t = (gamma_for_curvature(G, delta, iota) if gamma_init is None
                 else float(gamma_init))
        theta0 = basis.seed(max(gam_t, delta * (1.0 + 1e-9)), delta, iota)
        gamma0 = gam_t

    # shift beta_1 so that the chart covers the data's y-range (exact Newton step)
    xt = torch.as_tensor(np.asarray(xb, float))
    tmid = torch.full_like(xt, 0.5 * (t_lo + t_hi))
    P = basis.parts(xt, tmid)
    Phi, _, Phi_t = basis.phi(xt, tmid)
    y_now = -(theta0[:na] @ P['WT'][:na] + theta0[na:2 * na] @ P['WT'][na:]
              + gamma0 * Phi_t)
    slope = float((-P['WT'][na + 1]).mean())       # d y / d beta_1  (constant)
    if abs(slope) > 1e-12:
        theta0[na + 1] += (float(np.mean(yb)) - float(y_now.mean())) / slope
    return dict(basis=basis, theta0=theta0, box=box, G=G, gamma0=gamma0,
                quad_rms=qres, t_range=(t_lo_s, t_hi_s), t_box=(t_lo, t_hi))


def fx_surrogate(xb, yb, g):
    """Curvature/scale estimate for the f(x) chart.

    A six-term quadratic in (x,y) is not enough here: with f depending on x the
    exact solution carries F(x) of degree >= 2 in x, which contaminates the y^2
    coefficient (on the disk it biases 1/G = u_yy by an order of magnitude, and the
    initial t-box then clips every point).  We therefore fit

        u ~ p(x) + b(x) y + d y^2,     p, b polynomials of degree <= 4,

    which absorbs the x-dependence and leaves d = u_yy/2.
    """
    x = np.asarray(xb, float)
    y = np.asarray(yb, float)
    cols = [np.ones_like(x)] + [x ** k for k in (1, 2, 3, 4)]
    cols += [y, x * y, x * x * y, y * y]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    d = 2.0 * coef[-1]
    uy = coef[5] + coef[6] * x + coef[7] * x * x + 2.0 * coef[-1] * y
    rms = float(np.sqrt(np.mean((A @ coef - g) ** 2)))
    return d, uy, rms


def dirichlet_init_fx(region, xb, yb, g, source, K: int = 16,
                     delta: float = 0.1, iota: float = 0.02,
                     margin: float = 0.35, t_box=None, seed: str = "paper",
                     refine: int = 0, sweeps: int = 3, K_start: int = 3,
                     stages=(3, 5, 7, 9, 11, 13, 15)):
    """Initialisation of the f(x) chart (``lens.basis_fx.FxBasis``).

    Three things have to be done together, and none of them works alone:

    * the reference curvature cannot be read off a quadratic surrogate of the data
      (on a circle y^2 is a combination of 1, x, x^2, ... along dOmega, so u_yy is
      not determined: the estimate is off by a factor four);
    * it cannot be found at full order either: with all modes the alternating map
      "solve t, solve (theta, gamma) linearly at frozen t" has no usable basin
      (rms = 1e-16 at the exact gamma, 25 at gamma - 10%, 3e2 at gamma + 10%),
      because a latent coordinate off by O(1) makes the design matrix at that
      coordinate nearly singular and the linear solve returns huge coefficients;
    * but at K = K_start (a handful of modes) the landscape rms(gamma) is smooth
      and unimodal, and a 13-point scan of gamma finds it.

    So: scan gamma at K = K_start, then continue in the order (K = 5, 7, ... up to
    the requested K) carrying the latent coordinates as the warm start.  At every
    stage the previous chart already fits the data to 1e-11, so the linearisation
    point is accurate and the stage is a pure refinement.  The LM then takes over.
    Measured on f = 4, 1+x, 1+2x^2, 1+0.3cos(2pi x), square and disk: 6e-15 to
    4e-14 after the continuation, 1e-15 after the LM, margin 1.0.
    """
    from .basis_fx import FxBasis
    from .cert import Certificate
    from .chart import Chart

    x0, x1, y0, y1 = region.bbox
    span = max(x1 - x0, y1 - y0)
    if t_box is not None:
        t_lo, t_hi = float(t_box[0]), float(t_box[1])
    else:
        t_lo, t_hi = y0 - span, y1 + span
    box = (float(x0), float(x1), t_lo, t_hi)

    xt = torch.as_tensor(np.asarray(xb, float))
    yt = torch.as_tensor(np.asarray(yb, float))
    gt = torch.as_tensor(np.asarray(g, float))

    def sweep(ch, basis, t, gam, n_it):
        """n_it alternations of (solve t, solve (theta, gamma) linearly)."""
        n, na = basis.nmodes, basis.n_alpha
        rows = [i for i in range(n) if i != na]
        th = torch.zeros(n + 1, dtype=torch.float64)
        th[-1] = gam
        if t is None:
            t, _ = ch.solve_t(xt, yt, th)
        for _ in range(n_it):
            st = ch.eval(xt, t, th)
            A = np.column_stack([st["parts"]["W"][rows].numpy().T,
                                 st["Phi"].numpy()])
            rhs = (gt - yt * t).numpy()
            lam = 1e-11 * float(np.trace(A.T @ A)) / A.shape[1] + 1e-30
            try:
                sol = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ rhs)
            except Exception:
                sol, *_ = np.linalg.lstsq(A, rhs, rcond=1e-12)
            if not np.all(np.isfinite(sol)):
                break
            th = torch.zeros(n + 1, dtype=torch.float64)
            th[rows] = torch.as_tensor(sol[:-1])
            th[-1] = max(float(sol[-1]), float(ch.cert_bound(th[:n])) + 0.05)
            t, _ = ch.solve_t(xt, yt, th, t0=t)
        st = ch.eval(xt, t, th)
        return th, t, float((st["w"] + yt * t - gt).pow(2).mean().sqrt())

    ks = sorted({k for k in stages if k < K} | {max(int(K_start), 1), int(K)})
    ks = [k for k in ks if k <= K]

    u_span = float(np.ptp(g)) if float(np.ptp(g)) > 1e-12 else 1.0
    gam = u_span / max((y1 - y0) ** 2, 1e-12)
    t = None
    theta = None
    rms = np.inf
    basis = chart = None
    for i, kk in enumerate(ks):
        basis = FxBasis(source, box, K=kk, seed=seed)
        cert = Certificate(basis.cert_S(61), "box", delta, iota,
                           W_disc=basis.W_disc())
        chart = Chart(basis, cert)
        if i == 0:
            best = (np.inf, gam, None, None)
            for k in range(-6, 7):                 # gamma scan at the low order
                G = gam * 2.0 ** (0.5 * k)
                th, tt, r = sweep(chart, basis, None, G, 2)
                if r < best[0]:
                    best = (r, G, th, tt)
            rms, gam, theta, t = best
        else:
            theta, t, rms = sweep(chart, basis, t, gam, sweeps)
            gam = float(theta[-1])

    out = dict(basis=basis, theta0=theta, box=box, G=float(theta[-1]),
               gamma0=float(theta[-1]), quad_rms=rms, t_box=(t_lo, t_hi),
               stages=ks)
    if refine > 0:
        try:
            out["theta0"] = refine_init(chart, xb, yb, g, theta, n_iter=refine)
            out["refined"] = True
        except Exception:
            out["refined"] = False
    return out


def transport_init(chart, region, T, n: int = 240):
    """Initial guess for the transport problem from the arclength correspondence.

    The optimal map sends dOmega monotonically onto dT, so we pair the two
    boundaries by equal arclength fraction and fit, by *linear* least squares,

        omega_x(x_i, q_i) = p_i          (first component of grad u)

    over (theta, gamma).  The latent coordinates are then obtained from the exact
    inversion omega_t + y = 0, and gamma is projected onto the certificate (2.5).
    This replaces exp7's covariance-matching init and works for any target shape.

    Returns (theta, n_points).
    """
    b = chart.basis
    xb, yb = region.boundary(n)
    s = (np.arange(n) + 0.5) / n * T.per
    p, q = T.at(s)
    X = torch.as_tensor(xb)
    Q = torch.as_tensor(q)
    P = b.parts(X, Q)
    _, Phi_x, _ = b.phi(X, Q)
    A = torch.cat([P['WX'], Phi_x[None, :]]).numpy()        # (nmodes+1, n)
    keep = np.zeros(A.shape[0], dtype=bool)
    keep[:3] = True
    keep[b.n_alpha + 1:b.n_alpha + 3] = True      # affine init: modes up to zeta^2
    keep[b.n_alpha] = False                                 # beta_0 is the zero mode
    keep &= np.abs(A).max(axis=1) > 1e-14
    sol = np.zeros(A.shape[0])
    coef, *_ = np.linalg.lstsq(A[keep].T, p, rcond=1e-12)
    sol[keep] = coef
    theta = torch.as_tensor(sol)
    # beta_1 is invisible to omega_x (d_x of -Im zeta = 0): choose it so that the
    # chart covers the target's q-range, i.e. centre y on [q_min, q_max]
    na = b.n_alpha
    tmid = torch.full_like(X, b.tc)
    Pt = b.parts(X, tmid)
    _, _, Phi_t = b.phi(X, tmid)
    gam = float(theta[-1])
    y_now = -(theta[:na] @ Pt['WT'][:na] + theta[na:2 * na] @ Pt['WT'][na:]
              + gam * Phi_t)
    slope = float((-Pt['WT'][na + 1]).mean())
    if abs(slope) > 1e-12:
        theta[na + 1] += (0.5 * (T.qmin + T.qmax) - float(y_now.mean())) / slope
    lb = chart.cert_bound(theta[:b.nmodes])
    theta[-1] = max(float(theta[-1]), float(lb))
    return theta, n

def refine_init(chart, xb, yb, g, theta, n_iter: int = 4, cap: float = 60.0):
    """Improve an initial chart by alternating linear least squares.

    With the latent coordinates frozen, the Dirichlet residual
    omega(x_i,t_i) + y_i t_i - g_i is linear in (theta, gamma).  We alternate
    (i) a ridge-continued solve for the coefficients at fixed gamma, (ii) a scalar
    update of gamma, and (iii) the convexity floor gamma >= max_B H_tt, which the
    alternating structure makes consistent -- clamping gamma alone (without
    re-solving the coefficients at the new gamma) destroys the fit on waisted
    domains.  Returns the input unchanged if the iteration does not reduce the
    coefficient norm below `cap`.
    """
    import numpy as _np

    b = chart.basis
    x = torch.as_tensor(xb)
    y = torch.as_tensor(yb)
    gt = torch.as_tensor(g)
    th = theta.clone()
    keep = _np.ones(b.nmodes, dtype=bool)
    keep[b.n_alpha] = False                       # beta_0 is identically zero
    kidx = torch.as_tensor(_np.where(keep)[0])
    for _ in range(n_iter):
        t, _ = chart.solve_t(x, y, th)
        P = b.parts(x, t)
        Phi, _, _ = b.phi(x, t)
        gam = float(th[-1])
        rhs = gt - y * t - gam * Phi
        Am = P['W'][keep].t().numpy()
        bm = rhs.numpy()
        scale = float(np.sum(Am * Am)) / Am.shape[1]
        sol = None
        for mu in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            if mu == 0.0:
                cand = _np.linalg.lstsq(Am, bm, rcond=1e-12)[0]
            else:
                Gm = Am.T @ Am + mu * scale * _np.eye(Am.shape[1])
                cand = _np.linalg.solve(Gm, Am.T @ bm)
            sol = cand
            if float(np.linalg.norm(cand)) <= cap:
                break
        th = torch.zeros_like(theta)
        th[kidx] = torch.as_tensor(sol)
        # scalar update of gamma on the remaining residual
        r = bm - Am @ sol
        phi = Phi.numpy()
        dg = float(phi @ r) / max(float(phi @ phi), 1e-30)
        gam = gam + dg
        gam = max(gam, float(chart.cert.max_htt(th[:b.nmodes])))
        th[-1] = gam
    return th


def boundary_reference(region, xb, yb, g, source, K=16, basis_kind='const',
                       t_box=None, margin=0.35, delta=0.1, **unused):
    """One common convex reference, using f and boundary values only.

    u_ref=G F(x)+y^2/(2G)+b*x+d*y+e, G=max(delta,1/sqrt(mean(f))).
    Only (b,d,e) are fitted by linear least squares. It is not an inverse solve.
    """
    from .basis_fx import FxBasis
    from numpy.polynomial import chebyshev as C
    x0,x1,y0,y1=region.bbox
    G=max(float(delta),1/np.sqrt(float(np.mean(source(np.linspace(x0,x1,129))))))
    if basis_kind == 'fx':
        tmp=FxBasis(source,(x0,x1,-1,1),K=2,seed='paper')
        F=C.chebval((xb-tmp._xo)/tmp._xs,tmp.cF)
    else:
        c=float(np.mean(source(xb)))
        F=0.5*c*(xb-0.5*(x0+x1))**2
    A=np.column_stack([xb,yb,np.ones_like(xb)])
    b,d,e=np.linalg.lstsq(A,g-G*F-yb**2/(2*G),rcond=None)[0]
    tr=yb/G+d
    pad=(margin+0.15)*(y1-y0)/G+1e-3
    tl,th=(float(tr.min()-pad),float(tr.max()+pad)) if t_box is None else t_box
    box=(x0,x1,tl,th)
    basis=(FxBasis(source,box,K=K,seed='paper') if basis_kind=='fx'
           else ConstBasis(c=c,K=K,box=box))
    theta=torch.zeros(basis.nmodes+1)
    theta[-1]=G
    tc=basis.tc
    if basis_kind=='fx':
        theta[1]=G*(d-tc)*basis.g[1]
        theta[basis.n_alpha]=b*basis._xs*basis.g[basis.n_alpha]
        theta[0]=e+b*basis._xo-0.5*G*(d-tc)**2
    else:
        theta[1]=b*basis.R*basis.g[1]/basis.sq
        j=basis.n_alpha+1
        theta[j]=-G*(d-tc)*basis.R*basis.g[j]
        theta[0]=e+b*basis.xc-0.5*G*(d-tc)**2
    return dict(basis=basis,theta0=theta,G=G,box=box)
