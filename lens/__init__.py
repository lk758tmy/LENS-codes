"""LENS: Legendre-Embedded Nonlinear Solver for the 2-D Monge-Ampere equation.

Reproducibility package for Section 4/5 of the paper.  Import order matters:
``lens`` pins the thread count *before* numpy/torch are pulled in.

Layout
------
lens.env          thread pinning, timing, JSONL record
lens.regions      source domains and boundary sampling
lens.truth        manufactured / exact solutions
lens.basis        abstract basis protocol (chart generator omega)
lens.basis_const  generator (3.1) for f == const
lens.basis_fx     generator (3.3) for f = f(x)
lens.chart        omega(gamma) assembly, latent-coordinate inversion
lens.res          Dirichlet / transport residuals and analytic Jacobians
lens.lm           Schur-complement Levenberg-Marquardt (paper eq. 3.13-3.14)
lens.metrics      E2, Einf, H1, PDE-residual audit, transport diagnostics
lens.plot         figure style (one place for fonts/sizes/colors)
"""
from __future__ import annotations

from .env import pin_threads

THREADS = pin_threads()

import torch                                    # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(THREADS)
try:
    from threadpoolctl import threadpool_limits
    _blas_limits = threadpool_limits(limits=THREADS)
except ImportError:
    pass

__all__ = ['THREADS']
