"""Classical reference solvers (baselines).

  fdm.py    finite differences with Shortley-Weller for curved boundaries
            (9-point Monge-Ampere stencil + damped Newton), copied unchanged from
            the validated exp3 implementation.
  pinn.py   physics-informed MLP baseline (soft det-penalty), used for the
            comparison table of S4.2.

Both are driven through the thin wrappers at the bottom of each file, so that the
experiment scripts only see (region, source, truth) -> (solution, metrics).
"""
