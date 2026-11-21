"""Reference curvature truncation, exact inversion and driver stage wiring."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lens
import numpy as np
import torch
from lens import init, sources, regions, chart, cert, fit, truth


def main():
    region = regions.get_region('square')
    x, y = region.boundary(32)
    # Both dictionaries: exercise active truncation and the unchanged branch.
    for kind in ('const', 'fx'):
        cases = [(sources.Source('const', c=c), delta, expected)
                 for c, delta, expected in ((400., .1, .1), (4., .75, .75), (4., .1, .5))]
        if kind == 'fx':
            cases.append((sources.Source('quad', a=600.), .1, .1))
        for source, delta, expected in cases:
            g = source.F(x) + .5*y*y + .3*x - .2*y
            ii = init.boundary_reference(region, x, y, g, source, K=4,
                                         basis_kind=kind, delta=delta)
            assert ii['G'] == expected
            basis, theta = ii['basis'], ii['theta0']
            ch = chart.Chart(basis, cert.Certificate(basis.cert_S(17), delta=delta))
            t, outside = ch.solve_t(torch.as_tensor(x), torch.as_tensor(y), theta)
            assert outside == 0
            st = ch.eval(torch.as_tensor(x), t, theta)
            assert torch.max(torch.abs(st['wt'] + torch.as_tensor(y))) < 1e-12
            assert torch.max(torch.abs(st['wtt'] + expected)) < 1e-12
            assert ch.margin(theta) >= delta - 1e-12
            # Differentiate omega_x independently, rather than assuming its PDE.
            xx = torch.as_tensor(x).clone().requires_grad_()
            ex = ch.eval(xx, t.detach(), theta)
            wxx = torch.autograd.grad(ex['wx'].sum(), xx)[0]
            f = torch.as_tensor(source(x))
            assert torch.max(torch.abs(wxx + f*st['wtt'])) < 1e-10
            hyy = -1/st['wtt']
            hxy = -st['wxt']/st['wtt']
            hxx = wxx - st['wxt']**2/st['wtt']
            assert torch.max(torch.abs(hxx*hyy-hxy*hxy-f)) < 1e-10

    # Use a nondefault floor through build, coarse initialization and fine-box
    # construction; zero iterations isolate initialization/prolongation.
    original = init.boundary_reference
    seen = []
    def observed(*args, **kwargs):
        seen.append(kwargs.get('delta'))
        return original(*args, **kwargs)
    init.boundary_reference = observed
    try:
        for kind in ('const', 'fx'):
            source = sources.Source('const', c=4.)
            obj = fit.DirichletFit(region, source, K=5, m=32,
                                  truth=truth.SeparableTruth(source, alpha=1, y0=.5),
                                  basis_kind=kind, delta=.75, rounds1=0, rounds2=0)
            obj.build().fit()
            assert float(obj.theta[-1]) == .75
            assert obj.chart.margin(obj.theta) >= .75 - 1e-12
            assert obj.prolongation_error < 1e-10
    finally:
        init.boundary_reference = original
    assert seen == [.75]*6, seen
    print('PASS: clipped/unclipped references, both dictionaries, inverse/PDE identities, delta in all initialization stages')


if __name__ == '__main__':
    main()
