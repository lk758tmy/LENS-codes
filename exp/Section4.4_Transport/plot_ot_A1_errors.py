"""Plot errors of the frozen D-to-A1 fit against its analytic solution."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=root/'exp/release_2026-11-05/transport_K16')
    parser.add_argument('--out', type=Path, default=root/'report/figures/fig_ot_A1_errors.png')
    args = parser.parse_args()
    sys.path.insert(0, str(root))
    from lens import metrics, transport_cases as cases, transport_fit
    import torch

    release = args.data_root
    row = next(r for r in map(json.loads, (release/'results.jsonl').read_text().splitlines())
               if r['run_id'] == 'A_f1')
    region = cases.make_region('D')
    exact = cases.Manufactured(1)
    fit = transport_fit.TransportFit(region, exact.target(), K=row['case']['K'],
                                     n_edge=row['case']['m'], source=cases.make_source(1)).build()
    np.testing.assert_array_equal(fit.box, row['box'])
    fit._set_chart(fit.K)
    with np.load(release/'checkpoints/A_f1.npz') as saved:
        theta = torch.as_tensor(saved['theta'])

    x, y = region.interior_grid(65)
    u, px, py = exact(x, y)
    field = metrics.field_on_grid(fit.chart, theta, x, y)
    assert field['n_out'] == 0
    gauge = float(np.mean(field['u']-u))
    e2 = metrics.errors(field['u'], u, center_reference=True, align_mean=True)['E2']
    egrad = np.linalg.norm(np.stack([field['ux']-px, field['uy']-py])) / np.linalg.norm(np.stack([px, py]))
    np.testing.assert_allclose([e2, egrad], [row['metrics']['E2'], row['metrics']['E_grad']], rtol=1e-10)

    X, Y = np.meshgrid(np.linspace(0, 1, 241), np.linspace(0, 1, 241))
    keep = region.contains(X.ravel(), Y.ravel()).reshape(X.shape)
    u, px, py = exact(X[keep], Y[keep])
    field = metrics.field_on_grid(fit.chart, theta, X[keep], Y[keep])
    assert field['n_out'] == 0
    potential, gradient = (np.full_like(X, np.nan) for _ in range(2))
    potential[keep] = field['u']-u-gauge
    gradient[keep] = np.hypot(field['ux']-px, field['uy']-py)
    vmax = float(np.nanmax(np.abs(potential)))
    print(f'A_f1: E2={e2:.8g}, Egrad={egrad:.8g}, dense maxima={vmax:.8g}, {np.nanmax(gradient):.8g}')

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.98), layout='constrained')
    th = np.linspace(0, 2*np.pi, 600)
    panels = [('Potential error', potential, 'RdBu_r', -vmax, vmax),
              ('Gradient error', gradient, 'magma', 0., float(np.nanmax(gradient)))]
    for ax, (title, error, cmap, lo, hi) in zip(axes, panels):
        image = ax.pcolormesh(X, Y, error, cmap=cmap, vmin=lo, vmax=hi, shading='nearest')
        ax.plot(.5+.5*np.cos(th), .5+.5*np.sin(th), color='.3', lw=.7)
        ax.set(xlabel='$x$', ylabel='$y$', aspect='equal', xlim=(-.03,1.03), ylim=(-.03,1.03), title=title)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        fig.colorbar(image, ax=ax, shrink=.83, format=fmt)
    fig.savefig(args.out, dpi=220, facecolor='white')


if __name__ == '__main__':
    main()
