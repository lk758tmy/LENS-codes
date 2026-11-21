"""Regenerate the unweighted peanut error plot in Section 4.2."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lens import basis_const, cert, chart, regions, truth
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'exp/release_2026-11-05/dirichlet')
    parser.add_argument('--out', type=Path, default=ROOT / 'report/figures/fig_peanut_error.pdf')
    args = parser.parse_args()
    region = regions.peanut()
    reference = truth.RadialTruth()
    model_path = args.data_root / 'models/comparison_peanut_radial_K16_m256.npz'
    with np.load(model_path) as model:
        basis = basis_const.ConstBasis(c=1, K=16, box=model['box'])
        fitted = chart.Chart(basis, cert.Certificate(basis.cert_S(61), delta=.1, iota=.02))
        theta = torch.as_tensor(model['theta'])

    def error(x, y):
        field = fitted.field(x.ravel(), y.ravel(), theta)
        if field['n_out']:
            raise RuntimeError(f"{field['n_out']} plot points have no inverse inside the fitted box")
        return (field['u'].numpy() - reference(x.ravel(), y.ravel())[0]).reshape(x.shape)

    x, eta = np.meshgrid(np.linspace(-1, 1, 181), np.linspace(-1, 1, 151))
    knots = region.V[:len(region.V) // 2]
    y = eta * np.interp(x, knots[:, 0], -knots[:, 1])
    interior_error = error(x, y)
    s = (np.arange(2048) + .5) / 2048
    bx, by = region.at(s * region.per).T.copy()
    boundary_error = np.abs(error(bx, by))
    lower_arc = region.cum[len(knots) - 1] / region.per
    mask = s < lower_arc
    print(f'Lower arc fraction: {lower_arc:.6f}; squared boundary error share: '
          f'{np.sum(boundary_error[mask] ** 2) / np.sum(boundary_error ** 2):.6f}')

    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(5.7, 2.25), layout='constrained')
    grid = fig.add_gridspec(1, 2, width_ratios=(1.06, 1.20), wspace=.14)
    ax = fig.add_subplot(grid[0], aspect='equal')
    vmax = float(np.abs(interior_error).max())
    mesh = ax.pcolormesh(x, y, interior_error, shading='gouraud', cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax, rasterized=True)
    outline = np.vstack([region.V, region.V[0]])
    ax.plot(*outline.T, color='.22', linewidth=.65)
    ax.set(xlabel='$x$', ylabel='$y$', xticks=(-1, 0, 1), yticks=(-.8, 0, .8),
           xlim=(-1.06, 1.06), ylim=(-.91, .91), title='Signed field error')
    fig.colorbar(mesh, ax=ax, shrink=.83, fraction=.08, pad=.03, ticks=(-.06, 0, .06),
                 label=r'$u_h-u^\star$')

    ax = fig.add_subplot(grid[1])
    ax.axvspan(0, lower_arc, color='#edece7')
    ax.plot(s, boundary_error, color='#3b567c', linewidth=1.0)
    ax.text(.5 * lower_arc, .0725, 'lower arc', ha='center', fontsize=7.4)
    ax.grid(axis='y', color='.89', linewidth=.45)
    ax.set(xlim=(0, 1), ylim=(0, .080), xticks=(0, .25, .5, .75, 1), yticks=(0, .02, .04, .06),
           xlabel='Normalized arc length', ylabel=r'$|u_h-g|$', title='Absolute boundary error')
    fig.savefig(args.out, dpi=350, facecolor='white')


if __name__ == '__main__':
    main()
