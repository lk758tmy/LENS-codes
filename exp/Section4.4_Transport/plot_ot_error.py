"""Plot the frozen square-to-triangle corner-loss comparison."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lens import metrics, transport_cases, transport_fit
import torch


def closest(points, polygon):
    edges = np.roll(polygon, -1, axis=0)-polygon
    offsets = points[:, None]-polygon
    t = np.clip((offsets*edges).sum(axis=2) / np.maximum((edges**2).sum(axis=1), 1e-30), 0, 1)
    feet = polygon + t[:, :, None]*edges
    return np.linalg.norm(points[:, None]-feet, axis=2).min(axis=1)


def training_image(base, row):
    cfg = row['case']
    region = transport_cases.make_region(cfg['region'])
    source = transport_cases.make_source(cfg['source'])
    target = transport_cases.make_target(cfg['target'], row['mass'])
    fit = transport_fit.TransportFit(region, target, K=cfg['K'], n_edge=cfg['m'],
                                    source=source, source_vertices=cfg['vertices'],
                                    lam=cfg['lam'], corner_weight=cfg['nu']).build()
    np.testing.assert_array_equal(fit.box, row['box'])
    fit._set_chart(cfg['K'])
    with np.load(base/'checkpoints'/f"{row['run_id']}.npz") as model:
        theta = torch.as_tensor(model['theta'])
        assert model['t'].size == fit.m == 260
    field = metrics.field_on_grid(fit.chart, theta, fit.xb, fit.yb)
    assert field['n_out'] == 0
    # The fit appends the four vertices; recover cyclic boundary order before connecting images.
    angle = np.arctan2(fit.yb-.5, fit.xb-.5)
    return np.column_stack((field['ux'], field['uy']))[np.argsort(angle)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'exp/release_2026-11-05/corners')
    parser.add_argument('--out', type=Path, default=ROOT/'report/figures/fig_ot_error.pdf')
    args = parser.parse_args()
    base = args.data_root
    records = {r['run_id']: r for r in map(json.loads, (base/'results.jsonl').read_text().splitlines())}

    plt.rcParams.update({'font.size': 9.0, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(7.5, 2.48), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=(1, .10), width_ratios=(1.45, 1, 1.12))
    profile_ax, gaps_ax, zoom_ax = [fig.add_subplot(grid[0, j]) for j in range(3)]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.set_axis_off()

    runs = {}
    region = transport_cases.make_region('S')
    corners = region.cum[:-1]
    for weight in (0, 1, 10):
        rid = f'C_S_P_f0_n{weight}'
        row = records[rid]
        vertices = np.array(row['target']['vertices'])
        vertex = vertices[np.argmax(vertices[:, 0])]
        scale = np.sqrt(row['mass'])
        with np.load(base/'fields'/f'{rid}.npz') as fields:
            source = np.column_stack((fields['x'], fields['y']))
            image = np.column_stack((fields['px'], fields['py']))
        steps = np.diff(np.vstack((source, source[0])), axis=0)
        arc = np.r_[0, np.cumsum(np.linalg.norm(steps, axis=1))]
        np.testing.assert_allclose(source[0], region.V[0])
        np.testing.assert_allclose(arc[-1], region.per)
        error = closest(image, vertices)
        gaps = closest(vertices, image)
        np.testing.assert_allclose([error.max()/scale, gaps.max()/scale],
                                   [row['metrics']['forward_max'], row['metrics']['corner_max']], rtol=1e-10)
        print(f'nu={weight}: vertex gaps={gaps}; dH={row["metrics"]["dH"]:.8g}', flush=True)
        runs[weight] = (arc, image, error, gaps, training_image(base, row))

    for corner in corners:
        profile_ax.axvline(corner, color='.62', lw=.65, linestyle='--', zorder=1)
    profile_ax.set(title='Forward boundary error', xlabel=r'Source arc length $s$',
                   xlim=(-.07, region.per+.07), xticks=region.cum, yscale='log',
                   ylim=(3e-7, .08), yticks=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
    profile_ax.grid(axis='y', color='.90', linewidth=.55)
    profile_ax.set_axisbelow(True)

    boundary = np.vstack((vertices, vertices[0]))
    target_line, = zoom_ax.plot(*boundary.T, '--', color='.28', lw=1.0, label='Target')
    fit_lines = []
    xlim = (vertex[0]-.033, vertex[0]+.009)
    ylim = (vertex[1]-.009, vertex[1]+.033)
    styles = [
        (0, '#ba4a32', 'o', '-'),
        (1, '#6350a4', '^', ':'),
        (10, '#00747d', 's', '-'),
    ]
    for j, (weight, color, marker, linestyle) in enumerate(styles):
        arc, image, error, gaps, controls = runs[weight]
        positive = np.where(error > 0, error, np.nan)
        profile_ax.plot(arc, np.r_[positive, positive[0]], color=color,
                        lw=.95, linestyle=linestyle, zorder=2)
        abscissa = np.arange(3) + .22*(j-1)
        gaps_ax.vlines(abscissa, 0, gaps, lw=1.3, color=color)
        gaps_ax.plot(abscissa, gaps, linestyle='none', marker=marker, ms=4, color=color)

        visible = ((controls[:, 0] >= xlim[0]) & (controls[:, 0] <= xlim[1])
                   & (controls[:, 1] >= ylim[0]) & (controls[:, 1] <= ylim[1]))
        print(f'nu={weight}: {visible.sum()} training boundary points in the zoom', flush=True)
        zoom_ax.plot(*np.vstack((image, image[0])).T, color=color, lw=.65, alpha=.26)
        line, = zoom_ax.plot(*np.vstack((controls, controls[0])).T, color=color, lw=1.05,
                             linestyle=linestyle, marker=marker, ms=2.8, label=rf'$\nu={weight}$')
        fit_lines.append(line)

    gaps_ax.set(xlim=(-.55, 2.55), ylim=(0, .0067), xticks=(0,1,2),
                xticklabels=(r'$90^\circ$', 'Right', 'Top'), yticks=(0,.002,.004,.006), title='Vertex gaps')
    gaps_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    gaps_ax.grid(axis='y', color='.90', linewidth=.55)
    gaps_ax.set_axisbelow(True)
    zoom_ax.set(aspect='equal', xlim=xlim, ylim=ylim, xticks=(.91,.93,.95),
                yticks=(-.48,-.46,-.44), title='Right acute vertex')
    zoom_ax.plot(*vertex, marker='*', ms=5.5, color='.18', zorder=4)
    for ax in (profile_ax, gaps_ax, zoom_ax):
        ax.tick_params(labelsize=7.3, length=2)

    legend_ax.legend(handles=[*fit_lines, target_line], loc='center', ncols=4, fontsize=8.0,
                     frameon=False, handlelength=2.0, columnspacing=1.55)
    fig.savefig(args.out, dpi=300, facecolor='white')


if __name__ == '__main__':
    main()

