"""Figure style: one place for fonts, sizes, colours and export settings.

All figures in the paper are produced through this module so that they share the
same look (SIAM single column = 3.4 in wide, Times/STIX, no grid, thin lines).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402

PALETTE = ['#1f4e79', '#b45309', '#2e7d32', '#7b1fa2', '#c62828', '#00695c']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']

RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.2,
    'lines.markersize': 4.0,
    'lines.markeredgewidth': 0.7,
    'legend.frameon': False,
    'axes.grid': False,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,
}

W1, H1 = 3.4, 2.4          # single column
W2, H2 = 7.0, 2.4          # double column (one row)


def apply():
    plt.rcParams.update(RC)


def new(w=W1, h=H1, ncols=1, nrows=1, **kw):
    apply()
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, ax


def style(i: int):
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                markerfacecolor='none', markeredgecolor=PALETTE[i % len(PALETTE)])


def save(fig, path, formats=('pdf', 'png')):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    base, _ = os.path.splitext(path)
    outs = []
    for ext in formats:
        p = f'{base}.{ext}'
        fig.savefig(p)
        outs.append(p)
    plt.close(fig)
    return outs


def loglog_err(ax, x, y, label, i=0, ls='-'):
    ax.loglog(x, y, ls=ls, label=label, **style(i))
