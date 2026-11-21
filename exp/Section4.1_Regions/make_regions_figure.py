"""Figure for Section 4.1: all source domains Omega used in the paper.

Each panel uses the corresponding experimental source-domain definition, so
the figure cannot drift away from the code.  Output: out/fig_regions.{pdf,png}.

    python exp/Section4.1_Regions/make_regions_figure.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np                                    # noqa: E402

from lens import plot                                 # noqa: E402
from lens import regions as RG
from lens import transport_cases as cases                        # noqa: E402

FILL = '#cfe0ee'
EDGE = '#1f4e79'


def _outline(reg):
    """Closed polygonal outline of a region (dense, arc-length spaced)."""
    if isinstance(reg, RG.PolygonRegion):
        V = reg.V
    else:
        x, y = reg.boundary(4000)
        V = np.column_stack([x, y])
    return np.vstack([V, V[:1]])


def panel(ax, reg, title, ticks=3, dots=(), note=None):
    """``note`` = (text, xy_data) draws a leader line to a feature of the shape."""
    V = _outline(reg)
    ax.fill(V[:, 0], V[:, 1], facecolor=FILL, edgecolor=EDGE, linewidth=0.7,
            zorder=2)
    for (px, py) in dots:
        ax.plot([px], [py], marker='o', ms=2.2, color='#c62828', zorder=4,
                clip_on=False)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.tick_params(labelsize=7, length=2.2, width=0.4, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.45)
        s.set_color('0.55')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(plot.plt.MaxNLocator(ticks, prune=None))
    if note:
        txt, xy = note
        ax.annotate(txt, xy=xy, xytext=(0.5, 0.5), textcoords='axes fraction',
                    fontsize=6.5, color='#7a1f1f', ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.45, color='#7a1f1f',
                                    shrinkA=0, shrinkB=1), zorder=5)


def main():
    plot.apply()
    plt = plot.plt

    panels = [
        (cases.make_region('S'), 'square (S)'),
        (cases.make_region('D'), 'disk (D)'),
        (cases.make_region('P'), 'triangle (T)'),
        (cases.make_region('U', superellipse_center=(.5, .5)), 'superellipse (U)'),
        (RG.peanut(), 'peanut (P)'),
    ]

    fig = plt.figure(figsize=(plot.W2, 1.62))
    gs = fig.add_gridspec(1, 5, left=0.03, right=0.99, top=0.87,
                          bottom=0.11, wspace=0.45)
    spec = [gs[0, i] for i in range(5)]

    for ax_pos, (reg, title) in zip(spec, panels):
        ax = fig.add_subplot(ax_pos)
        dots = [(0.0, 0.0)] if title.endswith('L-shape') else ()
        panel(ax, reg, title, dots=dots)

    out = os.environ.get('LENS_FIGURE_OUT', os.path.join(HERE, 'out'))
    os.makedirs(out, exist_ok=True)
    print(plot.save(fig, os.path.join(out, 'fig_regions')))


if __name__ == '__main__':
    main()
