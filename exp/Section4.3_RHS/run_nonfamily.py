"""Fixed-order Dirichlet benchmark with a nonpolynomial transformed reference."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LENS_THREADS', '4')

import lens
import numpy as np
import torch

from lens import env, fit, regions, transport_cases


class ShiftedSuperellipse(regions.ParamRegion):
    """The legacy order-four superellipse translated into the unit square."""

    name = 'superellipse4'
    bbox = (0., 1., .1, .9)

    def __init__(self):
        self.centered = transport_cases.SuperellipseRegion(a=.5, b=.4)

    def _xy(self, th):
        x, y = self.centered._xy(th)
        return x + .5, y + .5

    def contains(self, x, y):
        return self.centered.contains(np.asarray(x) - .5, np.asarray(y) - .5)


def source(x):
    return (1 + np.asarray(x) / 2) ** -4


def omega(x, t, log):
    a = 1 + x / 2
    b = 3 / 4 + x / a
    primitive = (2 / 3) * (a ** -2 - 1 + x)
    return primitive - t * (t + 1) / 2 - a * log(b * b + t * t) / 16


def exact(x, y):
    """Evaluate u and its gradient by scalar inversion on 0 <= y <= 1."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    a = 1 + x / 2
    b = 3 / 4 + x / a
    low = np.full_like(y, -.5)
    high = np.full_like(y, .5)
    for _ in range(55):
        mid = (low + high) / 2
        below = .5 + mid + a * mid / (8 * (b * b + mid * mid)) < y
        low = np.where(below, mid, low)
        high = np.where(below, high, mid)
    t = (low + high) / 2
    denom = b * b + t * t
    ux = (2 / 3) * (1 - a ** -3) - np.log(denom) / 32 - b / (8 * a * denom)
    return omega(x, t, np.log) + y * t, ux, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'out/nonfamily_reproduction')
    parser.add_argument('--regions', nargs='+', choices=('S','D','T','U','E'), default=('S','D','T','U'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'results.jsonl'

    x, t = torch.meshgrid(torch.linspace(0, 1, 33), torch.linspace(-.6, .6, 33), indexing='ij')
    x.requires_grad_()
    t.requires_grad_()
    wx, wt = torch.autograd.grad(omega(x, t, torch.log).sum(), (x, t), create_graph=True)
    wxx = torch.autograd.grad(wx.sum(), x, retain_graph=True)[0]
    wtt = torch.autograd.grad(wt.sum(), t)[0]
    reference_residual = float((wxx + (1 + x / 2) ** -4 * wtt).abs().max().detach())
    assert reference_residual < 1e-13
    audit_x, audit_y = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
    _, _, audit_t = exact(audit_x, audit_y)
    audit_a = 1 + audit_x / 2
    audit_b = 3 / 4 + audit_x / audit_a
    y_roundtrip = .5 + audit_t + audit_a * audit_t / (8 * (audit_b**2 + audit_t**2))
    reference_inverse_error = float(np.max(abs(y_roundtrip - audit_y)))
    assert reference_inverse_error < 2e-15

    names = {'S': 'square', 'D': 'disk', 'T': 'triangle', 'U': 'superellipse4', 'E': 'ellipse'}
    cases = {code: names[code] for code in args.regions}
    files = [Path(__file__), *sorted((ROOT / 'lens').glob('*.py'))]
    protocol = dict(source='reciprocal_quartic', regions=list(cases), K=16, m=256,
                    lambda_joint=.4, delta=.1, iota=.02, stages=[60, 60, 200], mu=[0, 1e-8, 0],
                    grid=65, pde_grid=65, threads=lens.THREADS,
                    boundary_metric='absolute RMS of reconstructed field at the m boundary sites',
                    reference_max_linear_residual=reference_residual,
                    reference_max_inverse_error=reference_inverse_error,
                    hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    meta = out / 'protocol.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Protocol changed; use a new output directory')
    meta.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    done = {json.loads(line)['run_id'] for line in log.read_text(encoding='utf-8').splitlines()} if log.exists() else set()
    for code, domain in cases.items():
        run_id = f'rhs_{code}_reciprocal_quartic_K16_m256'
        if run_id in done:
            continue
        print(f'BEGIN {run_id}', flush=True)
        region = ShiftedSuperellipse() if code == 'U' else transport_cases.make_region('P' if code == 'T' else code)
        solver = fit.DirichletFit(region, source, 16, 256, truth=exact, basis_kind='fx').build().fit()
        measured = solver.measure()
        boundary = solver.chart.field(solver.xb, solver.yb, solver.theta, t0=solver.t)
        error = boundary['u'].numpy() - solver.g
        measured.update(E2B=float(np.sqrt(np.mean(error**2))), EinfB=float(np.max(abs(error))),
                        boundary_n_out=boundary['n_out'])
        margin = float(solver.theta[-1] - (solver.theta[:-1] @ solver.basis.cert_S(65)).max())
        status = 'admissible' if measured['n_out'] == measured['boundary_n_out'] == 0 and margin >= .1 - 1e-10 else 'invalid'
        record = dict(run_id=run_id, group='rhs', method='LENS', domain=domain, region_code=code,
                      source='reciprocal_quartic', K=16, m=256, metrics=measured, timing=solver.timings,
                      iters=solver.iters, dense_margin=margin, box=solver.basis.box, status=status)
        np.savez_compressed(out / (run_id + '.npz'), theta=solver.theta.numpy(), t=solver.t.numpy(), box=solver.basis.box)
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print('RESULT', json.dumps(record), flush=True)
    (out / 'environment.json').write_text(json.dumps(env.env_info(lens.THREADS), indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
