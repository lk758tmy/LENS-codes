"""Compare LENS with Bonnet--Mirebeau's monotone OT finite differences."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np

HERE = Path(__file__).resolve().parent
SQUARE_RECONSTRUCTION = 'second-order grid gradients with linear extrapolation to the boundary'


def boundary_condition(n, boundary, *, disk_radius=None):
    from agd.AutomaticDifferentiation import remove_ad

    # Sample the support function of the prescribed target boundary.
    angles = np.arange(4*n) * (2*np.pi/(4*n))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    directions[np.abs(directions) < 1e-15] = 0
    supports = (directions @ boundary.T).max(axis=1) if disk_radius is None else np.full(len(directions), disk_radius)
    negative = np.maximum(-directions, 0)
    positive = np.maximum(directions, 0)

    def evaluate(x, r, p_plus, p_minus):
        # Differentiate the maximizing branch, as in the upstream AD maximum.
        # Select the branch in ordinary arrays to bound AD memory.
        plus = remove_ad(p_plus)
        minus = remove_ad(p_minus)
        shape = plus.shape[1:]
        slopes = -np.concatenate((plus, minus), axis=0).reshape(4, -1)
        active = np.zeros(slopes.shape[1], dtype=int)
        best = np.full(slopes.shape[1], -np.inf)
        weights = np.column_stack((negative, positive))
        # Zero coefficients ignore a direction whose neighbor is outside.
        for start in range(0, len(weights), 32):
            w = weights[start:start+32]
            score = -np.broadcast_to(supports[start:start+32, None], (len(w), len(best))).copy()
            for j in range(4):
                rows = w[:, j] > 0
                finite = np.where(np.isfinite(slopes[j]), slopes[j], -np.inf)
                score[rows] += w[rows, j, None] * finite[None, :]
            selected = score.argmax(axis=0)
            values = score[selected, np.arange(len(best))]
            better = values > best
            active[better] = start + selected[better]
            best[better] = values[better]
        valid = np.isfinite(best).reshape(shape)
        result = -supports[active].reshape(shape)
        for j, slope in enumerate((p_plus[0], p_plus[1], p_minus[0], p_minus[1])):
            w = weights[active, j].reshape(shape)
            result = result - w * np.where((w > 0) & valid, slope, 0.0)
        return 20 * np.where(valid, result, -np.inf)

    return evaluate


def norm_errors(u, p, exact_u, exact_p):
    # Scalar gauge and relative norms agree with the LENS benchmark.
    u_error = u-exact_u
    u_error -= u_error.mean()
    map_error = np.linalg.norm(p-exact_p, axis=0)
    return dict(E2=float(np.linalg.norm(u_error)/np.linalg.norm(exact_u-exact_u.mean())),
                E_grad=float(np.linalg.norm(map_error)/np.linalg.norm(exact_p)),
                map_error_max=float(map_error.max()))


def solve_fdm(n, domain, source, boundary=None, *, disk_radius=None):
    from agd import Domain
    import ma

    start = time.perf_counter()
    nodes = np.linspace(-.5, .5, n+1)
    x = np.stack(np.meshgrid(nodes, nodes, indexing='ij'))
    stencil = ma.StencilForConditioning(15)
    F = boundary_condition(n, boundary, disk_radius=disk_radius)

    def A(x, u, du):
        return np.zeros((2, 2) + x.shape[1:])

    def B(x, u, du):
        return source(x[0] + .5)

    initial = .5*np.sum(x*x, axis=0)
    before_solve = time.perf_counter()
    # Python 3.14 cannot deepcopy the default itertools.chain of log indices.
    stopper = ma.Optimization.stop_default(niter_max=100, niter_print=tuple(range(1, 102)))
    with patch.object(ma.Optimization, 'stop_default', new=lambda **_: stopper), np.errstate(invalid='ignore'):
        u, alpha, iterations = ma.NewtonRootBV2(initial, x, domain, A, B, 0., F, stencil)
    after_solve = time.perf_counter()
    norm = float(stopper.residue_norms[-1])
    assert np.isfinite(norm) and norm < 1.1e-8, norm
    bc = Domain.Dirichlet(domain, np.inf, x)
    result = dict(n=n, spacing=1/n, unknown_nodes=int(bc.interior.sum()),
                  stencil_condition=15, angular_directions=4*n,
                  iterations=iterations, residual_inf=norm, compatibility_alpha=float(alpha),
                  seconds=dict(build=before_solve-start, solve=after_solve-before_solve))
    return x, u, bc, result


def run(n, out, boundary, shift, lens_chart, lens_theta, source, exact):
    from agd import Domain
    from lens import metrics

    x, u, bc, result = solve_fdm(n, Domain.Ball(radius=.5), source, boundary-shift)
    offsets = np.eye(2, dtype=int)
    lap = bc.Diff2(u, offsets).sum(axis=0)
    valid = bc.interior & np.isfinite(lap)
    plus = bc.DiffUpwind(u, offsets)[:, valid]
    minus = bc.DiffUpwind(u, -offsets)[:, valid]
    p = .5*(plus-minus) + shift[:, None]
    physical_u = u[valid] + shift @ x[:, valid]
    a, b = x[:, valid] + .5
    exact_u, exact_px, exact_py = exact(a, b)
    exact_p = np.stack([exact_px, exact_py])
    lens = metrics.field_on_grid(lens_chart, lens_theta, a, b)
    assert lens['n_out'] == 0
    fdm = norm_errors(physical_u, p, exact_u, exact_p)
    lens_errors = norm_errors(lens['u'], np.stack([lens['ux'], lens['uy']]), exact_u, exact_p)
    result.update(common_eval_nodes=int(valid.sum()), support_samples=len(boundary),
                  fdm=fdm, lens=lens_errors)
    np.savez_compressed(out/f'fields_n{n}.npz', x=x[:, valid], exact_u=exact_u, exact_p=exact_p,
                        fd_u=physical_u, fd_p=p, lens_u=lens['u'], lens_p=np.stack([lens['ux'], lens['uy']]))
    (out/f'fdm_n{n}.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


def image_errors(px, py, tx, ty, radius, transport_fit):
    """Geometry against the exact disk; the returned image is an ordered polyline."""
    area = .5*np.sum(px*np.roll(py, -1) - py*np.roll(px, -1))
    target_area = np.pi*radius**2
    scale = np.sqrt(target_area)
    radial = np.abs(np.hypot(px, py)-radius)
    backward = transport_fit.distance_polyline(tx, ty, px, py)
    return dict(dH=float(max(radial.max(), backward.max())/scale),
                relative_area_error=float(abs(area-target_area)/target_area),
                signed_area=float(area), forward_max=float(radial.max()/scale),
                backward_max=float(backward.max()/scale))


def square_boundary_gradient(u, nodes, bx, by):
    from scipy.interpolate import RegularGridInterpolator

    gradients = np.stack(np.gradient(u, nodes, nodes, edge_order=2), axis=-1)
    evaluate = RegularGridInterpolator((nodes, nodes), gradients, bounds_error=False, fill_value=None)
    return evaluate(np.column_stack((bx, by))).T


def square_to_disk(repo, out, sizes):
    import hashlib
    from agd import Domain
    from lens import metrics, transport_cases as cases, transport_fit
    import torch

    release = repo/'exp/release_2026-11-05/transport_K24'
    case_id = 'B_S_D_f1'
    frozen = next(row for row in map(json.loads, (release/'results.jsonl').read_text().splitlines())
                  if row['run_id']==case_id)
    K, m = (frozen['case'][key] for key in ('K', 'm'))
    region, source = cases.make_region('S'), cases.make_source(1)
    target = cases.make_target('D', cases.source_moments(region, source)[0])
    radius = np.sqrt(target.area/np.pi)
    bx, by = cases.boundary(region, 2048, vertices=True)
    tx, ty = target.boundary(2048, include_vertices=True)
    out.mkdir(parents=True, exist_ok=True)
    domain = Domain.Box(((-.5, .5), (-.5, .5)))
    rows = []
    for n in sizes:
        x, u, bc, row = solve_fdm(n, domain, source, disk_radius=radius)
        interior = u[1:-1, 1:-1]
        assert bc.interior.sum()==interior.size and bc.interior[1:-1, 1:-1].all()
        nodes = x[0, 1:-1, 0]+.5
        px, py = square_boundary_gradient(interior, nodes, bx, by)
        assert np.isfinite(px).all() and np.isfinite(py).all()
        row['fdm'] = image_errors(px, py, tx, ty, radius, transport_fit)
        row['fdm']['self_crossing'] = bool(transport_fit.self_crosses(px, py))
        rows.append(row)
        np.savez_compressed(out/f'fields_n{n}.npz', nodes=nodes, u=interior, bx=bx, by=by, px=px, py=py)
        (out/f'fdm_n{n}.json').write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(row), flush=True)

    start = time.perf_counter()
    fit = transport_fit.TransportFit(region, target, K=K, n_edge=m, source=source).build().fit()
    elapsed = time.perf_counter()-start
    model_path = release/f'checkpoints/{case_id}.npz'
    with np.load(model_path) as model:
        theta = torch.tensor(model['theta'])
    np.testing.assert_array_equal(fit.box, frozen['box'])
    field = metrics.field_on_grid(fit.chart, theta, bx, by)
    repeated = metrics.field_on_grid(fit.chart, fit.theta, bx, by)
    assert field['n_out']==0 and repeated['n_out']==0
    for key in ('ux', 'uy'):
        np.testing.assert_allclose(repeated[key], field[key], rtol=1e-7, atol=1e-8)
    px, py = (field[key] for key in ('ux', 'uy'))
    lens_errors = image_errors(px, py, tx, ty, radius, transport_fit)
    np.testing.assert_allclose(lens_errors['dH'], frozen['metrics']['dH'], rtol=1e-10, atol=1e-12)
    np.savez_compressed(out/'lens_boundary.npz', bx=bx, by=by, px=px, py=py, tx=tx, ty=ty)
    result = dict(case=case_id, baseline_commit='d84acdc3d18614a174ff6b4b39b3d3aaa01396ee',
                  agd_version=version('agd'), target_radius=float(radius), boundary_samples=len(bx),
                  fdm_reconstruction=SQUARE_RECONSTRUCTION,
                  fdm_support='analytic disk', lens=dict(K=K, m=m, iterations=fit.iters,
                  setup_and_fit_seconds=elapsed, metrics=lens_errors, **fit.timings),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['lens']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A_f1', 'S_D_f1'), default='A_f1')
    parser.add_argument('--n', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--upstream', type=Path, required=True,
                        help='Checkout of Bonnet--Mirebeau ma-ot-monotone')
    parser.add_argument('--agd-dir', type=Path, help='Optional directory containing the agd package')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    repo = HERE.parents[1]
    upstream = args.upstream.resolve()/'main'
    if not (upstream/'ma.py').is_file():
        parser.error('--upstream must point to the checkout containing main/ma.py')
    sys.path[:0] = [str(repo), str(upstream)]
    if args.agd_dir:
        sys.path.insert(0, str(args.agd_dir.resolve()))
    out = (args.out or HERE/f'out/fdm_{args.case}_reproduction').resolve()
    if args.case=='S_D_f1':
        if any(n<8 or n%2 for n in args.n):
            parser.error('S_D_f1 requires even n >= 8')
        square_to_disk(repo, out, args.n)
        return
    from lens import cert, chart, transport_cases as cases, transport_fit
    import torch

    release = repo/'exp/release_2026-11-05/transport_K16'
    frozen = next(r for r in map(json.loads, (release/'results.jsonl').read_text().splitlines())
                  if r['run_id'] == 'A_f1')
    order, box = 16, frozen['box']
    exact = cases.Manufactured(1)
    curve = exact.curve
    source = cases.make_source(1)
    model_path = release/'checkpoints/A_f1.npz'
    basis = transport_fit.TransportFit(cases.make_region('D'), exact.target(), K=order, source=source).build()._basis(order)
    np.testing.assert_array_equal(basis.box, box)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(model_path) as model:
        lens_theta = torch.tensor(model['theta'])
    lens_chart = chart.Chart(basis, cert.Certificate(basis.cert_S(61), delta=.1, iota=.02))
    theta = torch.arange(8192)*2*torch.pi/8192
    p, q, *_ = curve(theta)
    boundary = np.column_stack([p.numpy(), q.numpy()])
    # Translation changes the potential by an affine term and is undone before evaluation.
    shift = (boundary.min(axis=0) + boundary.max(axis=0)) / 2
    print('agd', version('agd'), 'target shift', shift.tolist(), flush=True)
    for n in args.n:
        run(n, out, boundary, shift, lens_chart, lens_theta, source, exact)

    start = time.perf_counter()
    target = exact.target()
    fit = transport_fit.TransportFit(cases.make_region('D'), target, K=order, n_edge=256, source=source).build().fit()
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(fit.theta.numpy(), lens_theta.numpy(), rtol=1e-10, atol=1e-11)
    rows = [json.loads((out/f'fdm_n{n}.json').read_text(encoding='utf-8')) for n in args.n]
    result = dict(case='A_f1', baseline_commit='d84acdc3d18614a174ff6b4b39b3d3aaa01396ee',
                  agd_version=version('agd'), lens=dict(K=order, m=256, iterations=fit.iters,
                  setup_and_fit_seconds=elapsed, **fit.timings), rows=rows)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['lens']), flush=True)


if __name__ == '__main__':
    main()
