"""Differentiate the discrete forward solve and test the source-only update."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lens
import numpy as np
from refs.fdm_inverse import FDMSourceInverse


def main():
    n = 32
    angles = 2*np.pi*(np.arange(64)+1/3)/64
    normal = np.column_stack((np.cos(angles), np.sin(angles)))
    v = np.sqrt(1+.3**2)
    matrices = [np.eye(2), np.diag([v+.3, v-.3]), np.array([[v, .3], [.3, v]])]
    observed = np.array([.5*np.einsum('ij,jk,ik->i', normal, matrix, normal) for matrix in matrices])
    problem = FDMSourceInverse(observed, n=n, degree=16, rho=1e-7, forward_tol=1e-10)
    eta = np.zeros(17)
    eta[0] = 1
    initial = problem.evaluate(eta)
    np.testing.assert_allclose(initial['q'], observed, atol=3e-9, rtol=0)
    eta[1:4] = [.03, -.02, .01]
    state = problem.evaluate(eta)
    rng = np.random.default_rng(987)
    for direction in (rng.standard_normal(17), np.eye(17)[16]):
        direction /= np.linalg.norm(direction)
        epsilon = 1e-4
        plus = problem.evaluate(eta+epsilon*direction, state['fields'], jacobian=False)
        minus = problem.evaluate(eta-epsilon*direction, state['fields'], jacobian=False)
        fd = (plus['residual']-minus['residual'])/(2*epsilon)
        analytic = state['jacobian'] @ direction
        relative = np.linalg.norm(analytic[:192]-fd[:192])/np.linalg.norm(analytic[:192])
        assert relative < 3e-5, relative
        np.testing.assert_allclose(analytic[192:], fd[192:], rtol=1e-7, atol=1e-10)
        print('Jacobian relative error', relative)
    quadratic = np.zeros(17)
    quadratic[2] = 1
    np.testing.assert_allclose(np.linalg.norm(problem.roughness @ quadratic)**2, 256., rtol=1e-13)
    truth = np.zeros(5)
    truth[:3] = [1., .08, -.05]
    small = FDMSourceInverse(observed, n=n, degree=4, forward_tol=1e-10)
    synthetic = small.evaluate(truth)['q']
    inverse = FDMSourceInverse(synthetic, n=n, degree=4, forward_tol=1e-10)
    recovered = inverse.solve(max_steps=8)
    assert recovered['loss'] < recovered['history'][0]['loss']*1e-6
    assert np.linalg.norm(recovered['eta']-truth) < 1e-4
    assert all(row['min_eigenvalue'] > 0 and row['residual_inf'] < 1e-10 for row in recovered['forward'])
    assert inverse.source_feasible(recovered['eta'])
    c = .05
    v = np.sqrt(c+.3**2)
    matrices = [np.eye(2)*np.sqrt(c), np.diag([v+.3, v-.3]), np.array([[v, .3], [.3, v]])]
    lower_data = np.array([.5*np.einsum('ij,jk,ik->i', normal, matrix, normal) for matrix in matrices])
    constrained = FDMSourceInverse(lower_data, n=n, degree=2, forward_tol=1e-10)
    lower = constrained.solve(max_steps=4)
    assert lower['status']=='small_step', lower['status']
    np.testing.assert_allclose(constrained.source_check @ lower['eta'], .1, atol=1e-9)
    print('PASS: quadratic controls, discrete sensitivities, physical roughness, active bounds, and source recovery')


if __name__ == '__main__':
    main()
