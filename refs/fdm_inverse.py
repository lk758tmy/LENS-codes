"""Source-only Gauss--Newton inversion with convex finite-difference fields."""
from time import perf_counter

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
import torch

from lens.constrained_lm import solve_qp
from lens.inverse_data import Control, DiskFlux, R, observation_angles
from refs.fdm import CurvedFD


class FDMSourceInverse:
    """Fit one Chebyshev source to the normal responses of fixed inputs.

    Only the 64 observed responses per input enter the objective. Each field
    satisfies the discrete Dirichlet problem at every accepted source. The
    response Jacobian differentiates that same discrete system and the linear
    boundary-flux reconstruction, sharing one sparse factorization per input.
    """

    def __init__(self, observed, n=40, degree=16, rho=0., forward_tol=2e-8):
        self.started = perf_counter()
        self.observed = np.asarray(observed, dtype=float)
        if self.observed.shape != (3, 64) or not np.isfinite(self.observed).all():
            raise ValueError('Expected finite observations for three inputs and 64 sensors')
        if n < 24 or degree < 0 or rho < 0:
            raise ValueError('Require n >= 24, degree >= 0, and rho >= 0')
        self.n, self.degree, self.rho = int(n), int(degree), float(rho)
        self.forward_tol = float(forward_tol)
        self.scale = np.maximum(np.sqrt(np.mean(self.observed**2, axis=1)), 1e-6)
        self.controls = [Control(index) for index in range(3)]
        inside = lambda x, y: (np.asarray(x)-.5)**2 + (np.asarray(y)-.5)**2 <= R**2*(1+1e-13)
        self.problems = [CurvedFD(0, 1, 0, 1, n, n, inside, control.value,
                                 lambda x: np.ones_like(x)) for control in self.controls]
        self.initial_fields = []
        for problem in self.problems:
            problem.initial()
            self.initial_fields.append(problem.U.copy())
        self.vandermonde = cheb.chebvander(2*self.problems[0].X[self.problems[0].intm]-1, degree)
        self.source_check = cheb.chebvander(np.linspace(-1, 1, 257), degree)
        _, sensors, self.validation_angles = observation_angles()
        self.flux = DiskFlux(self.problems[0], sensors)
        row = np.repeat(np.arange(len(sensors)), [len(indices) for indices in self.flux.indices])
        column = np.concatenate(self.flux.indices)
        weight = np.concatenate(self.flux.weights)
        self.flux_matrix = csr_matrix((weight, (row, column)), shape=(64, self.problems[0].n))
        nodes, weights = np.polynomial.legendre.leggauss(degree+3)
        derivatives = []
        for index in range(degree+1):
            coefficient = np.zeros(degree+1)
            coefficient[index] = 1
            derivatives.append(4*cheb.chebval(nodes, cheb.chebder(coefficient, 2)))
        self.roughness = np.sqrt(weights/2)[:, None]*np.array(derivatives).T
        self.build_seconds = perf_counter()-self.started

    def source_feasible(self, eta):
        values = self.source_check @ eta
        return np.isfinite(values).all() and values.min() >= .1-1e-10 and values.max() <= 3+1e-10

    def evaluate(self, eta, initial_fields=None, jacobian=True):
        eta = np.asarray(eta, dtype=float)
        if eta.shape != (self.degree+1,) or not self.source_feasible(eta):
            raise ValueError('Candidate source violates its sampled bounds')
        if initial_fields is None:
            initial_fields = self.initial_fields
        residuals, jacobians, fields, responses, diagnostics = [], [], [], [], []
        source_values = self.vandermonde @ eta
        for index, (problem, control, initial) in enumerate(zip(self.problems, self.controls, initial_fields)):
            problem.fv[problem.intm] = source_values
            problem.U = initial.copy()
            problem.solve(tol=self.forward_tol)
            if problem.status != 'converged' or not problem.is_convex():
                raise RuntimeError(f'Dirichlet solve failed for input {index}: {problem.status}')
            response = self.flux.evaluate(problem, control)
            response_scale = self.scale[index]*np.sqrt(3*64)
            residuals.append((response-self.observed[index])/response_scale)
            if jacobian:
                factor = splu(problem.jacobian(problem.U))
                sensitivities = factor.solve(self.vandermonde)
                jacobians.append(np.asarray(self.flux_matrix @ sensitivities)/response_scale)
            fields.append(problem.U.copy())
            responses.append(response)
            diagnostics.append(dict(input=index, iterations=problem.iterations,
                                    residual_inf=problem.residual_inf, min_eigenvalue=problem.min_eig()))
        residuals.append(np.sqrt(self.rho)*self.roughness @ eta)
        residual = np.concatenate(residuals)
        result = dict(eta=eta.copy(), fields=fields, q=np.array(responses), residual=residual,
                      loss=float(residual @ residual), forward=diagnostics)
        if jacobian:
            result['jacobian'] = np.vstack([*jacobians, np.sqrt(self.rho)*self.roughness])
        return result

    def solve(self, max_steps=8, seconds=None, callback=None, state_callback=None):
        eta = np.zeros(self.degree+1)
        eta[0] = 1
        current = self.evaluate(eta)
        damping, history = 1e-3, []
        status = 'step_limit'
        history.append(dict(iteration=0, loss=current['loss'], eta=eta.tolist(),
                            elapsed=perf_counter()-self.started))
        if state_callback:state_callback(current,history[-1])
        constraints = np.vstack((self.source_check, -self.source_check))
        for iteration in range(1, max_steps+1):
            if seconds is not None and perf_counter()-self.started >= seconds:
                status = 'time_limit'
                break
            jacobian, residual = current['jacobian'], current['residual']
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ residual
            if np.linalg.norm(gradient, ord=np.inf) < 1e-10:
                status = 'stationary'
                break
            diagonal = np.maximum(np.diag(normal), 1e-8)
            values = self.source_check @ eta
            bounds = np.r_[3-values, values-.1]
            accepted = False
            for trial in range(12):
                if seconds is not None and perf_counter()-self.started >= seconds:
                    status = 'time_limit'
                    break
                hessian = normal + damping*np.diag(diagonal)
                step, _ = solve_qp(torch.as_tensor(hessian), torch.as_tensor(-gradient),
                                    torch.as_tensor(constraints), torch.as_tensor(bounds))
                step = step.numpy()
                if np.linalg.norm(step) <= 1e-10*(1+np.linalg.norm(eta)):
                    status = 'small_step'
                    break
                try:
                    candidate = self.evaluate(eta+step, current['fields'], jacobian=False)
                except (RuntimeError, ValueError):
                    damping *= 10
                    continue
                predicted = -float(2*gradient @ step + step @ normal @ step)
                reduction = current['loss']-candidate['loss']
                if predicted > 0 and reduction > 1e-4*predicted:
                    next_current = self.evaluate(candidate['eta'], candidate['fields'])
                    if seconds is not None and perf_counter()-self.started >= seconds:
                        status = 'time_limit'
                        break
                    eta = candidate['eta']
                    current = next_current
                    damping = max(damping/3, 1e-12)
                    accepted = True
                    break
                damping *= 10
            record = dict(iteration=iteration, loss=current['loss'], accepted=accepted, trials=trial+1,
                          damping=damping, eta=eta.tolist(), elapsed=perf_counter()-self.started)
            history.append(record)
            if state_callback:state_callback(current,record)
            if callback:
                callback(record)
            if not accepted:
                status = status if status in ('time_limit', 'small_step') else 'stalled'
                break
        return dict(status=status, history=history, elapsed=perf_counter()-self.started,
                    build_seconds=self.build_seconds, **current)

    def validation_responses(self, result):
        stencil = DiskFlux(self.problems[0], self.validation_angles)
        responses = []
        for problem, control, field in zip(self.problems, self.controls, result['fields']):
            problem.U = field.copy()
            responses.append(stencil.evaluate(problem, control))
        return np.array(responses)
