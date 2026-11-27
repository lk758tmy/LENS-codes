"""Same observations, source space, fixed rho and elapsed-time budget for two solvers.

Reference fields and withheld responses are read only by post-fit scoring.
Accepted iterates are retained without selecting by their true errors.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('LENS_THREADS', '4')
import lens
import numpy as np
import torch
from numpy.polynomial import chebyshev as cheb
from lens import env
from lens.inverse_fit import InverseProblem, solve_joint, prolong
from lens.inverse_data import InverseSource, observation_angles
from refs.fdm_inverse import FDMSourceInverse

_spec = importlib.util.spec_from_file_location('work_profile', ROOT/'exp/Section4.4_Transport/work_profile.py')
_profile = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_profile)
power_state = _profile.power_state

LABELS = {'A': 'C', 'B': 'A'}
REFERENCE = Path(__file__).parent/'out/reference_v1'


def dump(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def training_data(label, noise, seed):
    clean = []
    for ell in range(3):
        with np.load(REFERENCE/f'{LABELS[label]}_g{ell}_n640.npz') as data:
            clean.append(data['q'][:64])
    clean = np.array(clean)
    return clean+noise*np.sqrt(np.mean(clean**2, axis=1))[:, None]*np.random.default_rng(seed).standard_normal(clean.shape)


def scorer(label):
    # This function is called only after the fitting clock has stopped.
    validation = []
    for ell in range(3):
        with np.load(REFERENCE/f'{LABELS[label]}_g{ell}_n640.npz') as data:
            validation.append(data['q'][64:])
    validation = np.array(validation)
    points, weights = np.polynomial.legendre.leggauss(256)
    truth = InverseSource(LABELS[label])((points+1)/2)
    x = np.linspace(0, 1, 1001)
    true_grid = InverseSource(LABELS[label])(x)

    def score(eta, response):
        estimate = cheb.chebval(points, eta)
        recovered = cheb.chebval(2*x-1, eta)
        q_each = np.linalg.norm(response-validation, axis=1)/np.linalg.norm(validation, axis=1)
        return dict(source_E2=float(np.sqrt(np.sum(weights*(estimate-truth)**2)/np.sum(weights*truth**2))),
                    source_Einf=float(np.max(abs(recovered-true_grid))/np.max(abs(true_grid))),
                    E_q_val=float(np.sqrt(np.mean(q_each**2))))
    return score, x, true_grid


def run_lens(observed, budget, rho, interval, progress):
    start = time.perf_counter()
    deadline = start+budget
    snapshots, history, problems = [], [], {}
    previous = state = limits = None
    latest = None
    total_iterations = 0

    def capture(p, z, tau, rec, force=False):
        nonlocal latest
        elapsed = time.perf_counter()-start
        if elapsed <= budget and (rec['iteration']==0 or rec.get('accepted')):
            latest = dict(elapsed=elapsed, M=p.M, K=p.K, z=z.numpy().copy(),
                          tau=tau.numpy().copy(), eta=p.unpack(z)[0].numpy().copy(),
                          loss=rec['loss'], iteration=total_iterations+rec['iteration'])
        if latest is not None and (force or not snapshots or latest['elapsed']-snapshots[-1]['elapsed']>=interval):
            if not snapshots or latest['elapsed']>snapshots[-1]['elapsed']:
                snapshots.append(latest)
                progress({key:latest[key] for key in ('elapsed','M','K','loss','iteration')})

    for stage, (M, K) in enumerate(((0, 4), (4, 8), (8, 12), (16, 24))):
        stage_start = time.perf_counter()
        p = InverseProblem(observed, (0, 1, 2), M, K, lam=.4, rho=rho,
                           mu=1e-8, tmax=limits, exact_weight=10., grid=129)
        problems[M, K] = p
        if state is None:
            z, tau = p.initial()
            limits = p.tlimits
        else:
            z, tau = prolong(previous, p, *state)
        stage_deadline = min(deadline, stage_start+min(20., budget/10))
        z, tau, rows = solve_joint(p, z, tau, rounds=80, deadline=stage_deadline,
                                  state_callback=capture)
        history.extend([dict(stage=stage, M=M, K=K, **row) for row in rows])
        total_iterations += len(rows)
        state, previous = (z, tau), p
        capture(p, z, tau, dict(iteration=0, loss=p.evaluate(z, tau, False)['loss']), True)
        progress(dict(stage=stage, M=M, K=K, iterations=total_iterations,
                      elapsed=time.perf_counter()-start))
    p.mu = 0.
    z, tau, rows = solve_joint(p, z, tau, rounds=100000, deadline=deadline,
                              ftol=None, state_callback=capture)
    history.extend([dict(stage=4, M=p.M, K=p.K, **row) for row in rows])
    total_iterations += len(rows)
    capture(p, z, tau, dict(iteration=-1), True)
    fit_seconds = time.perf_counter()-start
    # If the last completed state fell between observer intervals, retain it.
    eta = p.unpack(z)[0].numpy().copy()
    if not snapshots or not np.array_equal(snapshots[-1]['z'], z.numpy()):
        assert latest is not None and np.array_equal(latest['z'], z.numpy())
        snapshots.append(latest)
    status = 'time_limit' if fit_seconds >= budget else 'numerical_stop'
    # No training data or objective is evaluated below this point.
    _, _, angles = observation_angles()
    for snapshot in snapshots:
        model = problems[snapshot['M'], snapshot['K']]
        metrics = model.field_metrics(torch.tensor(snapshot['z']), angles)
        snapshot['q_validation'] = np.array([m['q'] for m in metrics])
        snapshot['n_out'] = sum(m['n_out'] for m in metrics)
    return dict(status=status, fit_seconds=fit_seconds, history=history, snapshots=snapshots,
                iterations=total_iterations, eta=eta, tlimits=limits,
                source_min=float((p.unpack(z)[0] @ p.Tsource).min()),
                source_max=float((p.unpack(z)[0] @ p.Tsource).max()))


def run_fdm(observed, budget, rho, interval, n, progress):
    start = time.perf_counter()
    problem = FDMSourceInverse(observed, n=n, degree=16, rho=rho, forward_tol=2e-10)
    problem.started = start
    snapshots, latest = [], None

    def capture(current, rec):
        nonlocal latest
        if rec['elapsed']<=budget and (rec['iteration']==0 or rec.get('accepted')):
            latest = dict(elapsed=rec['elapsed'], eta=current['eta'].copy(),
                          fields=np.array(current['fields']), loss=current['loss'],
                          iteration=rec['iteration'])
            if not snapshots or latest['elapsed']-snapshots[-1]['elapsed']>=interval:
                snapshots.append(latest)

    result = problem.solve(max_steps=100000, seconds=budget, state_callback=capture,
                           callback=lambda r: progress({k:v for k,v in r.items() if k!='eta'}))
    fit_seconds = time.perf_counter()-start
    if latest is None:
        raise RuntimeError('Initial feasible state not ready within the budget')
    if not snapshots or snapshots[-1]['iteration']!=latest['iteration']:
        snapshots.append(latest)
    for snapshot in snapshots:
        snapshot['q_validation'] = problem.validation_responses(snapshot)
        snapshot['n_out'] = 0
    values = problem.source_check @ result['eta']
    return dict(status=result['status'], fit_seconds=fit_seconds, history=result['history'],
                snapshots=snapshots, iterations=len(result['history'])-1, eta=result['eta'],
                source_min=float(values.min()), source_max=float(values.max()),
                forward=result['forward'], build_seconds=problem.build_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=200.)
    parser.add_argument('--n', type=int, default=160)
    parser.add_argument('--methods', default='LENS,FDM')
    parser.add_argument('--sources', default='A,B')
    parser.add_argument('--noises', default='0,0.01')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--interval', type=float, default=5.)
    args = parser.parse_args()
    if args.seconds<=0 or args.interval<=0:parser.error('Positive budget and interval required')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), *list((ROOT/'lens').glob('inverse_*.py')),
             ROOT/'refs/fdm_inverse.py', ROOT/'refs/fdm.py',
             ROOT/'lens/constrained_lm.py', ROOT/'lens/env.py', ROOT/'lens/__init__.py',
             ROOT/'exp/Section4.4_Transport/work_profile.py']
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    references = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in REFERENCE.glob('*_n640.npz')}
    protocol = dict(version=1, seconds=args.seconds, interval=args.interval, FDM_n=args.n,
        source_degree=16, LENS_K=24, noise_rho={'0':0., '0.01':1e-9},
        rho_rule='zero for noiseless, otherwise fixed 1e-9; no scan or discrepancy gate',
        source_initial='1', source_bounds=[.1,3.],
        normal_sensors=64, dirichlet_points=192, validation_points=257, loads=3,
        lens_warmup_stages=[[0,4],[4,8],[8,12],[16,24]], warmup_steps=80,
        warmup_stage_max_seconds=min(20.,args.seconds/10), temporary_field_mu=1e-8,
        final_mu=0., final_ftol=None, final_selection='last completed feasible iterate',
        timing='construction and fitting; exclude imports, shared reference IO and post-fit evaluation',
        deadline='keep last completed accepted iterate at or before budget; current atomic operation may finish later',
        early_stop='retain numerically stationary/stalled final state for remaining budget',
        curve='accepted states at >=5s intervals plus stage/final states; true errors evaluated afterward',
        environment=env.env_info(lens.THREADS), hashes=hashes, references=references)
    meta = out/'protocol.json'
    if meta.exists() and json.loads(meta.read_text())!=protocol:
        raise RuntimeError('Use a fresh directory for another protocol or code version')
    dump(meta, protocol)
    for label in args.sources.split(','):
        for noise in map(float,args.noises.split(',')):
            for seed in ([0] if noise==0 else map(int,args.seeds.split(','))):
                for method in args.methods.split(','):
                    rid = f'{method}_{label}_e{noise:g}_s{seed}'
                    path = out/f'{rid}.json'
                    if path.exists():
                        print('SKIP',rid,flush=True)
                        continue
                    power_before = power_state()
                    if power_before.get('available') and power_before.get('ac_line_status')!=1:
                        raise RuntimeError('AC power required for elapsed-time comparison; saved completed runs are resumable')
                    observed = training_data(label,noise,seed)
                    rho = 0. if noise==0 else 1e-9
                    def progress(rec):
                        dump(out/'progress.json',dict(case=rid,**rec))
                        print('PROGRESS',rid,json.dumps(rec),flush=True)
                    print('BEGIN',rid,flush=True)
                    result = (run_lens(observed,args.seconds,rho,args.interval,progress) if method=='LENS'
                              else run_fdm(observed,args.seconds,rho,args.interval,args.n,progress))
                    score,x,truth = scorer(label)
                    states = result.pop('snapshots')
                    records,arrays = [], dict(observed=observed,x=x,true_source=truth)
                    for index,state in enumerate(states):
                        record = {k:v for k,v in state.items() if k not in ('eta','z','tau','fields','q_validation')}
                        record.update(score(state['eta'],state['q_validation']))
                        records.append(record)
                        for key in ('eta','z','tau','fields','q_validation'):
                            if key in state:arrays[f'{key}_{index:03d}'] = state[key]
                    eta = result.pop('eta')
                    arrays.update(eta=eta,source=cheb.chebval(2*x-1,eta))
                    result.update(run_id=rid,source=label,legacy_source=LABELS[label],method=method,
                        noise=noise,seed=seed,rho=rho,budget_seconds=args.seconds,
                        FDM_n=args.n if method=='FDM' else None,eta=eta.tolist(),
                        power_before=power_before,power_after=power_state(),curve=records,
                        metrics={key:records[-1][key] for key in ('source_E2','source_Einf','E_q_val')})
                    np.savez_compressed(out/f'{rid}.npz',**arrays)
                    dump(path,result)
                    print('RESULT',rid,json.dumps(result['metrics']),result['status'],flush=True)
    dump(out/'progress.json',dict(status='completed'))
    print('DONE',flush=True)


if __name__=='__main__':main()
