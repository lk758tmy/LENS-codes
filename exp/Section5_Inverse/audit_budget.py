"""Recheck final saved inverse states, independently of fitting clocks."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import lens
import numpy as np
import torch
from lens.inverse_fit import InverseProblem
from lens.inverse_diagnostics import audit, discrepancy
from refs.fdm_inverse import FDMSourceInverse
from exp.Section5_Inverse.run_budget import scorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path(__file__).parent/'out/budget200_2026-11-19')
    args=parser.parse_args()
    paths=sorted(args.data_root.glob('*_s*.json'))
    assert len(paths)==24,len(paths)
    results=[]
    fdm_curves={}
    for path in paths:
        rec=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as data:
            np.testing.assert_array_equal(data['eta'],data[f'eta_{len(rec["curve"])-1:03d}'])
            if rec['method']=='LENS':
                p=InverseProblem(data['observed'],(0,1,2),16,24,tmax=rec['tlimits'],
                                 rho=rec['rho'],exact_weight=10.,grid=97)
                p.curvature_floor=.1
                z=torch.tensor(data[f'z_{len(rec["curve"])-1:03d}'])
                admissible=p.feasible(z)
                training=InverseProblem(data['observed'],(0,1,2),16,24,tmax=rec['tlimits'],
                                        rho=rec['rho'],exact_weight=10.,grid=129)
                train_feasible=training.feasible(z)
                fields=audit(p,z,Path(__file__).parent/'out/reference_v1',rec['legacy_source'])
                actual=discrepancy(p,z)
                assert train_feasible and actual['n_out']==0 and all(v['n_out']==0 for v in fields),path
                assert max(v['source_pde_max'] for v in fields)<1e-7,path
                result=dict(run_id=rec['run_id'],training_129_grid_feasible=train_feasible,
                            additional_97_grid_floor_01=admissible,
                            physical_fields=fields,actual_boundary=actual)
                eta,coefficients=p.unpack(z)
                generator=p.basis.build(eta,False)
                chart_max=[]
                for ell,(theta,gamma) in enumerate(zip(coefficients,p.gamma)):
                    a,_=p.basis.omega_coefficients(generator,theta,gamma,False)
                    chart_max.append(float(((a @ p.Tgrid).T @ p.Pttgrids[ell]).max()))
                result['chart_max_wtt_97']=chart_max
                result['additional_97_strictly_concave']=all(value<0 for value in chart_max)
                assert result['additional_97_strictly_concave'],(path,chart_max)
            else:
                assert all(v['min_eigenvalue']>0 and v['residual_inf']<=2e-10 for v in rec['forward']),path
                assert .1-1e-9<=rec['source_min']<=rec['source_max']<=3+1e-9,path
                result=dict(run_id=rec['run_id'],convex_discrete_fields=True,forward=rec['forward'])
                # Most FDM updates occur before the first 5-second full-state
                # interval. Reconstruct each accepted discrete field from the
                # saved source iterate, after all fitting clocks have stopped.
                # The final replay must agree with the saved original field.
                p=FDMSourceInverse(data['observed'],n=rec['FDM_n'],degree=16,
                                   rho=rec['rho'],forward_tol=2e-10)
                score,_,_=scorer(rec['source'])
                current=None
                curve=[]
                for h in rec['history']:
                    if h['elapsed']>rec['budget_seconds'] or (h['iteration'] and not h.get('accepted')):continue
                    current=p.evaluate(np.array(h['eta']),None if current is None else current['fields'],jacobian=False)
                    assert abs(current['loss']-h['loss'])<1e-10,(path,h['iteration'])
                    response=p.validation_responses(current)
                    point=dict(elapsed=h['elapsed'],iteration=h['iteration'],n_out=0,**score(current['eta'],response))
                    curve.append(point)
                saved=data[f'q_validation_{len(rec["curve"])-1:03d}']
                replay_error=float(np.max(abs(response-saved)))
                assert replay_error<1e-7,(path,replay_error)
                result['final_replay_max_q_difference']=replay_error
                fdm_curves[rec['run_id']]=curve
        results.append(result)
        print('PASS',rec['run_id'],flush=True)
    assert len(results)==24,len(results)
    (args.data_root/'final_audit.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    (args.data_root/'fdm_iteration_curves.json').write_text(json.dumps(dict(
        provenance='Post-fit forward evaluation of every saved accepted FDM source iterate; original elapsed times retained; final responses checked against original saved fields.',
        curves=fdm_curves),indent=2)+'\n',encoding='utf-8')
    print('PASS: 24 saved final states; truth did not enter fitting or final-state selection.')


if __name__=='__main__':main()
