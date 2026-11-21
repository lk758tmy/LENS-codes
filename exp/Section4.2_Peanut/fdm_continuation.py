"""Convex FDM with boundary-data continuation after a nonconvex Poisson seed.

No solution values in the interior are queried. The lifting is obtained by a
discrete harmonic solve with the difference of the two known boundary data.
"""
from pathlib import Path
import sys,time,json,copy,hashlib,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import lens,numpy as np
from scipy.sparse.linalg import spsolve
from refs.fdm import CurvedFD
from lens import truth,sources,metrics
from exp.run_release import power_state
from peanut_geometry import make

def solve(region,g,source,n):
    start=time.perf_counter();x0,x1,y0,y1=region.bbox;h=(x1-x0)/n;ny=int(np.ceil((y1-y0)/h-1e-10))
    def build(data):return CurvedFD(x0,x1,y0,y0+ny*h,n,ny,region.contains,data,source)
    goal=build(g);goal.initial();initial=goal.initial_min_eig
    attempts=[]
    if initial>0:
        build_seconds=time.perf_counter()-start;t=time.perf_counter();goal.solve();fit_seconds=time.perf_counter()-t
        return goal,dict(build=build_seconds,fit=fit_seconds),dict(initial_min_eig=initial,fallback=False,attempts=[])
    xb,yb=region.boundary(256);A=np.column_stack([np.ones_like(xb),xb,yb])
    affine=np.linalg.lstsq(A,g(xb,yb)-.5*(xb*xb+yb*yb),rcond=None)[0]
    def quadratic(x,y):return .5*(x*x+y*y)+affine[0]+affine[1]*x+affine[2]*y
    base=build(quadratic)
    # Explicit unit-determinant quadratic, not the manufactured interior truth.
    U=base.gv.copy();U[base.intm]=quadratic(base.X[base.intm],base.Y[base.intm])
    if not base.is_convex(U):raise RuntimeError('Quadratic seed lost discrete convexity')
    db=[getattr(goal,k)-getattr(base,k) for k in ('b_xx','b_yy','b_xy')]
    lifting=spsolve(base.c_xx+base.c_yy,-db[0]-db[1])
    build_seconds=time.perf_counter()-start;t=time.perf_counter();level=0.;step=.1;total=0
    while level<1-1e-14:
        nxt=min(1.,level+step);ds=nxt-level;p=copy.copy(goal)
        for k,d in zip(('b_xx','b_yy','b_xy'),db):setattr(p,k,getattr(base,k)+nxt*d)
        p.gv=base.gv+nxt*(goal.gv-base.gv);p.U=p.gv.copy();p.U[p.intm]=U[p.intm]+ds*lifting
        mineig=p.min_eig();status='nonconvex_predictor';iterations=0
        if mineig>0:
            p.solve(keep_convex=True);status=p.status;iterations=p.iterations;total+=iterations
        attempts.append(dict(level=nxt,step=ds,predictor_min_eig=mineig,status=status,iterations=iterations))
        if status=='converged':
            level=nxt;U=p.U.copy();step=min(.25,step*1.5)
        else:
            step*=.5
            if step<1/65536:raise RuntimeError(f'Boundary continuation stalled at {level:g}')
        if len(attempts)>1000:raise RuntimeError('Boundary continuation budget exhausted')
    p.iterations=total;p.initial_min_eig=initial
    return p,dict(build=build_seconds,fit=time.perf_counter()-t),dict(fallback=True,
        quadratic_affine=affine.tolist(),initial_min_eig=initial,attempts=attempts,
        interior_initialization='explicit quadratic plus discrete harmonic boundary lifting; no interior truth')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'out/fdm_continuation_2026-11-06');ap.add_argument('--only',default='');a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),HERE/'peanut_geometry.py',ROOT/'refs/fdm.py',ROOT/'lens/truth.py']
    protocol=dict(source=1,nx=[40,80,160],step_initial=.1,step_max=.25,step_min=1/65536,
        newton_maxit=60,newton_tolerance=2e-8,all_steps_convex=True,
        hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    meta=out/'protocol.json'
    if meta.exists() and json.loads(meta.read_text())!=protocol:raise RuntimeError('Use new output for changed code')
    meta.write_text(json.dumps(protocol,indent=2));tr=truth.RadialTruth();source=sources.Source('const',c=1)
    for domain,n in [('peanut',n) for n in (40,80,160)]+[('peanut_rounded',160)]:
        rid=f'{domain}_FDM_{n}';path=out/(rid+'.json')
        if a.only and rid not in a.only.split(','):continue
        if path.exists():continue
        print('BEGIN',rid,flush=True);r=make(domain)
        p,timing,details=solve(r,lambda x,y:tr(x,y)[0],source,n)
        rec=dict(run_id=rid,domain=domain,method='FDM',nx=n,ny=p.ny,power=power_state(),
            metrics=dict(**metrics.errors(p.U[p.intm],tr(p.X[p.intm],p.Y[p.intm])[0]),
                residual_inf=p.residual_inf,min_eig=p.min_eig(),unknowns=p.n),
            timing=timing,iterations=p.iterations,status=p.status,continuation=details)
        path.write_text(json.dumps(rec,indent=2));np.savez_compressed(out/(rid+'.npz'),x=p.X[p.intm],y=p.Y[p.intm],u=p.U[p.intm])
        print('RESULT',json.dumps({k:v for k,v in rec.items() if k!='continuation'}),flush=True)
    print('DONE',flush=True)

if __name__=='__main__':main()
