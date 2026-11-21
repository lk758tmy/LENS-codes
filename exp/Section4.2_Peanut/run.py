"""Section 4.2 supplement: peanut baselines and a common-LM rounded-end test."""
from pathlib import Path
import sys,os,argparse,json,time,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
os.environ.setdefault('LENS_THREADS','4')
import lens,numpy as np
from lens import metrics,sources,truth,fit,env
from refs.fdm import CurvedFD
from refs.pinn import solve_pinn
from exp.run_release import power_state
from peanut_geometry import make

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'out/supplement_2026-11-06');ap.add_argument('--only',default='');a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);log=out/'results.jsonl'
    files=[HERE/'run.py',HERE/'peanut_geometry.py']+list((ROOT/'lens').glob('*.py'))+[ROOT/'refs/fdm.py',ROOT/'refs/pinn.py']
    frozen=dict(protocol=dict(version=1,threads=4,source=1,truth='RadialTruth',K=16,m=256,
        lens_stages=[60,60,200],mu=[0,1e-8,0],lam=.4,grid=65,pde_grid=65,
        fdm_nx=[40,80,160],fdm_ny=[34,68,136],pinn_seeds=list(range(5)),
        pinn_width=48,pinn_depth=3,adam=1500,lbfgs=500,n_int=1024,n_bnd=256,
        smoothing='x^6+(y/[.85*(1-.55*exp(-(x/.35)^2))])^6<1'),
        hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    meta=out/'protocol.json'
    if meta.exists() and json.loads(meta.read_text())!=frozen:raise RuntimeError('Source changed; use a new output directory')
    meta.write_text(json.dumps(frozen,indent=2))
    done={r['run_id'] for r in map(json.loads,log.read_text().splitlines())} if log.exists() else set()
    tr=truth.RadialTruth();s=sources.Source('const',c=1)
    tasks=[('FDM','peanut',n) for n in (40,80,160)]+[('LENS','peanut_rounded',16),('FDM','peanut_rounded',160)]+[('PINN','peanut',seed) for seed in range(5)]
    for method,domain,param in tasks:
        rid=f'{domain}_{method}_{param}'
        if a.only and rid not in a.only.split(','):continue
        if rid in done:continue
        print('BEGIN',rid,flush=True);r=make(domain);rec=dict(run_id=rid,method=method,domain=domain,
            power=power_state(),time=time.strftime('%Y-%m-%dT%H:%M:%S'))
        tic=time.perf_counter()
        try:
            if method=='FDM':
                nx=param;x0,x1,y0,y1=r.bbox;h=(x1-x0)/nx;ny=int(np.ceil((y1-y0)/h-1e-10))
                p=CurvedFD(x0,x1,y0,y0+ny*h,nx,ny,r.contains,lambda X,Y:tr(X,Y)[0],s)
                p.initial();build=time.perf_counter()-tic;t0=time.perf_counter()
                p.solve(keep_convex=True);sec=time.perf_counter()-t0
                rec.update(nx=nx,ny=ny,metrics=dict(**metrics.errors(p.U[p.intm],tr(p.X[p.intm],p.Y[p.intm])[0]),
                    residual_inf=p.residual_inf,unknowns=p.n,min_eig=p.min_eig(),initial_min_eig=p.initial_min_eig),
                    timing=dict(build=build,fit=sec),iterations=p.iterations,status=p.status)
                np.savez_compressed(out/(rid+'.npz'),x=p.X[p.intm],y=p.Y[p.intm],u=p.U[p.intm])
            elif method=='LENS':
                f=fit.DirichletFit(r,s,16,256,truth=tr).run()
                rec.update(K=16,m=256,metrics=f.metrics,timing=f.timings,iters=f.iters,
                           box=f.basis.box,status='admissible' if f.metrics['n_out']==0 else 'invalid',history=f.hist)
                np.savez_compressed(out/(rid+'.npz'),theta=f.theta.numpy(),t=f.t.numpy(),box=f.basis.box)
            else:
                field,mm=solve_pinn(r,tr,s,seed=param);X,Y=r.interior_grid(65);U=field(X,Y)
                mm.update(metrics.errors(U,tr(X,Y)[0]));rec.update(seed=param,metrics=mm,timing=dict(fit=mm['wall']),
                    status='convex_samples' if mm['min_eig']>0 else 'nonconvex_samples')
                np.savez_compressed(out/(rid+'.npz'),x=X,y=Y,u=U)
        except Exception as e:
            rec.update(status='error',error=repr(e),wall=time.perf_counter()-tic)
            if method=='FDM' and 'p' in locals():rec['initial_min_eig']=p.initial_min_eig
        with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(rec)+'\n')
        print('RESULT',json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
    (out/'environment.json').write_text(json.dumps(dict(environment=env.env_info(lens.THREADS),power_end=power_state()),indent=2))
    print('DONE',flush=True)

if __name__=='__main__':main()
