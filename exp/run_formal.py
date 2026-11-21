"""Reproduce the repaired Section 4.1--4.3 tables, including honest failures.

Run from repository root: python -u exp/run_formal.py --group all
Raw results are append-only and resumable by run_id, in formal_2026-10-31.
"""
import os,sys,argparse,json,time,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('LENS_THREADS','4')
import lens
import numpy as np
import torch
from lens import fit,regions,sources,truth,metrics,env,init,chart,cert,res,lm

OUT=ROOT/'exp'/'release_2026-11-05'/'dirichlet';OUT.mkdir(parents=True,exist_ok=True)
LOG=OUT/'results.jsonl'
DONE={r['run_id'] for r in map(json.loads,LOG.read_text(encoding='utf-8').splitlines())} if LOG.exists() else set()
ENV=env.env_info(lens.THREADS)
VERSION=None

def configure_output(path):
    global OUT,LOG,DONE,VERSION
    OUT=Path(path).resolve();OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'results.jsonl'
    files=[p for folder in ('lens','refs','tests') for p in (ROOT/folder).glob('*.py')]
    files += [Path(__file__),ROOT/'exp/coordinate_control.py']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    protocol=dict(coordinates='stable',metric='Dirichlet relative to uncentered exact solution',
                  grid=65,pde_grid=65,training_grid=61,threads=lens.THREADS,
                  stages=[60,60,200],mu=[0,1e-8,0],lambda_joint=.4)
    VERSION=hashlib.sha256(json.dumps(dict(code=hashes,protocol=protocol),sort_keys=True).encode()).hexdigest()
    meta=OUT/'run_config.json'
    if meta.exists() and json.loads(meta.read_text())['version']!=VERSION:
        raise RuntimeError('Code/protocol changed; choose a fresh output directory')
    old=[json.loads(s) for s in LOG.read_text().splitlines() if s.strip()] if LOG.exists() else []
    if any(r.get('version')!=VERSION for r in old):raise RuntimeError('Mixed output version')
    DONE={r['run_id'] for r in old}
    meta.write_text(json.dumps(dict(version=VERSION,protocol=protocol),indent=2),encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

def emit(r):
    r['env']=ENV;r['version']=VERSION;r['ts']=time.strftime('%Y-%m-%dT%H:%M:%S')
    with LOG.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    DONE.add(r['run_id']);print(json.dumps(r,ensure_ascii=False),flush=True)

def lens_run(group,domain,K,m,kind=None,kw=None):
    id=f'{group}_{domain}_{kind or "radial"}_K{K}_m{m}'
    if id in DONE:return
    r=regions.get_region(domain)
    s=sources.Source(kind or 'const',**(kw or {'c':1}))
    tr=truth.SeparableTruth(s,alpha=1,y0=.5) if kind else truth.RadialTruth()
    cfg=dict(run_id=id,group=group,method='LENS',domain=domain,source=kind or 'const1',K=K,m=m)
    try:
        f=fit.DirichletFit(r,s,K,m,truth=tr,basis_kind='fx' if kind else 'const').run()
        # Independent denser convexity sample, in addition to the fitting certificate.
        Sd=f.basis.cert_S(65)
        dense=float(f.theta[-1]-(f.theta[:-1]@Sd).max())
        cfg.update(metrics=f.metrics,timing=f.timings,iters=f.iters,
                   loss_initial=f.hist[0]['loss'] if f.hist else None,
                   loss_final=f.hist[-1]['loss'] if f.hist else None,
                   dense_margin=dense,box=f.basis.box,
                   prolongation_error=f.prolongation_error,
                   status='valid' if f.metrics['n_out']==0 and dense>=f.delta-1e-10 else 'invalid')
        (OUT/'models').mkdir(exist_ok=True)
        np.savez_compressed(OUT/'models'/(id+'.npz'),theta=f.theta.numpy(),t=f.t.numpy(),box=f.basis.box)
        if group=='comparison':
            xx,yy=r.interior_grid(101);ff=metrics.field_on_grid(f.chart,f.theta,xx,yy)
            np.savez(OUT/(id+'.npz'),x=xx,y=yy,u=ff['u'],exact=tr(xx,yy)[0])
        # Stationarity can occur at a poor local minimum; report residuals, not
        # a blanket "converged" status based on convexity alone.
        cfg['boundary_fit_rms']=f.metrics['res1']
    except Exception as e:cfg.update(status='error',error=repr(e))
    emit(cfg)

def fdm_run(domain,n):
    from refs.fdm import solve_fdm
    id=f'comparison_{domain}_FDM_n{n}'
    if id in DONE:return
    r=regions.get_region(domain);tr=truth.RadialTruth();s=sources.Source('const',c=1)
    cfg=dict(run_id=id,group='comparison',method='FDM',domain=domain,n=n)
    try:
        p=solve_fdm(r,tr,s,n=n,keep_convex=True)
        e=metrics.errors(p.U[p.intm],tr(p.X[p.intm],p.Y[p.intm])[0])
        cfg.update(metrics=dict(**e,residual_inf=p.residual_inf,min_eig=p.min_eig(),
                         initial_min_eig=p.initial_min_eig,unknowns=p.n),
                   timing=dict(build=p.build_seconds,fit=p.fit_seconds),
                   iterations=p.iterations,status=p.status)
    except Exception as e:cfg.update(status='error',error=repr(e))
    emit(cfg)

def pinn_run(domain,seed):
    from refs.pinn import solve_pinn
    id=f'comparison_{domain}_PINN_seed{seed}'
    if id in DONE:return
    r=regions.get_region(domain);tr=truth.RadialTruth();s=sources.Source('const',c=1)
    ff,mm=solve_pinn(r,tr,s,seed=seed)
    X,Y=r.interior_grid(65)
    prediction=ff(X,Y);exact=tr(X,Y)[0]
    mm.update(metrics.errors(prediction,exact))
    (OUT/'fields').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'fields'/(id+'.npz'),x=X,y=Y,u=prediction,exact=exact)
    emit(dict(run_id=id,group='comparison',method='PINN',domain=domain,seed=seed,
              metrics=mm,timing=dict(fit=mm['wall']),
              status='convex_samples' if mm['min_eig']>0 else 'nonconvex_samples'))

def coordinates_run(domain,K,kind=None):
    r=regions.get_region(domain);s=sources.Source(kind or 'const',c=1)
    tr=truth.SeparableTruth(s,alpha=1,y0=.5) if kind else truth.RadialTruth()
    x,y=r.boundary(256);g=tr(x,y)[0]
    ii=init.boundary_reference(r,x,y,g,s,K=K,basis_kind='fx' if kind else 'const')
    b=ii['basis'];ch=chart.Chart(b,cert.Certificate(b.cert_S(61),delta=.1,iota=.02))
    stable=ii['theta0'];native,GG=ch.equality_coordinates(stable)
    tt,_=ch.solve_t(torch.as_tensor(x),torch.as_tensor(y),stable)
    for mode,theta in [('native_equality',native),('physical_coordinates',stable)]:
        id=f'coordinates_{domain}_{kind or "radial"}_K{K}_{mode}'
        if id in DONE:continue
        pr=res.DirichletBVP(ch,x,y,g,lam=.4,mu=0,tau_mode='joint',
                            gamma_mode='cert' if mode=='native_equality' else 'gauge')
        tic=time.perf_counter();o=lm.solve(pr,theta,tt,rounds=260,ftol=1e-26*len(x))
        sec=time.perf_counter()-tic;X,Y=r.interior_grid(65)
        fld=metrics.field_on_grid(ch,o['theta'],X,Y);ev=pr.eval(o['theta'],o['t'])
        emit(dict(run_id=id,group='coordinates',method=mode,domain=domain,K=K,source=kind or 'radial',
                  metrics=dict(**metrics.errors(fld['u'],tr(X,Y)[0]),r1=ev['r1'],r2=ev['r2'],
                      margin=ch.margin(o['theta']),n_out=fld['n_out']),
                  timing=dict(fit=sec),iterations=o['iters'],initial_gamma=float(ch.gamma(theta))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='all');ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
    configure_output(a.out)
    groups=a.group.split(',')
    if 'all' in groups:groups=['geometry','resolution','comparison','rhs','pinn']
    if 'coordinates' in groups:
        for K in (12,16):coordinates_run('square',K)
        for K in (3,16):coordinates_run('disk',K,'quad')
    if 'geometry' in groups:
        for K in range(4,33,2):lens_run('geometry','square',K,512)
    if 'resolution' in groups:
        for K in (8,16,24):
            for m in (16,24,32,40,48,64,96,128,192,256):lens_run('resolution','square',K,m)
    if 'comparison' in groups:
        for domain in ('square','disk','peanut'):
            lens_run('comparison',domain,16,256)
        for domain in ('square','disk'):
            for n in (40,80,160):fdm_run(domain,n)
    if 'coordinate_reference' in groups:
        lens_run('rhs','disk',16,256,'quad',dict(a=2))
    if 'rhs' in groups:
        for domain in ('square','disk'):
            for kind,kw in [('const',dict(c=4)),('lin',dict(a=1)),('quad',dict(a=2)),('cos',dict(a=.3))]:
                lens_run('rhs',domain,16,256,kind,kw)
    if 'pinn' in groups:
        for domain in ('square','disk'):
            for seed in range(5):pinn_run(domain,seed)

if __name__=='__main__':main()
