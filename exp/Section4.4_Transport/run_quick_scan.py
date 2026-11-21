"""Quick controlled S->P f=1 sweep; no core or manuscript changes."""
import sys,os,json,time,hashlib,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LENS_THREADS','4')
import lens,numpy as np,torch
from lens import transport_cases as cases,transport_fit as tf,env
from run import cleaned

class TargetProjectionBoxFit(tf.TransportFit):
    """Data-only smaller rectangle: x in source projection, t in target projection.

    Curvature remains constrained on the WHOLE new rectangle and inversion uses
    that SAME interval. No iterate-dependent deletion or hidden outside Newton.
    """
    def _set_chart(self,K):
        self.box=(*self.box[:2],self.T.qmin,self.T.qmax)
        super()._set_chart(K)

def main():
    p=argparse.ArgumentParser();p.add_argument('--Ks',default='16,24,32');p.add_argument('--lams',default='.4')
    p.add_argument('--box',choices=['full','target'],default='full')
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'out/quick_K_lambda_2026-11-01')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'lens').glob('*.py'))+[Path(__file__)]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    protocol=dict(case='S_to_P_f0',m=256,nu=0.,source_vertices=False,delta=.1,iota=.02,
        mu=1e-8,rounds=[60,60,200],coarse_K=4,threads=lens.THREADS,
        training_curvature='61^2+4x1025',audit='65^2+4x4097',pde_grid=65,metrics_grid=65,boundary_audit=2048,
        query_timing_omitted=True,box_modes='full:5%+0.001; target:exact known target t projection')
    version=hashlib.sha256(json.dumps(dict(code=hashes,protocol=protocol),sort_keys=True).encode()).hexdigest()
    log=out/'results.jsonl';old=[json.loads(x) for x in log.read_text(encoding='utf-8').splitlines() if x.strip()] if log.exists() else []
    if any(r['version']!=version for r in old):raise RuntimeError('Code/protocol mismatch; use new --out')
    done={r['run_id'] for r in old};environment=env.env_info(lens.THREADS)
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(dict(version=version,protocol=protocol),indent=2),encoding='utf-8')
    def status(**kw):
        tmp=out/'status.tmp.json';tmp.write_text(json.dumps(dict(pid=os.getpid(),completed=len(done),time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kw),indent=2),encoding='utf-8')
        for _ in range(40):
            try:os.replace(tmp,out/'status.json');break
            except PermissionError:time.sleep(.025)
    for K in map(int,a.Ks.split(',')):
        for lam in map(float,a.lams.split(',')):
            rid=f'S_P_K{K}_l{lam:g}_{a.box}'
            if rid in done:print('SKIP',rid,flush=True);continue
            cfg=dict(K=K,lam=lam,box_mode=a.box);start=time.perf_counter();print('BEGIN',rid,flush=True)
            rec=dict(run_id=rid,version=version,case=cfg,protocol=protocol,env=environment)
            try:
                r=cases.make_region('S');s=cases.make_source(0);T=cases.make_target('P',cases.source_moments(r,s)[0])
                fp=hashlib.sha256(json.dumps(dict(version=version,case=cfg),sort_keys=True).encode()).hexdigest()
                def progress(q):
                    status(case=rid,phase='fit',**q);print('PROGRESS',json.dumps(dict(case=rid,**q)),flush=True)
                Cls=tf.TransportFit if a.box=='full' else TargetProjectionBoxFit
                fit=Cls(r,T,K=K,n_edge=256,lam=lam,corner_weight=0.,source_vertices=False,
                    checkpoint=out/'checkpoints'/(rid+'.npz'),fingerprint=fp,progress=progress)
                fit.build().fit();status(case=rid,phase='audit');fit.prob=fit._problem(0)
                t0=time.perf_counter();fit.measure();fit.timings['audit']=time.perf_counter()-t0
                # Also report the old full prior-box check for target-box runs.
                x0,x1=r.bbox[:2];pad=.05*(T.qmax-T.qmin)+.001
                xx,tt=torch.meshgrid(torch.linspace(x0,x1,65),torch.linspace(T.qmin-pad,T.qmax+pad,65),indexing='ij')
                ax=torch.linspace(x0,x1,4097);at=torch.linspace(T.qmin-pad,T.qmax+pad,4097)
                xx=torch.cat([xx.ravel(),ax,ax,torch.full_like(at,x0),torch.full_like(at,x1)])
                tt=torch.cat([tt.ravel(),torch.full_like(ax,T.qmin-pad),torch.full_like(ax,T.qmax+pad),at,at])
                oldmargin=float(fit.theta[-1]-(fit.theta[:-1]@fit.basis.parts(xx,tt)['WTT']).max())
                fit.metrics['original_padded_box_margin']=oldmargin
                rec.update(metrics=fit.metrics,status=fit.status,timing=fit.timings,iters=fit.iters,
                    box=fit.box,initial_inverse_failures=fit.initial_inverse_failures,history=fit.hist)
                (out/'fields').mkdir(exist_ok=True);np.savez_compressed(out/'fields'/(rid+'.npz'),**fit.audit_field)
            except Exception as e:
                rec.update(status='error',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
            rec['wall']=time.perf_counter()-start
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(cleaned(rec),allow_nan=False)+'\n')
            done.add(rid);print('RESULT',json.dumps({k:v for k,v in cleaned(rec).items() if k not in ('history','traceback')}),flush=True)
    status(phase='finished');print('DONE',len(done),flush=True)

if __name__=='__main__':main()
