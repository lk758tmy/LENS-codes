"""Paired lambda and corner tests, separate from frozen formal_v1 records."""
import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LENS_THREADS','4')
import lens
import numpy as np
from lens import env,transport_cases as cases,transport_fit
from run import PROTOCOL,cleaned

def definitions(superellipse=False):
    out=[]
    for lam in (.4,.6):out.append(dict(id=f'U_A_l{lam:g}',group='lambda_truth',region='U',target='analytic',source=0,K=16,m=256,lam=lam,nu=0.,vertices=False))
    for lam in (.1,.2,.4,.6,1.):out.append(dict(id=f'U_D_l{lam:g}',group='lambda',region='U',target='D',source=0,K=16,m=256,lam=lam,nu=0.,vertices=False))
    levels='SDPU' if superellipse else 'SDPE'
    for row in cases.balanced_design(levels):
        if row['target'] not in ('S','P'):continue
        for nu in (0.,1.,10.):
            d=dict(row);d.update(id=f"C_{row['region']}_{row['target']}_f{row['source']}_n{nu:g}",group='corner',lam=.4,nu=nu,vertices=True)
            out.append(d)
    for a,b in (('S','P'),('P','S')):
        out.append(dict(id=f'V_{a}_{b}_midpoints',group='sampling',region=a,target=b,source=0,K=16,m=256,lam=.4,nu=0.,vertices=False))
    if superellipse:
        return [row for row in out if row['group']=='corner' and row['region']=='U']
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).parent/'out'/'ablations_v2')
    p.add_argument('--superellipse',action='store_true',help='rerun the six corner ablations with source U in place of E')
    p.add_argument('--only',default='');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'lens').glob('*.py'))+[Path(__file__),Path(__file__).with_name('run.py')]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    protocol=dict(PROTOCOL,case_lambda='see case',corner_weight='see case',source_vertices='see case',
                  corner_index='nearest joint sample; linearization frozen, trial objective re-minimized')
    design=definitions(a.superellipse);version=hashlib.sha256(json.dumps(dict(code=hashes,protocol=protocol,cases=design),sort_keys=True).encode()).hexdigest()
    log=out/'results.jsonl';old=[json.loads(x) for x in log.read_text(encoding='utf-8').splitlines() if x.strip()] if log.exists() else []
    if any(r['version']!=version for r in old):raise RuntimeError('Fingerprint changed; use new --out')
    done={r['run_id'] for r in old};environment=env.env_info(lens.THREADS)
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'design.json').write_text(json.dumps(dict(version=version,protocol=protocol,cases=design),indent=2),encoding='utf-8')
    def status(**kw):
        tmp=out/'status.tmp.json';tmp.write_text(json.dumps(dict(pid=os.getpid(),completed=len(done),time=time.strftime('%Y-%m-%dT%H:%M:%S'),version=version,**kw),indent=2),encoding='utf-8')
        # Windows readers/virus scanners can briefly deny rename. Telemetry must
        # never terminate a numerical run or mark a valid field as failed.
        for attempt in range(40):
            try:os.replace(tmp,out/'status.json');break
            except PermissionError:time.sleep(.025)
    for cfg in design:
        rid=cfg['id']
        if a.only and rid not in a.only.split(','):continue
        if rid in done:print('SKIP',rid,flush=True);continue
        start=time.perf_counter();status(case=rid,phase='build');print('BEGIN',json.dumps(cfg),flush=True)
        record=dict(run_id=rid,version=version,case=cfg,protocol=protocol,env=environment)
        try:
            center=(.5,.5) if a.superellipse else (0.,0.)
            r=cases.make_region(cfg['region'],superellipse_center=center);s=cases.make_source(cfg['source']);M=cases.source_moments(r,s)[0]
            tr=cases.SuperellipseManufactured() if cfg['target']=='analytic' else None
            T=tr.target() if tr else cases.make_target(cfg['target'],M)
            fp=hashlib.sha256(json.dumps(dict(version=version,case=cfg,target=T.spec),sort_keys=True).encode()).hexdigest()
            def progress(rec):
                status(case=rid,phase='fit',**rec);print('PROGRESS',json.dumps(dict(case=rid,**rec)),flush=True)
            f=transport_fit.TransportFit(r,T,K=cfg['K'],n_edge=cfg['m'],source=s,truth=tr,
                 lam=cfg['lam'],corner_weight=cfg['nu'],source_vertices=cfg['vertices'],
                 checkpoint=out/'checkpoints'/(rid+'.npz'),fingerprint=fp,progress=progress)
            f.build().fit();status(case=rid,phase='query');f.query();status(case=rid,phase='audit');t0=time.perf_counter();f.measure()
            f.timings['audit']=time.perf_counter()-t0
            record.update(metrics=f.metrics,timing=f.timings,status=f.status,iters=f.iters,m_actual=f.m,
                mass=f.mass,target=T.spec,box=f.box,initial_inverse_failures=f.initial_inverse_failures,
                initial_residuals=f.initial_residuals,reference_A=f.reference_A.tolist(),history=f.hist,wall=time.perf_counter()-start)
            (out/'fields').mkdir(exist_ok=True);np.savez_compressed(out/'fields'/(rid+'.npz'),**f.audit_field)
        except Exception as e:
            record.update(status='error',error=repr(e),traceback=traceback.format_exc(),wall=time.perf_counter()-start);print(traceback.format_exc(),flush=True)
        with log.open('a',encoding='utf-8') as fh:fh.write(json.dumps(cleaned(record),ensure_ascii=False,allow_nan=False)+'\n')
        done.add(rid);status(case=rid,phase='case_finished',result_status=record['status'])
        print('RESULT',json.dumps({k:v for k,v in cleaned(record).items() if k not in ('history','traceback')},ensure_ascii=False),flush=True)
    status(phase='finished',all_cases=len(design));print('DONE',len(done),flush=True)

if __name__=='__main__':main()
