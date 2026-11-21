"""Resumable formal transport experiment driver.

python -u exp/Section4.4_Transport/run.py --group all
Completed IDs are skipped; unfinished stages resume from numeric NPZ checkpoints.
Use a NEW --out directory after any solver/configuration change.
"""
import os,sys,json,time,hashlib,argparse,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LENS_THREADS','4')
import lens
import numpy as np
from lens import env,transport_cases as cases,transport_fit
from work_profile import WorkProfile,power_state

PROTOCOL=dict(K=24,m=256,lambda_joint=.4,delta=.1,iota=.02,mu_warm=1e-8,
              rounds=[60,60,200],coarse_K=4,certificate_grid=61,certificate_edge=1025,audit_grid=65,pde_grid=65,metrics_grid=65,audit_edge=4097,
              t_pad=.05,curvature_guard_factor=1.02,
              residual='joint_square',coordinates='stable_fixed_representative',
              curvature_step='active_set_Schur_LM',query_points=10000,query_reps=5,
              boundary_audit=2048,threads=lens.THREADS)

def cleaned(v):
    if isinstance(v,dict):return {k:cleaned(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cleaned(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    return v

def definitions(superellipse=False):
    levels='SDPU' if superellipse else 'SDPE'
    return [dict(r,K=24) for r in cases.balanced_design(levels)
            if not superellipse or 'U' in (r['region'],r['target'])]

def code_hashes():
    files=list((ROOT/'lens').glob('*.py'))+[Path(__file__),Path(__file__).with_name('work_profile.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',default='B',help='B: all sixteen crossed pairs')
    p.add_argument('--superellipse',action='store_true',help='rerun the seven pairs involving U in place of E')
    p.add_argument('--source-index',type=int,choices=(0,1),help='override the crossed-design source for diagnostic runs')
    p.add_argument('--only',default='',help='comma-separated exact case IDs')
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'out'/'formal_K24_2026-11-01')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    hashes=code_hashes();version=hashlib.sha256(json.dumps(dict(code=hashes,protocol=PROTOCOL,superellipse=a.superellipse,source_index=a.source_index),sort_keys=True).encode()).hexdigest()
    log=out/'results.jsonl';old=[json.loads(x) for x in log.read_text(encoding='utf-8').splitlines() if x.strip()] if log.exists() else []
    if any(r['version']!=version for r in old):raise RuntimeError('Output contains another code/protocol version; use a new --out')
    done={r['run_id'] for r in old};all_cases=definitions(a.superellipse)
    if a.source_index is not None:
        all_cases=[dict(r,source=a.source_index,id=f"B_{r['region']}_{r['target']}_f{a.source_index}") for r in all_cases]
    selected=[r for r in all_cases if (a.group=='all' or r['group'] in a.group.split(',')) and (not a.only or r['id'] in a.only.split(','))]
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'design.json').write_text(json.dumps(dict(version=version,protocol=PROTOCOL,cases=all_cases,
          design=('Seven U pairs of the SDPU design' if a.superellipse else 'Sixteen SDPE pairs')
                 + (f'; source override f{a.source_index}' if a.source_index is not None else '; f=(i+j)%2')
                 + '; no independent full interaction estimates'),indent=2),encoding='utf-8')
    environment=env.env_info(lens.THREADS)
    def status(**kw):
        q=dict(pid=os.getpid(),time=time.strftime('%Y-%m-%dT%H:%M:%S'),completed=len(done),version=version,**kw)
        tmp=out/'status.tmp.json';tmp.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf-8')
        for _ in range(40):
            try:os.replace(tmp,out/'status.json');break
            except PermissionError:time.sleep(.025)
    for cfg in selected:
        rid=cfg['id']
        if rid in done:print('SKIP',rid,flush=True);continue
        start=time.perf_counter();status(case=rid,phase='build')
        print('BEGIN',json.dumps(cfg),flush=True)
        record=dict(run_id=rid,version=version,case=cfg,protocol=PROTOCOL,env=environment,power_start=power_state())
        try:
            r=cases.make_region(cfg['region'],superellipse_center=(.5,.5));s=cases.make_source(cfg['source'])
            M=cases.source_moments(r,s)[0]
            tr=cases.Manufactured(cfg['source']) if cfg['group']=='A' else None
            T=tr.target() if tr else cases.make_target(cfg['target'],M)
            fp=hashlib.sha256(json.dumps(dict(version=version,case=cfg,target=T.spec),sort_keys=True).encode()).hexdigest()
            def progress(rec):
                status(case=rid,phase='fit',**rec)
                print('PROGRESS',json.dumps(dict(case=rid,**rec)),flush=True)
            f=transport_fit.TransportFit(r,T,K=cfg['K'],n_edge=cfg['m'],source=s,truth=tr,
                 checkpoint=out/'checkpoints'/(rid+'.npz'),fingerprint=fp,progress=progress)
            f.build()
            with WorkProfile() as profile:f.fit()
            record['work_profile']=profile.summary()
            status(case=rid,phase='query');f.query()
            status(case=rid,phase='audit');ta=time.perf_counter();f.measure()
            f.timings['audit']=time.perf_counter()-ta
            record.update(metrics=f.metrics,timing=f.timings,status=f.status,iters=f.iters,
              mass=f.mass,target=T.spec,box=f.box,initial_inverse_failures=f.initial_inverse_failures,
              initial_residuals=f.initial_residuals,reference_A=f.reference_A.tolist(),
              history=f.hist,wall=time.perf_counter()-start)
            (out/'fields').mkdir(exist_ok=True)
            np.savez_compressed(out/'fields'/(rid+'.npz'),**f.audit_field)
        except Exception as e:
            record.update(status='error',error=repr(e),traceback=traceback.format_exc(),wall=time.perf_counter()-start)
            print(traceback.format_exc(),flush=True)
        record['power_end']=power_state();record=cleaned(record)
        with log.open('a',encoding='utf-8') as file:file.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        done.add(rid);status(case=rid,phase='case_finished',result_status=record['status'])
        print('RESULT',json.dumps({k:v for k,v in record.items() if k not in ('history','traceback')},ensure_ascii=False),flush=True)
    status(phase='finished',selected=len(selected),all_design_cases=len(all_cases))
    print('DONE',len(done),'cases in',out,flush=True)

if __name__=='__main__':main()
