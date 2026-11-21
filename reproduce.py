"""Export frozen paper results, run checks, or explicitly start fresh fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',LENS_THREADS='4')

def call(script,*args,env=None,plan=False):
    command=[sys.executable,'-B',script,*map(str,args)]
    print(' '.join(command),flush=True)
    if not plan:subprocess.run(command,cwd=ROOT,env=env or ENV,check=True)

def export(args):
    out=args.out.resolve();fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
    extra=[]
    data=ROOT/'exp/release_2026-11-05'
    inverse=ROOT/'exp/Section5_Inverse/out/budget200_2026-11-19'
    if args.rerun_root:
        rerun=args.rerun_root.resolve();data=rerun/'forward';inverse=rerun/'inverse'
        required=[data/'dirichlet',data/'transport_K16',data/'corners',inverse]
        if not all(p.is_dir() for p in required):raise SystemExit('Complete all training stages before exporting a fresh run.')
        extra=['--data-root',data,'--supplement-root',rerun/'peanut_pinn',
               '--fdm-supplement-root',rerun/'peanut_fdm','--rhs-root',rerun/'rhs',
               '--ot-superellipse-root',rerun/'superellipse',
               '--ot-fdm-comparison',rerun/'ot_fdm/comparison.json']
    call('exp/export_tables.py','--out',out/'forward','--figures','--no-paper-check',*extra)
    call('exp/Section5_Inverse/export_budget.py','--data-root',inverse,'--out',out/'inverse',
         '--report-time','100','--time-max','100','--no-paper-check')
    env=dict(ENV,LENS_FIGURE_OUT=str(fig))
    call('exp/Section4.1_Regions/make_regions_figure.py',env=env)
    call('exp/Section4.2_Peanut/plot_peanut_error.py','--data-root',data/'dirichlet','--out',fig/'fig_peanut_error.pdf')
    call('exp/Section4.4_Transport/plot_ot_A1_errors.py','--data-root',data/'transport_K16','--out',fig/'fig_ot_A1_errors.png')
    call('exp/Section4.4_Transport/plot_ot_error.py','--data-root',data/'corners','--out',fig/'fig_ot_error.pdf')
    if not args.rerun_root:
        want=json.loads((ROOT/'EXPECTED_TABLES.json').read_text(encoding='utf-8'))
        for name,digest in want.items():
            actual=hashlib.sha256((out/name).read_bytes()).hexdigest()
            if actual!=digest:raise RuntimeError('Frozen numerical table changed: '+name)
        print('PASS: both frozen numerical table exports match the recorded reference.')
    print('DONE: '+str(out))

def train(args):
    out=args.out.resolve();base=out/'forward';tasks=[]
    upstream=(args.upstream or ROOT/'third_party/ma-ot-monotone').resolve()
    if args.section in ('all','ot-baseline') and not (upstream/'main/ma.py').is_file() and not args.plan:
        raise SystemExit('First run: python tools/fetch_ot_baseline.py; install requirements-ot.txt, or pass --upstream.')
    if args.section in ('all','forward'):
        tasks=[('exp/run_formal.py',['--group','geometry,resolution,comparison,pinn,coordinate_reference','--out',base/'dirichlet']),
               ('exp/coordinate_control.py',['--out',base/'dirichlet']),
               ('exp/Section4.2_Peanut/run.py',['--only',','.join(f'peanut_PINN_{i}' for i in range(5)),'--out',out/'peanut_pinn']),
               ('exp/Section4.2_Peanut/fdm_continuation.py',['--only','peanut_FDM_40,peanut_FDM_80,peanut_FDM_160','--out',out/'peanut_fdm']),
               ('exp/Section4.3_RHS/run_nonfamily.py',['--out',out/'rhs']),
               ('exp/Section4.4_Transport/run.py',['--group','A','--out',base/'transport_K16'])]
        ids=[f'B_{a}_{b}_f{(i+j)%2}' for i,a in enumerate('SDPE') for j,b in enumerate('SDPE') if 'E' not in (a,b)]
        tasks += [('exp/Section4.4_Transport/run_k24.py',['--only',','.join(ids),'--out',base/'transport_K24']),
                  ('exp/Section4.4_Transport/run_k24.py',['--superellipse','--out',out/'superellipse/main']),
                  ('exp/Section4.4_Transport/run_quick_scan.py',['--Ks','16,24,32','--lams','0.4,1,3','--out',base/'transport_parameters'])]
        corners=[f'C_{a}_{b}_f{f}_n{n}' for a,b,f in [('S','P',0),('D','S',1),('D','P',1),('P','S',0)] for n in (0,1,10)]
        tasks += [('exp/Section4.4_Transport/run_ablations.py',['--only',','.join(corners),'--out',base/'corners']),
                  ('exp/Section4.4_Transport/run_ablations.py',['--superellipse','--out',out/'superellipse/corners'])]
    if args.section in ('all','ot-baseline'):
        tasks += [('exp/Section4.4_Transport/run_fdm_comparison.py',['--case','S_D_f1','--n','32','64','128','256',
                                                                   '--upstream',upstream,'--out',out/'ot_fdm'])]
    if args.section in ('all','inverse'):
        tasks += [('exp/Section5_Inverse/run_budget.py',['--seconds','200','--n','160','--out',out/'inverse']),
                  ('exp/Section5_Inverse/audit_budget.py',['--data-root',out/'inverse'])]
    # Never append retraining output to an immutable packaged record directory.
    if any(out==p or p in out.parents for p in (ROOT/'exp',ROOT/'lens',ROOT/'refs')):
        raise SystemExit('Use a fresh output directory outside exp/, lens/ and refs/.')
    for script,options in tasks:call(script,*options,plan=args.plan)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check',help='Verify hashes and run the bundled mathematical regression checks')
    p=sub.add_parser('export',help='Regenerate every paper table and figure without training')
    p.add_argument('--out',type=Path,default=ROOT/'reproduced')
    p.add_argument('--rerun-root',type=Path,help='Use complete newly trained outputs instead of frozen results')
    p=sub.add_parser('train',help='Explicitly run fresh fitting; can take substantial time')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--section',choices=('all','forward','inverse','ot-baseline'),default='all')
    p.add_argument('--upstream',type=Path)
    p.add_argument('--plan',action='store_true',help='Print commands without running fits')
    args=parser.parse_args()
    if args.command=='check':
        call('tools/verify.py');call('tests/run_all.py')
    elif args.command=='export':export(args)
    else:train(args)

if __name__=='__main__':main()
