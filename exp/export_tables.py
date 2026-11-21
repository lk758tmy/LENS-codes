"""Read frozen records; export current paper tables and figures without fitting."""
from pathlib import Path
import argparse,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'exp/generated');ap.add_argument('--figures',action='store_true');ap.add_argument('--no-paper-check',action='store_true');ap.add_argument('--data-root',type=Path);ap.add_argument('--supplement-root',type=Path);ap.add_argument('--fdm-supplement-root',type=Path);ap.add_argument('--rhs-root',type=Path);ap.add_argument('--ot-superellipse-root',type=Path);ap.add_argument('--ot-fdm-comparison',type=Path);args=ap.parse_args()
BASE=args.out.resolve();BASE.mkdir(parents=True,exist_ok=True)
current=ROOT/'exp/current_release.json'
RELEASE=args.data_root.resolve() if args.data_root else ((ROOT/json.loads(current.read_text())['data_root']).resolve() if current.exists() else None)
OVERLAY_PATH=ROOT/'exp/evaluation_65_2026-11-04/results.jsonl' 
OVERLAY={r['key']:r['metrics'] for r in map(json.loads,OVERLAY_PATH.read_text(encoding='utf-8').splitlines())} if RELEASE is None and OVERLAY_PATH.exists() else {}
def read(path):
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    for r in rows:
        a=OVERLAY.get(path.parent.name+'::'+r['run_id'])
        if a is not None:r['metrics']['Rpde']=a['Rpde']
    return rows
DIR=RELEASE/'dirichlet' if RELEASE else ROOT/'exp/formal_2026-10-31'
records=read(DIR/'results.jsonl')

# Display labels change, while immutable historical case IDs retain P=triangle.
def paper_code(code):return 'T' if code=='P' else code
pointer=json.loads(current.read_text()) if current.exists() else {}
supp=pointer.get('supplements',{}) if args.data_root is None else {}
ps=args.supplement_root or (ROOT/supp['section42'] if 'section42' in supp else None)
fs=args.fdm_supplement_root or (ROOT/supp['section42_fdm'] if 'section42_fdm' in supp else None)
if ps:
    supplement=read(Path(ps)/'results.jsonl')
    records += [dict(r,group='comparison') for r in supplement if r['method']=='PINN' and r['domain']=='peanut']
if fs:
    records += [dict(json.loads(p.read_text()),group='comparison',n=int(p.stem.rsplit('_',1)[-1]))
                for p in sorted(Path(fs).glob('peanut_FDM_*.json'))]
rs=args.rhs_root or (ROOT/supp['section43'] if 'section43' in supp else None)
if rs:
    rhs_records=read(Path(rs)/'results.jsonl')
    if any(r['source'] == 'reciprocal_quartic' for r in rhs_records):
        expected = {'square', 'disk', 'triangle'}
        domains = {r['domain'] for r in rhs_records}
        if (len(rhs_records) != 4
                or domains not in (expected | {'ellipse'}, expected | {'superellipse4'})
                or not all(r['source'] == 'reciprocal_quartic' and r['K'] == 16 and r['m'] == 256
                           and r['status'] == 'admissible' for r in rhs_records)):
            raise RuntimeError('Expected four admissible non-family cases at K=16 and m=256')
    elif len(rhs_records) != 12:
        raise RuntimeError('Complete all 12 source/region cases before exporting')
    records=[r for r in records if r['group']!='rhs']+rhs_records
def domain_label(name):return dict(square='S',disk='D',peanut='P',triangle='T',ellipse='E',superellipse4='U').get(name,name)

def rows(group,method=None):return [r for r in records if r['group']==group and (not method or r['method']==method)]
def sci(x):
    if not np.isfinite(x):return r'---'
    if x==0:return '$0$'
    e=int(np.floor(np.log10(abs(x))));a=x/10**e
    return f'${a:.2f}\\times10^{{{e}}}$'
def table(caption,label,cols,head,body):
    return '\n'.join([r'\begin{table}[t]',r'\centering',r'\caption{'+caption+r'}\label{'+label+'}',r'\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{'+cols+'}',r'\toprule',head+r'\\',r'\midrule',body,r'\bottomrule',r'\end{tabular}',r'\end{table}'])
geom=sorted(rows('geometry'),key=lambda r:r['K'])
cmp=rows('comparison','LENS');rhs=rows('rhs');fdm=rows('comparison','FDM')
env=records[0]['env']
body=[]
for domain,K,kind in [('square',12,'radial'),('square',16,'radial'),('disk',3,'quad'),('disk',16,'quad')]:
    rr=[r for r in rows('fixedbox') if r['domain']==domain and r['K']==K]
    if len(rr)!=2:raise RuntimeError('Run coordinate_control.py first')
    n=next(r for r in rr if r['method']=='native_equality');s=next(r for r in rr if r['method']=='physical_coordinates')
    body.append(('S' if domain=='square' else 'D')+' & '+('$f=1$' if kind=='radial' else '$f=1+2x^2$')+f' & {K} & '+sci(n['metrics']['E2'])+' & '+sci(s['metrics']['E2'])+r'\\')
certtable=table(r'Controlled equality-$\gamma$ comparison, $m=256$, joint LM with $\lambda=0.4$, at most 260 iterations and no ridge. Each pair has the same fixed data-derived box and the same initial physical chart. Only the coordinates used by LM differ; these rows diagnose conditioning and use no curvature rejection.',
 'tab:cert','lllcc',r'region & source term & $K$ & $E_2$ (original) & $E_2$ (stable)','\n'.join(body))
body=[]
for r in cmp:
    z=r['metrics'];sec=r['timing']['build']+r['timing']['fit']
    body.append(f"LENS $K=16$ & {domain_label(r['domain'])} & 33 & "+sci(z['E2'])+' & '+sci(z['Rpde'])+f' & {sec:.2f}'+r'\\')
body.append(r'\midrule')
for r in sorted(fdm,key=lambda r:({'square':0,'disk':1,'peanut':2}.get(r['domain'],3),r['n'])):
    z=r['metrics'];sec=sum(r['timing'].values())
    body.append(f"FDM $n={r['n']}$ & {domain_label(r['domain'])} & {z['unknowns']} & "+sci(z['E2'])+' & '+sci(z['residual_inf'])+f' & {sec:.2f}'+r'\\')
body.append(r'\midrule')
pinns={}
for d in (('square','disk','peanut') if ps else ('square','disk')):
    pp=[r for r in rows('comparison','PINN') if r['domain']==d]
    if len(pp)!=5:raise RuntimeError('All five PINN seeds must finish')
    pinns[d]=pp
    e=np.array([r['metrics']['E2'] for r in pp]);rr=np.mean([r['metrics']['rms_pde'] for r in pp]);sec=np.mean([r['timing']['fit'] for r in pp])
    body.append(f'PINN & {domain_label(d)} & 4897 & '+sci(e.mean())+' & '+sci(rr)+f' & {sec:.1f}'+r'\\')
cmptable=table(r'Comparison for \eqref{eq:s-radius}. Time includes build and all fitting stages; PINN entries are means of five seeds. Internal residual means max AD residual for LENS, max discrete residual for FDM, and RMS continuous residual for PINN. FDM errors use its unknown nodes; LENS/PINN use the independent $65^2$ grid. S, D and P denote square, disk and peanut. The LENS peanut row is a poor fit; PINN means include all five seeds. For peanut, FDM uses $n_x=n$, $n_y=0.85n$ and boundary-data continuation after a nonconvex Poisson start; all initialization costs are included.',
 'tab:compare','llcccc',r'method & domain & size & $E_2$ & residual & time [s]','\n'.join(body))
if any(r['source'] == 'reciprocal_quartic' for r in rhs):
    body = []
    ordered = sorted(rhs, key=lambda r: 'SDTEU'.index(domain_label(r['domain'])))
    for r in ordered:
        z = r['metrics']
        iterations = '/'.join(str(n) for n in r['iters'])
        cells = [domain_label(r['domain']), sci(z['E2']), sci(z['E2B']), sci(z['Rpde']),
                 iterations, f"{r['timing']['fit']:.2f}"]
        body.append(' & '.join(cells) + r'\\')
    rhstable = table(
        r'Dirichlet tests with $f(x)=(1+x/2)^{-4}$, $K=16$, and $m=256$. LM counts are given for initialization, warm-up, and final fitting, respectively. Time includes all three fitting stages.',
        'tab:fx', 'lccccc', r'region & $E_2$ & $E_{2,B}$ & $R_{\rm PDE}$ & LM iterations & fit [s]',
        '\n'.join(body))
else:
    body=[]
    srcnames={'const':'$4$','lin':'$1+x$','quad':'$1+2x^2$','exp':'$e^x$','cos':r'$1+0.3\cos(2\pi x)$'}
    for r in rhs:
        z=r['metrics'];body.append(srcnames[r['source']]+f" & {domain_label(r['domain'])} & "+sci(z['E2'])+' & '+sci(z['res1'])+' & '+sci(z['Rpde'])+f" & {r['timing']['fit']:.2f}"+r'\\')
    rhstable=table(r'Space-dependent-source tests, $K=16$ (34 coefficients and 256 optimized latent variables), $m=256$, $\alpha=1$, $y_0=1/2$. Every row uses the same joint-LM stages as Section~4.2. Every audit inverse is inside its box.',
     'tab:fx','llcccc',r'$f(x)$ & domain & $E_2$ & boundary RMS & $R_{\rm PDE}$ & fit [s]','\n'.join(body))


B=ROOT/'exp/Section4.4_Transport/out'
main=read(RELEASE/'transport_K24/results.jsonl') if RELEASE else read(B/'formal_K24_2026-11-01/results.jsonl')
old=read(RELEASE/'transport_K16/results.jsonl') if RELEASE else read(B/'formal_v1/results.jsonl')
quick=read(RELEASE/'transport_parameters/results.jsonl') if RELEASE else read(B/'quick_K_lambda_2026-11-01/results.jsonl')
params=[] if RELEASE else read(B/'parameter_grid_2026-11-01/results.jsonl')
corner=read(RELEASE/'corners/results.jsonl') if RELEASE else read(B/'ablations_v2/results.jsonl')
levels='SDPE'
run_root=args.ot_superellipse_root or (ROOT/supp['section44_superellipse'] if 'section44_superellipse' in supp else None)
if run_root:
    replacements=read(run_root/'main/results.jsonl')
    expected={f'B_{a}_{b}_f{(i+j)%2}' for i,a in enumerate('SDPU')
              for j,b in enumerate('SDPU') if 'U' in (a,b)}
    if (len(replacements)!=len(expected) or {r['run_id'] for r in replacements}!=expected
            or any(r['status'] not in {'fit_converged','poor_fit','invalid'} for r in replacements)):
        raise RuntimeError('Complete all seven superellipse transport pairs before exporting')
    main=[r for r in main if 'E' not in (r['case']['region'],r['case']['target'])]+replacements
    corner_replacements=read(run_root/'corners/results.jsonl')
    expected={f'C_U_{b}_f1_n{nu}' for b in ('S','P') for nu in (0,1,10)}
    if (len(corner_replacements)!=len(expected)
            or {r['run_id'] for r in corner_replacements}!=expected
            or any(r['status'] not in {'fit_converged','poor_fit','invalid'} for r in corner_replacements)):
        raise RuntimeError('Complete all six superellipse corner fits before exporting')
    corner=[r for r in corner if r['case']['region']!='E']+corner_replacements
    levels='SDPU'
assert len(main)==16 and all(r['case']['K']==24 for r in main)
body=[]
for a in levels:
    vals=[]
    for b in levels:
        r=next(r for r in main if r['case']['region']==a and r['case']['target']==b)
        metric=sci(r['metrics']['dH'])
        suffix=r'^{\dagger}' if r['status']=='invalid' else ''
        vals.append(metric[:-1]+rf'\;({r["case"]["source"]}){suffix}$')
    body.append(paper_code(a)+' & '+' & '.join(vals)+r'\\')
manufactured = {r['run_id']: r['metrics'] for r in old if r['run_id'] in ('A_f0', 'A_f1')}
if manufactured:
    body.extend([r'\midrule', r'\multicolumn{5}{l}{In-family disk targets, $K=16$}\\',
                 r'Case & $d_H$ & $E_2$ & $E_{\nabla u}$ & $r_{\rm inv}$\\', r'\midrule'])
    for i in (0, 1):
        z = manufactured[f'A_f{i}']
        cells = [rf'$D\to A_{i}$'] + [sci(z[k]) for k in ('dH', 'E2', 'E_grad', 'r_inv')]
        body.append(' & '.join(cells) + r'\\')
main_table = table(
    r'OT tests with $m=256$ and $\lambda=0.4$. The crossed block reports $d_H$ at $K=24$ with source index in parentheses. The lower block gives the analytic-reference checks on the disk.',
    'tab:transport-cross', 'lcccc', r'Source $\backslash$ target & $S$ & $D$ & $T$ & '+f'${levels[-1]}$', '\n'.join(body))
grid={(r['case']['K'],r['case']['lam']):r for r in quick if r['case']['box_mode']=='full'}
grid.update({(r['case']['K'],r['case']['lam']):r for r in params})
grid[24,.4]=next(r for r in main if r['run_id']=='B_S_P_f0')
body=[]
for K in (16,24,32):
    body.append(str(K)+' & '+' & '.join(sci(grid[K,l]['metrics']['dH'])+' / '+sci(grid[K,l]['metrics']['r_inv']) for l in (.4,1.,3.))+r'\\')
param_table=table('Parameter sensitivity for S to T, f=1. Each cell is dH / r_inv.','tab:transport-sensitivity','lccc',r'$K\backslash\lambda$ & $0.4$ & $1$ & $3$','\n'.join(body))
by={r['run_id']:r for r in corner};body=[]
for a,b,f in [('S','P',0),('D','S',1),('D','P',1),('P','S',0),(levels[-1],'S',1),(levels[-1],'P',1)]:
    vals=[]
    for nu in (0,1,10):
        row=by[f'C_{a}_{b}_f{f}_n{nu}']
        m=row['metrics']
        suffix=r'$^{\dagger}$' if row['status']=='invalid' else ''
        vals.append(sci(m['dH'])+' / '+sci(m['corner_max'])+suffix)
    body.append(rf'${paper_code(a)}\to {paper_code(b)}$, {f} & '+' & '.join(vals)+r'\\')
corner_table=table('Corner comparison at K=16, lambda=0.4. Each cell is dH / c_max.','tab:transport-corner','lccc',r'Case, $f$ & $\nu=0$ & $\nu=1$ & $\nu=10$','\n'.join(body))
tables=[certtable,cmptable,rhstable,main_table,param_table,corner_table]
if args.ot_fdm_comparison or 'section44_fdm' in supp:
    comparison = json.loads((args.ot_fdm_comparison or ROOT/supp['section44_fdm']).read_text(encoding='utf-8'))
    refinement = sorted(comparison['rows'], key=lambda r: r['n'])
    body = []
    if comparison['case']=='B_S_D_f1':
        for r in refinement:
            z = r['fdm']
            area_error = r'---' if z['self_crossing'] else sci(z['relative_area_error'])
            cells = [f"FDM, $h=1/{r['n']}$", sci(z['dH']), area_error, r'---', f"{sum(r['seconds'].values()):.2f}"]
            body.append(' & '.join(cells) + r'\\')
        z = comparison['lens']['metrics']
        sec = comparison['lens']['setup_and_fit_seconds']
        inverse_residual = next(r['metrics']['r_inv'] for r in main if r['run_id']==comparison['case'])
        cells = [r'LENS, $K=24$, $m=256$', sci(z['dH']), sci(z['relative_area_error']),
                 sci(inverse_residual), f'{sec:.2f}']
        body.append(' & '.join(cells) + r'\\')
        tables.append(table(r'Finite differences and LENS for $S\to D$ with $f_1$. Times include setup and solve. The residual $r_{\rm inv}$ applies to the fitted latent variables of LENS.',
                            'tab:ot-fdm', 'lcccc', r'Method & $d_H$ & Relative area error & $r_{\rm inv}$ & time [s]', '\n'.join(body)))
    else:
        for r in refinement:
            z = r['fdm']
            cells = [f"FDM, $h=1/{r['n']}$", sci(z['E2']), sci(z['E_grad']), f"{sum(r['seconds'].values()):.2f}"]
            body.append(' & '.join(cells) + r'\\')
        z = refinement[-1]['lens']
        sec = comparison['lens']['setup_and_fit_seconds']
        body.append(' & '.join([r'LENS, $K=16$, $m=256$', sci(z['E2']), sci(z['E_grad']), f'{sec:.2f}']) + r'\\')
        tables.append(table(r'Traditional finite differences and LENS on $D\to A_1$. LENS errors use the valid centered-difference nodes of the finest FDM grid. Times include setup and solve.',
                            'tab:ot-fdm', 'lccc', r'Method & $E_2$ & $E_{\nabla u}$ & time [s]', '\n'.join(body)))
(BASE/'tables.tex').write_text('\n\n'.join(tables)+'\n',encoding='utf-8')
# Compare the numerical table bodies, independent of captions and table layout.
import re,hashlib
paper=(ROOT/'report/main25.tex').read_text(encoding='utf-8') if not args.no_paper_check else ''
def body_only(t):
    z=t[t.index(r'\midrule')+len(r'\midrule'):t.rindex(r'\bottomrule')]
    return re.sub(r'\s+','',z)
for t in ([] if args.no_paper_check else tables):
    label=re.search(r'\\label\{([^}]+)\}',t).group(1)
    pt=next(x for x in re.findall(r'\\begin\{table\}.*?\\end\{table\}',paper,re.S) if r'\label{'+label+'}' in x)
    assert body_only(t)==body_only(pt),'Paper/data mismatch: '+label
summary=dict(data_root=str(RELEASE) if RELEASE else 'historical',tables=len(tables),main_K=24,corner_K=16,parameter_cells=9,
    status_counts={s:sum(r['status']==s for r in main) for s in sorted({r['status'] for r in main})},
    paper_table_bodies_match=None if args.no_paper_check else True)
(BASE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

if args.figures:
    # Figures use the SAME recorded models and metrics as the tables.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for key,label in [('E2',r'$E_2$'),('Einf',r'$E_\infty$')]:
        ax[0].semilogy([r['K'] for r in geom],[r['metrics'][key] for r in geom],'o-',ms=3,label=label)
    ax[0].set(xlabel='Order K',ylabel='Relative error',xticks=list(range(4,33,4)));ax[0].legend();ax[0].grid(alpha=.25)
    for K in (8,16,24):
        rr=sorted([r for r in rows('resolution') if r['K']==K],key=lambda r:r['m'])
        ax[1].loglog([r['m'] for r in rr],[r['metrics']['E2'] for r in rr],'o-',ms=3,label=f'K={K}')
        ax[1].axvline(2*K+1,ls=':',lw=.7,alpha=.5)
    ax[1].set(xlabel='Boundary samples m',ylabel=r'$E_2$');ax[1].legend();ax[1].grid(alpha=.25)
    from matplotlib.ticker import FixedLocator,ScalarFormatter,NullFormatter
    ax[1].xaxis.set_major_locator(FixedLocator([16,32,64,128,256]))
    ax[1].xaxis.set_major_formatter(ScalarFormatter())
    ax[1].xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(BASE/'fig_s42_curves.pdf');fig.savefig(BASE/'fig_s42_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax,d in zip(axs,('square','disk')):
        f=np.load(DIR/f'comparison_{d}_radial_K16_m256.npz')
        e=(f['u']-f['exact'])/np.max(abs(f['exact']))
        v=np.max(abs(e));im=ax.tricontourf(f['x'],f['y'],e,levels=np.linspace(-v,v,25),cmap='RdBu_r')
        ax.set(title=d,aspect='equal',xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,format='%.1e')
    fig.savefig(BASE/'fig_s42_errormap.pdf');fig.savefig(BASE/'fig_s42_errormap.png',dpi=160);plt.close(fig)
