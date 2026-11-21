"""Equality-gamma ablation with identical physical starts and fixed boxes.

The boxes are obtained from the SAME data-driven coarse LM used by the formal
driver. They are fixed for both coordinate systems during this comparison.
"""
import run_formal as formal
from run_formal import *

def main():
  global OUT,LOG,DONE
  ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
  formal.configure_output(a.out);OUT,LOG,DONE=formal.OUT,formal.LOG,formal.DONE
  records=[json.loads(x) for x in LOG.read_text(encoding='utf-8').splitlines()]
  for domain,K,kind in [('square',12,None),('square',16,None),('disk',3,'quad'),('disk',16,'quad')]:
    base=next(r for r in records if r['group']==('rhs' if kind else 'comparison') and r['domain']==domain and (not kind or r['source']==kind) and r['method']=='LENS')
    r=regions.get_region(domain);s=sources.Source(kind or 'const',**({'a':2} if kind else {'c':1}))
    tr=truth.SeparableTruth(s,alpha=1,y0=.5) if kind else truth.RadialTruth()
    x,y=r.boundary(256);g=tr(x,y)[0]
    ii=init.boundary_reference(r,x,y,g,s,K=K,basis_kind='fx' if kind else 'const',t_box=base['box'][2:])
    b=ii['basis'];ch=chart.Chart(b,cert.Certificate(b.cert_S(61),delta=.1,iota=.02))
    th=ii['theta0'];h,GG=ch.equality_coordinates(th)
    tt,_=ch.solve_t(torch.as_tensor(x),torch.as_tensor(y),th)
    for mode,theta in [('native_equality',h),('physical_coordinates',th)]:
      id=f'fixedbox_{domain}_{kind or "radial"}_K{K}_{mode}'
      if id in DONE:continue
      pr=res.DirichletBVP(ch,x,y,g,lam=.4,mu=0,tau_mode='joint',gamma_mode='cert' if mode=='native_equality' else 'gauge')
      tic=time.perf_counter();o=lm.solve(pr,theta,tt,rounds=260,ftol=1e-26*len(x));sec=time.perf_counter()-tic
      X,Y=r.interior_grid(65);fld=metrics.field_on_grid(ch,o['theta'],X,Y);ev=pr.eval(o['theta'],o['t'])
      emit(dict(run_id=id,group='fixedbox',method=mode,domain=domain,K=K,source=kind or 'radial',
                box=b.box,metrics=dict(**metrics.errors(fld['u'],tr(X,Y)[0]),r1=ev['r1'],r2=ev['r2'],margin=ch.margin(o['theta']),n_out=fld['n_out']),timing=dict(fit=sec),iterations=o['iters']))

if __name__=='__main__':main()
