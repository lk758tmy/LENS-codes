"""Source derivatives include every recurrence level and the base primitive."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens,numpy as np,torch
from lens.inverse_basis import SourceGenerator

b=SourceGenerator(4,6,.8)
eta=torch.tensor([1.,.04,-.12,.015,.01]);st=b.build(eta)
torch.manual_seed(813);theta=.01*torch.randn(b.nm);gamma=1.1
a,da=b.omega_coefficients(st,theta,gamma)
x=torch.linspace(.04,.96,19);t=torch.linspace(-.65,.57,19)
v=b.evaluate(a,x,t);src=b.source_values(eta,x)
assert float((v['wxx']+src*v['wtt']).abs().max())<1e-10
dv=b.evaluate(da,x,t)
for j in range(b.ns):
    e=torch.zeros_like(eta);e[j]=2e-6
    pp=b.omega_coefficients(b.build(eta+e,False),theta,gamma,False)[0]
    mm=b.omega_coefficients(b.build(eta-e,False),theta,gamma,False)[0]
    vp=b.evaluate(pp,x,t);vm=b.evaluate(mm,x,t)
    for key in ('w','wx','wt','wtt','wxt'):
        fd=(vp[key]-vm[key])/(4e-6)
        assert float((fd-dv[key][j]).abs().max())<2e-8,(j,key)
# Polynomial identity and AD through independent physical x/t evaluation.
xx=x.clone().requires_grad_();tt=t.clone().requires_grad_();ev=b.evaluate(a,xx,tt)
wxx=torch.autograd.grad(ev['wx'].sum(),xx,retain_graph=True)[0]
wtt=torch.autograd.grad(ev['wt'].sum(),tt)[0]
assert float((wxx+src*wtt).abs().max())<1e-10
print('PASS: fixed-layout source generator, exact source Jacobian columns, independent physical AD PDE identity')
