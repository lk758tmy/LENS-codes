"""High-degree source differential commutation and physical roughness units."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens,numpy as np,torch
from lens.inverse_basis import SourceGenerator
from lens.inverse_fit import InverseProblem
from lens.inverse_data import observation_angles

_,angles,_=observation_angles();q=np.full((3,len(angles)),.5)
p=InverseProblem(q,(0,1,2),16,24)
# x=(1+xi)/2: 1+x^2=11/8+(1/2)T1+(1/8)T2, f''=2.
eta=torch.zeros(17);eta[:3]=torch.tensor([11/8,1/2,1/8])
assert abs(float(torch.linalg.vector_norm(p.R@eta)**2)-4.)<2e-12

G=SourceGenerator(16,24,.6)
eta=torch.zeros(17);eta[:7]=torch.tensor([1.,.05,-.12,.02,.03,-.01,.005])
theta=torch.tensor(np.random.default_rng(17).normal(size=G.nm))*1e-3
full=G.build(eta,True);effective=G.build(eta,False)
af,df=G.omega_coefficients(full,theta,1.1)
ae,de=G.omega_coefficients(effective,theta,1.1)
assert torch.equal(af,ae)
assert float((df-de).abs().max())<1e-10*(1+float(df.abs().max()))
x=torch.linspace(.01,.99,51);t=torch.linspace(-.5,.5,51)
analytic=G.evaluate(de,x,t)['w']
for j in (0,5,12,16):
    step=torch.zeros_like(eta);step[j]=2e-6
    ap,_=G.omega_coefficients(G.build(eta+step,False),theta,1.1,False)
    am,_=G.omega_coefficients(G.build(eta-step,False),theta,1.1,False)
    fd=(G.evaluate(ap,x,t)['w']-G.evaluate(am,x,t)['w'])/(4e-6)
    assert float((fd-analytic[j]).abs().max())<2e-7
print('PASS: high-degree effective/full source derivatives, physical source FD, physical roughness integral')
