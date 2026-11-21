"""Full shared-source/field/latent Jacobian and Schur normal equations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens,numpy as np,torch
from lens.inverse_fit import InverseProblem
from lens.inverse_data import observation_angles

_,a,_=observation_angles()
q=np.array([.5*np.ones_like(a),.5*(np.sqrt(1+.3**2)+.3*np.cos(2*a)),
            .5*np.sqrt(1+.3**2)+.15*np.sin(2*a)])
p=InverseProblem(q,(0,1,2),3,4,tmax=.85)
z,t=p.initial();z[1]=.01;z[2]=-.015
assert p.feasible(z)
v=p.evaluate(z,t);eps=2e-6
for j in (0,1,3,p.neta,p.neta+2,p.neta+p.nm+3,p.p-1):
    e=torch.zeros_like(z);e[j]=eps
    fd=(p.evaluate(z+e,t,False)['r']-p.evaluate(z-e,t,False)['r'])/(2*eps)
    assert float((fd-v['J'][:,j]).abs().max())<3e-8,(j,float((fd-v['J'][:,j]).abs().max()))
for j in (5,130,200,325,550):
    e=torch.zeros_like(t);e[j]=eps
    fd=(p.evaluate(z,t+e,False)['r']-p.evaluate(z,t-e,False)['r'])/(2*eps)
    expected=torch.zeros_like(fd)
    for c in range(3):expected[c*p.m+j]=v['jt'][c,j]
    assert float((fd-expected).abs().max())<3e-8
J=v['J'];jt=v['jt'];r=v['r'];B=torch.cat([torch.diag(c) for c in jt]);full=torch.cat([J,B],1)
kap=.03;A=J.T@J;scale=A.diagonal().clamp(min=1e-30);P=torch.diag(torch.cat([scale,torch.ones(p.m)]))
answer=torch.linalg.solve(full.T@full+kap*P,-full.T@r)
M=J.T@B;D=(jt*jt).sum(0);g=J.T@r;gt=B.T@r;inv=1/(D+kap)
dz=torch.linalg.solve(A+kap*torch.diag(scale)-(M*inv)@M.T,-g+M@(gt*inv))
dt=-(gt+M.T@dz)*inv
assert float((torch.cat([dz,dt])-answer).abs().max())<1e-9
print('PASS: all shared-source and field columns, sensor/non-sensor latent columns, full vs Schur damped step')

# Constraint slacks b(z) satisfy b(z+d)=b(z)-C(z)d+O(||d||^2).
con,slack=p.linear_constraints(z)
for j in (0,2,p.neta+2,p.neta+p.nm+3,p.p-1):
    e=torch.zeros_like(z);e[j]=eps
    _,bp=p.linear_constraints(z+e);_,bm=p.linear_constraints(z-e)
    assert float(((bp-bm)/(2*eps)+con[:,j]).abs().max())<2e-7
print('PASS: source/field curvature and coverage constraint derivatives')
