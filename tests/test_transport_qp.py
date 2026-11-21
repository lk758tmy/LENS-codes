import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens
import numpy as np,torch
from scipy.optimize import minimize
from lens.constrained_lm import solve_qp
rng=np.random.default_rng(17)
for trial in range(20):
    A=rng.normal(size=(6,6));H=A.T@A+.2*np.eye(6);g=rng.normal(size=6)
    C=rng.normal(size=(20,6));b=rng.uniform(.05,.5,20)
    z,info=solve_qp(torch.tensor(H),torch.tensor(g),torch.tensor(C),torch.tensor(b))
    z=z.numpy();f=lambda x:.5*x@H@x-g@x
    ref=minimize(f,np.zeros(6),jac=lambda x:H@x-g,method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:b-C@x,jac=lambda x:-C)],options=dict(ftol=1e-13,maxiter=200))
    assert max(C@z-b)<1e-8,(trial,info)
    assert abs(f(z)-ref.fun)<1e-8,(trial,info,f(z),ref.fun)
print('PASS: 20 convex damped-QP comparisons with an independent constrained optimizer')
