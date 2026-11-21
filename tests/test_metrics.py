"""Dirichlet constants must count as error; OT additive gauges must not."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens,numpy as np
from lens import metrics

u=np.array([1.,2.,4.,7.]);v=u+.5
d=metrics.errors(v,u)
assert np.isclose(d['E2'],.5/np.sqrt(np.mean(u*u)))
assert np.isclose(d['Einf'],.5/7)
assert d['E2']>0
ot=metrics.errors(v,u,center_reference=True,align_mean=True)
assert ot['E2']==0 and ot['Einf']==0
e=np.array([.2,-.1,.4,-.3])
a=metrics.errors(u+e,u,center_reference=True,align_mean=True)
b=metrics.errors(u+e+13,u-5,center_reference=True,align_mean=True)
assert np.isclose(a['E2'],b['E2']) and np.isclose(a['Einf'],b['Einf'])
assert metrics.errors(np.array([3.,3.]),np.array([2.,2.]))['E2']==.5
print('PASS: ordinary Dirichlet relative errors; OT mean alignment and independent additive-gauge invariance')
