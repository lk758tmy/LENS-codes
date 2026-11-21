"""Independent boundary-flux reconstruction: quadratic exactness and refinement."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import lens,numpy as np
from lens.inverse_data import Control,InverseSource,reference_record,boundary

angles=2*np.pi*(np.arange(67)+.37)/67
for ell in (0,1,2):
    p,q,r=reference_record(InverseSource('one'),Control(ell),32,angles)
    v=np.sqrt(1+.3**2)
    A=np.eye(2) if ell==0 else np.diag([v+.3,v-.3]) if ell==1 else np.array([[v,.3],[.3,v]])
    normal=np.column_stack([np.cos(angles),np.sin(angles)])
    expected=.5*np.einsum('ij,jk,ik->i',normal,A,normal)
    assert r['status']=='converged'
    assert np.max(abs(q-expected))<2e-9,(ell,np.max(abs(q-expected)))
print('PASS: independent FDM and boundary-vanishing flux regression reproduce all three constant-source exact quadratic controls')
