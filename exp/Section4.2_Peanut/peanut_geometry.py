"""Exact legacy polygon and analytic rounded-end comparison; no core edits."""
import numpy as np
from lens import regions

class Peanut(regions.PolygonRegion):
    def __init__(self):
        r=regions.peanut();super().__init__(r.V.copy(),'peanut')
        self.knots=r.V[:len(r.V)//2,0];self.heights=-r.V[:len(r.V)//2,1]
    def contains(self,x,y):
        x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
        # Exactly the same piecewise-linear upper/lower polygon, evaluated in O(N).
        return (np.abs(x)<=1)&(np.abs(y)<=np.interp(x,self.knots,self.heights))

class RoundedPeanut(regions.ParamRegion):
    name='peanut_rounded';bbox=(-1.,1.,-.85,.85)
    def _xy(self,th):
        c,s=np.cos(th),np.sin(th);r=(c**6+s**6)**(-1/6)
        x=r*c;return x,.85*(1-.55*np.exp(-(x/.35)**2))*r*s
    def contains(self,x,y):
        x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
        h=.85*(1-.55*np.exp(-(x/.35)**2))
        return (x**6+(y/h)**6)<=1+2e-14

def make(name):return Peanut() if name=='peanut' else RoundedPeanut() if name=='peanut_rounded' else None
