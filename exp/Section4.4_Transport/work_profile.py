"""Read-only work counters. Wrappers return the original solver outputs."""
import time,ctypes,collections
from ctypes import wintypes as w

def power_state():
    class P(ctypes.Structure):
        _fields_=[('ac',w.BYTE),('flag',w.BYTE),('percent',w.BYTE),('reserved',w.BYTE),('life',w.DWORD),('full',w.DWORD)]
    p=P()
    try:
        ok=ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
        return dict(available=bool(ok),ac_line_status=int(p.ac),battery_percent=int(p.percent),battery_flag=int(p.flag))
    except Exception:return dict(available=False)

class WorkProfile:
    def __init__(self):
        self.c=collections.Counter();self.t=collections.Counter();self.stages=[]
    def __enter__(self):
        from lens import constrained_lm as q,lm,res
        self.q,self.lm,self.res=q,lm,res
        self.old=(q.solve_qp,q._working_qp,lm.solve,res.TransportBVP.eval)
        def qp(*a,**k):
            start=time.perf_counter();cpu=time.process_time();z,rec=self.old[0](*a,**k)
            self.c['qp_calls']+=1;self.c['qp_internal_iterations']+=rec['qp_iterations'];self.c['qp_constraint_cuts']+=rec.get('qp_cuts',0)
            self.c['qp_unconstrained_shortcuts']+=int(rec['qp_iterations']==0)
            self.t['qp_wall']+=time.perf_counter()-start;self.t['qp_cpu']+=time.process_time()-cpu
            return z,rec
        def working(*a,**k):
            z,rec=self.old[1](*a,**k);self.c['working_qp_calls']+=1
            self.c['working_qp_cap_hits']+=int(rec['qp_iterations']>=k.get('maxit',160))
            return z,rec
        def evaluate(prob,*a,**k):
            start=time.perf_counter();ans=self.old[3](prob,*a,**k)
            self.c['eval_with_jacobian' if k.get('need_jac',a[2] if len(a)>2 else True) else 'eval_without_jacobian']+=1
            self.t['residual_wall']+=time.perf_counter()-start;return ans
        def solve(*a,**k):
            start=time.perf_counter();cpu=time.process_time();before=self.c.copy();bt=self.t.copy()
            ans=self.old[2](*a,**k);h=ans['hist']
            rec=dict(stage=len(self.stages),outer_iterations=len(h),accepted=sum(x['accepted'] for x in h),wall=time.perf_counter()-start,cpu=time.process_time()-cpu)
            rec.update({key:val-before[key] for key,val in self.c.items()});rec.update({key:val-bt[key] for key,val in self.t.items()})
            self.stages.append(rec);return ans
        q.solve_qp=qp;q._working_qp=working;lm.solve=solve;res.TransportBVP.eval=evaluate
        self.start=time.perf_counter();self.cpu=time.process_time();return self
    def __exit__(self,*exc):
        self.wall=time.perf_counter()-self.start;self.cpu_elapsed=time.process_time()-self.cpu
        self.q.solve_qp,self.q._working_qp,self.lm.solve,self.res.TransportBVP.eval=self.old
    def summary(self):return dict(counters=dict(self.c),timers=dict(self.t),stages=self.stages,fit_call_wall=self.wall,fit_call_cpu=self.cpu_elapsed)
