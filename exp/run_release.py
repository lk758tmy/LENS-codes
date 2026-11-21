"""Portable power telemetry used by the published peanut baselines."""
import ctypes

def power_state():
    from ctypes import wintypes as w
    class P(ctypes.Structure):
        _fields_=[('ac',w.BYTE),('flag',w.BYTE),('percent',w.BYTE),('reserved',w.BYTE),('life',w.DWORD),('full',w.DWORD)]
    p=P()
    try:
        ok=ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))
        return dict(available=bool(ok),ac_line_status=int(p.ac),battery_percent=int(p.percent))
    except (AttributeError,OSError):
        return dict(available=False)
