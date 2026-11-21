"""Environment pinning, timing and the unified JSONL record.

Every experiment in the reproducibility package goes through this module so that
* thread count, CPU model and library versions are recorded identically,
* timings are always split into ``build`` / ``fit`` / ``query``,
* every table can be regenerated from the JSONL log alone.

Importing :mod:`lens` calls :func:`pin_threads` *before* numpy/torch are imported,
so the BLAS thread count is fixed for the whole process.
"""
from __future__ import annotations

import json
import os
import platform
import time
from contextlib import contextmanager

# Fixed for the whole package (all methods, all tables).  Measured on
# Intel Core Ultra 9 285H (16 logical cores): one LENS LM step (K=24, m=512)
# takes 1.46 / 0.69 / 0.56 / 0.84 / 1.42 ms at 1/2/4/8/16 threads, while a
# 1024x256 float64 matmul takes 2.06 / 1.04 / 0.55 / 0.50 / 0.46 ms.
# 4 threads is the sweet spot for the small-batch LENS kernels and still gets
# 90% of the matmul speed-up, so it is used everywhere for consistency.
DEFAULT_THREADS = 4
_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def pin_threads(n: int | None = None) -> int:
    """Force the BLAS/OpenMP thread count (default 4, override with LENS_THREADS)."""
    if n is None:
        n = int(os.environ.get('LENS_THREADS', DEFAULT_THREADS))
    for v in _THREAD_VARS:
        os.environ[v] = str(int(n))
    return int(n)


def cpu_brand() -> str:
    try:                                        # Windows
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        return str(winreg.QueryValueEx(key, 'ProcessorNameString')[0]).strip()
    except Exception:                           # pragma: no cover - other OS
        return platform.processor() or platform.machine()


def env_info(threads: int) -> dict:
    import numpy
    import torch
    return dict(cpu=cpu_brand(), python=platform.python_version(),
                numpy=numpy.__version__, torch=torch.__version__,
                threads=int(threads))


class Timer:
    """Accumulates named timing segments (seconds)."""

    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    @contextmanager
    def seg(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, dt: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + float(dt)

    def as_dict(self) -> dict:
        return {k: round(v, 6) for k, v in self.t.items()}


def time_median(fn, reps: int = 5) -> tuple[float, object]:
    """Run ``fn`` ``reps`` times, return (median seconds, last result)."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], out


# ---------------------------------------------------------------- JSONL schema
# One line per run.  Keys are fixed; unused metrics are None (never omitted),
# so that pandas.read_json / make_tables.py see a rectangular table.
SCHEMA_KEYS = ('run_id', 'exp', 'case', 'method', 'hyper', 'metrics',
               'timing', 'env', 'status', 'notes')


def record(run_id: str, exp: str, case: dict, method: dict, hyper: dict,
           metrics: dict, timing: dict, env: dict,
           status: str = 'converged', notes: str = '') -> dict:
    return dict(run_id=run_id, exp=exp, case=case, method=method, hyper=hyper,
                metrics=metrics, timing=timing, env=env,
                status=status, notes=notes)


def append_jsonl(path: str, rec: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
