"""Mesh-refined, fixed-Dirichlet FDM labels; no inverse training here."""
from pathlib import Path
import sys,os,argparse,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LENS_THREADS','4')
import lens,numpy as np
from lens.inverse_data import InverseSource,Control,observation_angles,reference_record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent/'out/reference_v1')
    ap.add_argument('--sources',default='C,A');ap.add_argument('--ns',default='80,160,320');ap.add_argument('--controls',default='0,1,2');a=ap.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    _,sensor,valid=observation_angles();angles=np.r_[sensor,valid]
    files=[ROOT/'refs/fdm.py',ROOT/'lens/inverse_data.py',Path(__file__)]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    meta=out/'source_hashes.json'
    if meta.exists() and json.loads(meta.read_text())!=hashes:raise ValueError('Reference version changed; use a new output directory')
    meta.write_text(json.dumps(hashes,indent=2))
    for src in a.sources.split(','):
        for ell in map(int,a.controls.split(',')):
            for n in map(int,a.ns.split(',')):
                rid=f'{src}_g{ell}_n{n}';cp=out/(rid+'.npz')
                if cp.exists():print('SKIP',rid,flush=True);continue
                print('BEGIN',rid,flush=True)
                start=time.perf_counter()
                p,q,r=reference_record(InverseSource(src),Control(ell),n,angles)
                r.update(run_id=rid,version=version,wall=time.perf_counter()-start)
                np.savez_compressed(cp,x=p.xs,y=p.ys,u=p.U,inside=p.ins,unknown=p.intm,
                                    angles=angles,q=q,sensor_angles=sensor,validation_angles=valid)
                (out/(rid+'.json')).write_text(json.dumps(r,indent=2))
                with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
                print('RESULT',json.dumps(r),flush=True)
    print('DONE',flush=True)

if __name__=='__main__':main()
