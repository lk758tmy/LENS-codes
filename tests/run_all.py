"""Run the standalone mathematical regression scripts sequentially."""
from pathlib import Path
import os,subprocess,sys

root=Path(__file__).resolve().parents[1]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for path in sorted((root/'tests').glob('test_*.py')):
    print('\nRUN '+path.name,flush=True)
    subprocess.run([sys.executable,'-B',str(path)],cwd=root,env=env,check=True)
print('\nAll mathematical regression scripts passed.',flush=True)
