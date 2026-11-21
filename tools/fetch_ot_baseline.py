"""Fetch the single pinned upstream module required for the OT FDM rerun."""
from pathlib import Path
import hashlib
import urllib.request

COMMIT='d84acdc3d18614a174ff6b4b39b3d3aaa01396ee'
SHA256='9c6b641d248061a36fb1f2bab19b2c00dd866ce33bba45b981e669d5fc4c8b41'
URL=f'https://raw.githubusercontent.com/gbonnet1/ma-ot-monotone/{COMMIT}/main/ma.py'
destination=Path(__file__).resolve().parents[1]/'third_party/ma-ot-monotone/main/ma.py'
data=destination.read_bytes() if destination.is_file() else urllib.request.urlopen(URL,timeout=60).read()
if hashlib.sha256(data).hexdigest()!=SHA256:raise SystemExit('Upstream SHA-256 mismatch; no file was written.')
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_bytes(data)
print('Verified upstream ma.py at commit '+COMMIT)
