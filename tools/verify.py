"""Verify packaged files using only the Python standard library."""
from pathlib import Path
import ast
import hashlib
import json

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for name,want in manifest['files'].items():
    path=root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=want:
        raise SystemExit('Missing or changed packaged file: '+name)
for path in root.rglob('*.py'):
    if any(part in ('third_party','reruns','reproduced') for part in path.relative_to(root).parts):continue
    ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path.relative_to(root)))
print(f"PASS: {len(manifest['files'])} packaged files and all Python syntax verified.")
