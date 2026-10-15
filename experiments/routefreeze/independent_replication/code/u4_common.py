# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""U4 OLMoE final-layer sparse-MoE replication — common utilities.

Independent second-model reimplementation. Written from the mathematical / statistical
specification in `PhysicsForAI_Project1_U4_ComputationalPreregistration_v1_FROZEN.md` and
`PhysicsForAI_Project1_U4_TheoryNote_v1.md`. No Codex implementation file was read.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
TREE = Path(__file__).resolve().parent.parent
PATHS = {'tree': TREE, 'code': TREE / 'code', 'raw': TREE / 'raw', 'raw_tables': TREE / 'raw' / 'tables', 'reports': TREE / 'reports', 'provenance': TREE / 'provenance', 'logs': TREE / 'logs', 'figures': TREE / 'figures', 'tests': TREE / 'tests', 'config': TREE / 'replication_config.json', 'model_dir': Path(_p1_path('path_f2699b70cfbf')), 'data_dir': Path(_p1_path('path_f6b2b72181c3')), 'reference_docs': TREE / 'provenance' / 'reference_docs'}

def ensure_dirs() -> None:
    for k in ('raw', 'raw_tables', 'reports', 'provenance', 'logs', 'figures', 'raw_superseded'):
        p = TREE / 'raw' / 'superseded' if k == 'raw_superseded' else PATHS.get(k, TREE / k)
        if not Path(p).is_dir():
            Path(p).mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_text_utf8(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def source_tree_digest(root: Path, subdir: str='code') -> dict:
    """Deterministic digest of every file under `root/subdir` (sorted by relpath)."""
    base = root / subdir
    entries = []
    for p in sorted(base.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            rel = str(p.relative_to(root))
            entries.append((rel, sha256_file(p), p.stat().st_size))
    h = hashlib.sha256()
    for rel, dig, size in entries:
        h.update(f'{rel}\x00{dig}\x00{size}\n'.encode())
    return {'n_files': len(entries), 'files': entries, 'TREE_DIGEST_SHA256': h.hexdigest()}

def load_config() -> dict:
    with open(PATHS['config'], 'r', encoding='utf-8') as fh:
        return json.load(fh)

def json_dump(obj, path) -> None:
    """np-safe JSON dump (audit scripts can die on raw ndarray)."""
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_tolist(obj), fh, indent=2, sort_keys=False)
        fh.write('\n')

def _tolist(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return _tolist(o.tolist())
    if isinstance(o, dict):
        return {str(k): _tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_tolist(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

def write_csv(rows, path, header=None) -> None:
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if header is not None:
            w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))

def now_utc() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def torch_setup(threads: int | None=None, seed: int | None=None):
    import torch
    if threads is None:
        threads = max(1, os.cpu_count() or 8)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if seed is not None:
        torch.manual_seed(seed)
    return torch

def git_rev(path: Path) -> str:
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:
        return f'<unavailable: {exc}>'

def log(msg: str) -> None:
    sys.stdout.write(f'[{now_utc()}] {msg}\n')
    sys.stdout.flush()
