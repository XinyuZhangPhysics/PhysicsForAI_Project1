# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import hashlib, importlib.util, json, zipfile
from pathlib import Path
import numpy as np
RUN = Path(__file__).resolve().parents[1]
PRIOR = RUN.parent / 'P2D_RECOVERY_20261008_085240'
AUTHORITY = RUN / 'work/P2D_T1_NUMERICAL_IDENTITY_DISPOSITION.md'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def ident(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, o):
    with Path(p).open('x') as f:
        json.dump(o, f, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
        f.write('\n')

def add_npz(p, arrays, new=False):
    with zipfile.ZipFile(p, 'x' if new else 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for k, a in arrays.items():
            with z.open(k + '.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asarray(a), allow_pickle=False)

def engine(stage):
    s = importlib.util.spec_from_file_location('inherited_engine', PRIOR / 'work/tomography_engine.py')
    e = importlib.util.module_from_spec(s)
    s.loader.exec_module(e)
    e.RUN = RUN
    e.PKG = RUN / 'control_package/Project1_P2D_Codex_Control_Package_v1p1'
    e.initialize(stage)
    e.META.update(technical_disposition=ident(AUTHORITY), resume_prerequisite_audit=ident(RUN / 'T1_RESUME_PREREQUISITE_AUDIT.json'))
    e.META['implementation_sources'] += [ident(Path(__file__))]
    return e
