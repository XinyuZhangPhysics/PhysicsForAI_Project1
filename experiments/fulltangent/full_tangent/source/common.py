# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, hashlib, csv, math, datetime, zoneinfo
import numpy as np, torch
from torch.nn import functional as F
from safetensors.torch import load_file
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'outputs/v283M_r1'
ROOT = Path(_p1_path('path_8c32eaf570f6'))
MIG = BASE / 'work/PhysicsForAI_Project1_Migration_v283M'
MODELS = {'S1': Path(_p1_path('path_44ab72c5d90e')), 'S2': Path(_p1_path('path_3397fdd42660'))}
DIM = {'S1': (512, 2048), 'S2': (576, 1536)}

def read(p):
    return json.loads(Path(p).read_text())

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def csvwrite(n, rows):
    with (OUT / n).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def state(fam):
    all = load_file(str(MODELS[fam] / 'model.safetensors'))
    prefix = 'gpt_neox.layers.3.mlp.' if fam == 'S1' else 'model.layers.19.mlp.'
    return {k[len(prefix):]: v.float() for k, v in all.items() if k.startswith(prefix)}

def features(h, s, fam, device='cpu', dtype=torch.float64):
    h = torch.as_tensor(h, dtype=dtype, device=device)
    s = {k: v.to(device=device, dtype=dtype) for k, v in s.items()}
    if fam == 'S1':
        z = F.linear(h, s['dense_h_to_4h.weight'], s['dense_h_to_4h.bias'])
        return F.gelu(z, approximate='none')
    gate = F.silu(F.linear(h, s['gate_proj.weight']))
    up = F.linear(h, s['up_proj.weight'])
    return gate * up

def qc_from_h(h, w, g, device='mps'):
    with torch.no_grad():
        probs = F.softmax(F.linear(torch.from_numpy(h).to(device), w.to(device)), dim=-1).cpu().double().numpy()
    return probs

def project(q, g):
    v = q.reshape(-1, 4, g)
    avg = v.mean(axis=(1, 2), keepdims=True)
    a = np.broadcast_to(v.mean(axis=2, keepdims=True) - avg, v.shape).reshape(-1, 4 * g).copy()
    c = np.broadcast_to(v.mean(axis=1, keepdims=True) - avg, v.shape).reshape(-1, 4 * g).copy()
    m = q - q.mean(axis=1, keepdims=True) - a - c
    return (a, c, m)

def center_child(child):
    n, g = child.shape
    k = np.broadcast_to(child[:, None, :], (n, 4, g)).reshape(n, 4 * g).copy()
    return (k, project(k, g))

def coords(q, k, group):
    vals = np.sum(q * k, axis=1)
    return np.array([vals[group == i].mean() for i in range(12)])
