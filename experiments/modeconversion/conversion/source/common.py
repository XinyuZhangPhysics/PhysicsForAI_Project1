# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, csv, datetime
import numpy as np
import torch
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[1]
HIST = Path(_p1_path('path_176b5fe52446'))
MATCH = Path(_p1_path('path_8a53678fb13c'))
R1 = Path(_p1_path('path_49eb03aa7e97'))
MODELBASE = Path(_p1_path('path_87e0c0eab882'))
MODELS = {'S1': MODELBASE / 'pythia-70m/a39f36b100fe8a5377810d56c3f4789b9c53ac42', 'S2': MODELBASE / 'SmolLM2-135M/28e66ca6931668447a3bac213f23d990ad3b0e2b'}
DATA = Path(_p1_path('path_8c5dea5cdb13'))
DB = Path(_p1_path('path_c94848b205c1'))
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
GRID = np.array([-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0])
AMPS = np.array([0.00125, 0.0025, 0.005, 0.01, 0.02])
torch.set_num_threads(4)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(x).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v: v.item()) + '\n')

def read(p):
    return json.loads(Path(p).read_text())

def table(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def proj(g):
    j4 = np.ones((4, 4)) / 4
    jg = np.ones((g, g)) / g
    return {'A': np.kron(np.eye(4) - j4, jg), 'C': np.kron(j4, np.eye(g) - jg), 'M': np.kron(np.eye(4) - j4, np.eye(g) - jg)}

def directions(f, g, seed):
    hidden = 512 if f == 'S1' else 576
    out = []
    for r in range(4):
        attempt = 0
        while True:
            key = f'v282M-r2-new-directions|{f}|{g}|{seed}|{r}|{attempt}'
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'little'))
            v = rng.standard_normal(hidden)
            v /= np.linalg.norm(v)
            z = rng.standard_normal(4 * g)
            es = {s: p @ z for s, p in proj(g).items()}
            if min((np.linalg.norm(e) for e in es.values())) > 1e-12:
                break
            attempt += 1
        es = {s: e / np.linalg.norm(e) for s, e in es.items()}
        assert max((abs(np.linalg.norm(np.outer(e, v)) - 1) for e in es.values())) < 1e-12
        assert max((np.linalg.norm(e - proj(g)[s] @ e) for s, e in es.items())) < 1e-12
        out.append((v, es, {'key': key, 'v_sha256': digest(v.tobytes()), 'e_sha256': {s: digest(e.tobytes()) for s, e in es.items()}}))
    return out

def bind():
    return read(ROOT / 'preregistration/INPUT_BINDINGS.json')

def weight(f, g, seed):
    x = next((x for x in bind()['copied_points'] if (x['family'], x['g'], x['seed']) == (f, g, seed)))
    assert sha(x['path']) == x['sha256']
    w = load_file(x['path'])['router.weight'].double().numpy()
    assert np.array_equal(w, w.reshape(4, g, -1)[:, 0].repeat(g, axis=0))
    return w.reshape(4, g, -1)[:, 0]

def seeds(f):
    return list(range(2026100101, 2026100109)) if f == 'S1' else list(range(2026110201, 2026110209))

def verify_freeze():
    p = ROOT / 'preregistration/FREEZE.json'
    lock = read(ROOT / 'preregistration/FREEZE_LOCK.json')
    assert sha(p) == lock['freeze_sha256']
    fr = read(p)
    for p, h in fr['asset_hashes'].items():
        assert sha(p) == h, p
    return fr
