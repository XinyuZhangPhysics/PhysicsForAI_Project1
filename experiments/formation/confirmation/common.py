# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, csv, datetime
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/P2M_MobilityConfirmation'
PKG = ROOT / 'work/Project1_P2M_MobilityConfirmation_Codex_Package_v1'
SOURCE = Path(__file__).parent
OLD = ROOT.parent / 'jie'
J = ROOT.parent / 'jie-2/outputs/P2J1_RESUME_20261008_205226'
COUNTERS = dict(NEW_MODEL_DOWNLOAD_BYTES=0, NEW_DATA_DOWNLOAD_BYTES=0, NEW_BANK_COUNT=0, OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT=0, MODEL_FILES_NEWLY_DOWNLOADED=0, DATA_FILES_NEWLY_DOWNLOADED=0, BYTES_NEWLY_DOWNLOADED=0)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def csvread(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))

def write(n, d):
    p = OUT / n
    assert not p.exists(), p
    p.write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
    p.chmod(292)

def textout(n, s):
    p = OUT / n
    assert not p.exists(), p
    p.write_text(s)
    p.chmod(292)

def csvout(n, rows):
    p = OUT / n
    assert not p.exists(), p
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    p.chmod(292)

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify_seal():
    seal = read(OUT / 'P2L_PREFLIGHT_SEAL.json')
    assert seal['status'] == 'PASS'
    for p, h in seal['sealed_sha256'].items():
        assert sha(p) == h, p
    return seal
