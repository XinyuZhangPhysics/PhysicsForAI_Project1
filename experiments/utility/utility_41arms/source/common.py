# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib, json, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / 'review'
BUNDLE.mkdir(exist_ok=True)
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
ARMS = ['BASELINE', 'CANDIDATE'] + [f'RANDOM_{i:03d}' for i in range(39)]
MODEL_SHA = 'e5642d2d7b37c138775ddf59f685f91788e173b7'
PROJECTS = Path(_p1_path('path_65e87eb57756'))
BANK = PROJECTS / '2026-10-05/j-y-2/outputs/Project1_R2_v162_LocalArtifacts'
FRAME = BANK / 'BankV_candidate_frame_v162.jsonl'
CACHE = PROJECTS / '2026-10-04/jie/work/hf-cache'
SNAPSHOT = CACHE / f'models--allenai--OLMoE-1B-7B-0924/snapshots/{MODEL_SHA}'
NULLS = PROJECTS / '2026-10-06/jie/outputs/Project1_R2_v164_LocalArtifacts'
PILOT = PROJECTS / '2026-10-04/jie/outputs/Project1_R2_Pilot'
HISTORY = dict(V175_RAW_CANDIDATE_FRAME_ACCESS_OCCURRED=True, V175_HELDOUT_FIRST_ROW_DISPLAYED=True, V175_RESERVE_BYTES_TRAVERSED=True, V175_MODEL_FORWARDS=0, V175_Q3_OUTCOMES_GENERATED=False)

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(value).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b'\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def info(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

def event(kind, **fields):
    append_event(BUNDLE / 'execution_events.jsonl', dict(event=kind, **fields))
_chain_cache = {}

def append_event(path, fields):
    import fcntl
    with Path(path).open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        key = str(Path(path).resolve())
        size = os.fstat(f.fileno()).st_size
        cached = _chain_cache.get(key)
        if cached is not None and cached[0] == size:
            _, previous, seq = cached
        else:
            previous = '0' * 64
            seq = 0
            for line in f:
                row = json.loads(line)
                claimed = row.pop('event_sha256')
                if row['previous_event_sha256'] != previous or row['sequence'] != seq or digest(canonical(row)) != claimed:
                    raise RuntimeError('EVENT_CHAIN_INTEGRITY_FAILURE')
                previous = claimed
                seq += 1
        row = dict(time_UTC=now(), sequence=seq, previous_event_sha256=previous, **fields)
        row['event_sha256'] = digest(canonical(row))
        f.seek(0, 2)
        f.write(canonical(row) + b'\n')
        f.flush()
        os.fsync(f.fileno())
        _chain_cache[key] = (f.tell(), row['event_sha256'], seq + 1)

def frozen():
    p = BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'
    if sha(p) != (BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.sha256').read_text().split()[0]:
        raise RuntimeError('FREEZE_INTEGRITY_FAILURE')
    return read(p)
