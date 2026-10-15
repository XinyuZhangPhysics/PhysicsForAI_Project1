# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Serial stage dispatch, each child independently verifies all predecessor seals."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, subprocess, sys, time
RUN = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def verified(name):
    p = RUN / (name + '.json')
    expected = (RUN / (name + '.sha256')).read_text().split()[0]
    assert sha(p) == expected
    d = json.loads(p.read_text())
    for a in d['artifacts'].values():
        assert sha(Path(a['path'])) == a['sha256']
    return d
print('WAITING FOR COMPLETE T1 PREDICTION SEAL', flush=True)
while not (RUN / 'T1_PREDICTION_SEAL.sha256').is_file():
    log = RUN / 'logs/T1_RESUME_V2_STDOUT.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text():
        raise RuntimeError('T1 v2 failed; no later stage dispatched')
    time.sleep(2)
verified('T1_PREDICTION_SEAL')
for stage, gate in [('T2', 'T1_PREDICTION_SEAL'), ('T3', 'T2_STAGE_SEAL'), ('T4', 'T3_STAGE_SEAL')]:
    verified(gate)
    script = RUN / 'work' / ('final_t4.py' if stage == 'T4' else 'production_resume_v2.py')
    cmd = [sys.executable, str(script)] + ([] if stage == 'T4' else [stage])
    print('DISPATCH', stage, 'PREDECESSOR HASH VERIFIED', flush=True)
    with (RUN / 'logs' / (stage + '_STDOUT.log')).open('x') as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if ret.returncode:
        raise RuntimeError((stage, 'EXECUTION_BLOCKED', 'exit_code', ret.returncode))
    print('STAGE CLOSED', stage, flush=True)
print('T2-T4 COMPLETE; logs closed; delivery hash finalizer may now run', flush=True)
