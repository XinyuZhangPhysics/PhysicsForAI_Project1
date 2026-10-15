# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
BASE = ROOT / 'outputs'
WORK = ROOT / 'work/scientific'
ARMS = ('S1', 'S2', 'S4', 'S8')
SEEDS = tuple(range(2026100101, 2026100109))
TIMES = (0, 16000, 64000, 256000)
FLOOR = 1e-20

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
integrity = json.loads((OUT / 'TRAINING_INTEGRITY.json').read_text())
checkpoints = json.loads((OUT / 'CHECKPOINT_HASHES.json').read_text())
if integrity['status'] != 'PASS' or checkpoints['status'] != 'FROZEN':
    raise RuntimeError('EXP_S1_BLOCKED_TRAINING_INTEGRITY')
if integrity['checkpoint_hashes_json_sha256'] != sha(OUT / 'CHECKPOINT_HASHES.json'):
    raise RuntimeError('CHECKPOINT_HASHES_CHANGED')
meta = json.loads((OUT / 'CALIBRATION_GEOMETRY.json').read_text())
assert meta['status'] == 'FROZEN'
assert sha(BASE / 'calibration_pair_manifest.jsonl') == meta['pair_manifest_sha256']
responses = []
failures = []
max_initial_qc = 0.0
max_response_residual = 0.0
max_factor_error = 0.0
max_s1_power = 0.0
for seed in SEEDS:
    for arm in ARMS:
        expected_K = meta['arms'][arm]['K_C_0']
        for t in TIMES:
            path = OUT / 'calibration_responses' / f'{arm}_seed_{seed}_t{t}.json'
            if not path.exists():
                failures.append(f'missing {arm} {seed} {t}')
                continue
            r = json.loads(path.read_text())
            responses.append({'arm': arm, 'seed': seed, 'horizon': t, 'sha256': sha(path)})
            if r['arm'] != arm or r['seed'] != seed or r['horizon_nonpadding_tokens'] != t:
                failures.append(f'identity {arm} {seed} {t}')
            if r['K_C_0'] != expected_K:
                failures.append(f'geometry drift {arm} {seed} {t}')
            if not all((math.isfinite(x) for x in (r['Q_C'], r['K_C_0'], r['P_B']))):
                failures.append(f'nonfinite {arm} {seed} {t}')
            max_response_residual = max(max_response_residual, r['response_projector_max_abs_error'])
            if r['factorization_abs_error'] is not None:
                max_factor_error = max(max_factor_error, r['factorization_abs_error'])
            if t == 0:
                max_initial_qc = max(max_initial_qc, r['Q_C'])
            if arm == 'S1':
                max_s1_power = max(max_s1_power, max(r['B_coordinate_powers']))
                if not r['S1_H1_all_coordinates_below_zero_floor']:
                    failures.append(f'S1 null {seed} {t}')
            elif r['factorization_abs_error'] is not None and r['factorization_abs_error'] > 1e-08 or (r['factorization_abs_error'] is None and r['P_B'] > FLOOR):
                failures.append(f'factorization {arm} {seed} {t}')
if len(responses) != 128:
    failures.append(f'response count {len(responses)}')
if max_initial_qc > FLOOR:
    failures.append(f'initial qC leakage {max_initial_qc}')
if max_response_residual > 1e-08:
    failures.append(f'response residual {max_response_residual}')
response_sha = hashlib.sha256(b''.join(((json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n').encode() for x in sorted(responses, key=lambda r: (r['seed'], r['arm'], r['horizon']))))).hexdigest()
code_names = ('response_s1.py', 'evaluate_s1.py', 'materialize_audit.py', 'freeze_calibration.py', 'freeze_training.py', 'train_s1.py')
code_hashes = {name: sha(WORK / name) for name in code_names}
result = {'status': 'CALIBRATION_FROZEN' if not failures else 'EXP_S1_BLOCKED_CALIBRATION', 'timestamp_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'training_integrity_sha256': sha(OUT / 'TRAINING_INTEGRITY.json'), 'checkpoint_hashes_sha256': sha(OUT / 'CHECKPOINT_HASHES.json'), 'checkpoint_aggregate_sha256': checkpoints['aggregate_sha256'], 'calibration_geometry_sha256': sha(OUT / 'CALIBRATION_GEOMETRY.json'), 'calibration_response_count': len(responses), 'calibration_response_aggregate_sha256': response_sha, 'S1_max_coordinate_power': max_s1_power, 'frozen_H1_floor': FLOOR, 'max_initial_Q_C': max_initial_qc, 'max_response_projector_abs_error': max_response_residual, 'max_factorization_abs_error': max_factor_error, 'analysis_code_sha256': code_hashes, 'failures': failures, 'no_audit_scientific_response_opened': not (OUT / 'audit_responses').exists()}
(OUT / 'CALIBRATION_FREEZE.json').write_text(json.dumps(result, indent=2) + '\n')
print(result['status'], 'responses', len(responses), 'failures', failures, flush=True)
if failures:
    raise SystemExit(2)
