# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
BASE = ROOT / 'outputs'
WORK = ROOT / 'work/scientific'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
record_path = OUT / 'AUDIT_OPENING_RECORD.json'
if record_path.exists():
    raise RuntimeError('AUDIT_ALREADY_OPENED')
cal = json.loads((OUT / 'CALIBRATION_FREEZE.json').read_text())
integ = json.loads((OUT / 'TRAINING_INTEGRITY.json').read_text())
check = json.loads((OUT / 'CHECKPOINT_HASHES.json').read_text())
assert cal['status'] == 'CALIBRATION_FROZEN' and cal['no_audit_scientific_response_opened']
assert integ['status'] == 'PASS' and integ['run_count'] == 32
assert check['status'] == 'FROZEN' and check['checkpoint_count'] == 128
assert cal['checkpoint_aggregate_sha256'] == check['aggregate_sha256']
assert cal['checkpoint_hashes_sha256'] == sha(OUT / 'CHECKPOINT_HASHES.json')
assert cal['training_integrity_sha256'] == sha(OUT / 'TRAINING_INTEGRITY.json')
for name, expected in cal['analysis_code_sha256'].items():
    assert sha(WORK / name) == expected
assert not (OUT / 'audit_responses').exists()
assert not (OUT / 'AUDIT_REPRESENTATION.json').exists()
assert not (OUT / 'AUDIT_GEOMETRY.json').exists()
assert not (OUT / 'AUDIT_PRIMARY_64K.json').exists()
assert not (OUT / 'FORMAL_LABEL.json').exists()
assert not (ROOT / 'work/s1_audit_representation.pt').exists()
data = json.loads((BASE / 'DATA_MANIFEST_HASHES.json').read_text())
train = json.loads((BASE / 'TRAINING_MANIFEST_HASHES.json').read_text())
assert sha(BASE / 'audit_bank_manifest.jsonl') == data['manifest_sha256']['audit']
assert sha(BASE / 'audit_pair_manifest.jsonl') == data['manifest_sha256']['audit_pairs']
record = {'status': 'AUDIT_OPENED_AFTER_FROZEN_PREREQUISITES', 'timestamp_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'training_integrity_sha256': sha(OUT / 'TRAINING_INTEGRITY.json'), 'checkpoint_hashes_sha256': sha(OUT / 'CHECKPOINT_HASHES.json'), 'checkpoint_aggregate_sha256': check['aggregate_sha256'], 'calibration_freeze_sha256': sha(OUT / 'CALIBRATION_FREEZE.json'), 'calibration_response_aggregate_sha256': cal['calibration_response_aggregate_sha256'], 'analysis_code_sha256': cal['analysis_code_sha256'], 'manifest_sha256': {'audit_bank': sha(BASE / 'audit_bank_manifest.jsonl'), 'audit_pairs': sha(BASE / 'audit_pair_manifest.jsonl'), 'training_stream_index': sha(BASE / 'TRAINING_MANIFEST_HASHES.json'), 'router_initialization': sha(BASE / 'ROUTER_INITIALIZATION.json')}, 'hypothesis_evaluator_frozen': True, 'no_audit_scientific_response_previously_computed': True, 'primary_horizon_nonpadding_tokens': 64000, 'bootstrap_replicates': 20000, 'bootstrap_seed': 2026100199, 'frozen_H1_zero_floor': 1e-20}
record_path.write_text(json.dumps(record, indent=2) + '\n')
print('AUDIT_OPENED', record['timestamp_asia_shanghai'], flush=True)
