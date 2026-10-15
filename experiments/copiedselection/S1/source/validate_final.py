# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
WORK = ROOT / 'work/scientific'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(name):
    return json.loads((OUT / name).read_text())
required = ('TRAINING_INTEGRITY.json', 'CHECKPOINT_HASHES.json', 'CALIBRATION_FREEZE.json', 'AUDIT_OPENING_RECORD.json', 'AUDIT_PRIMARY_64K.json', 'PRIMARY_REGRESSIONS.json', 'BOOTSTRAP.json', 'FORMAL_HYPOTHESES.json', 'FORMAL_LABEL.json', 'SECONDARY_TEMPORAL.json', 'EXP_S1_REPORT.md', 'EXP_S1_RESULTS.json', 'PROVENANCE.json')
assert all(((OUT / n).is_file() for n in required))
integ = read('TRAINING_INTEGRITY.json')
checks = read('CHECKPOINT_HASHES.json')
opening = read('AUDIT_OPENING_RECORD.json')
cal = read('CALIBRATION_FREEZE.json')
primary = read('AUDIT_PRIMARY_64K.json')
reg = read('PRIMARY_REGRESSIONS.json')
boot = read('BOOTSTRAP.json')
formal = read('FORMAL_LABEL.json')
freeze = read('PRIMARY_FREEZE.json')
secondary = read('SECONDARY_TEMPORAL.json')
results = read('EXP_S1_RESULTS.json')
prov = read('PROVENANCE.json')
assert integ['status'] == 'PASS' and integ['run_count'] == 32 and (integ['total_nonpadding_tokens'] == 8192000)
assert checks['status'] == 'FROZEN' and checks['checkpoint_count'] == 128
assert len(list((OUT / 'checkpoints').glob('*/RUN_RESULT.json'))) == 32
assert len(list((OUT / 'checkpoints').glob('*/router_*.safetensors'))) == 128
assert len(list((OUT / 'calibration_responses').glob('*.json'))) == 128
assert len(list((OUT / 'audit_responses').glob('*.json'))) == 104
assert len(primary['runs']) == 24 and len(primary['S1_H1_rows']) == 32
assert primary['resolved_count_by_arm'] == {'S2': 8, 'S4': 8, 'S8': 8}
assert all((r['resolved_positive'] for r in primary['runs']))
assert all((r['all_12_below_floor'] for r in primary['S1_H1_rows']))
assert opening['timestamp_asia_shanghai'] > cal['timestamp_asia_shanghai']
for name, expected in opening['analysis_code_sha256'].items():
    assert sha(WORK / name) == expected
for name, expected in freeze['files_sha256'].items():
    assert sha(OUT / name) == expected
assert sha(WORK / 'evaluate_s1.py') == freeze['evaluator_code_sha256']
assert boot['replicates'] == 20000 and boot['seed'] == 2026100199
assert sha(OUT / 'BOOTSTRAP_INDICES.npy') == boot['indices_sha256']
assert formal['primary_horizon_nonpadding_tokens'] == 64000 and formal['final']
assert formal['label'] == results['formal_label'] == secondary['primary_label_unchanged'] == freeze['central_label']
assert all((formal['hypotheses'][h] == 'PASS' for h in ('H1', 'H2', 'H3', 'H4')))
assert formal['resolved_requirement'] == 'PASS'
assert formal['ci95']['beta_Theta'][1] < 0
assert formal['ci95']['beta_Q'][0] > 0
assert formal['ci95']['beta_B'][0] > 0
slopes = formal['central_slopes']
assert abs(slopes['beta_B'] - slopes['beta_Q'] - reg['descriptive_beta_K'] - slopes['beta_Theta']) < 1e-10
assert prov['formal_label_sha256'] == sha(OUT / 'FORMAL_LABEL.json')
assert prov['primary_freeze_sha256'] == sha(OUT / 'PRIMARY_FREEZE.json')
assert results['provenance_sha256'] == sha(OUT / 'PROVENANCE.json')
assert results['formal_label_sha256'] == sha(OUT / 'FORMAL_LABEL.json')
result = {'status': 'PASS', 'formal_label': formal['label'], 'runs_verified': 32, 'checkpoints_verified': 128, 'calibration_response_files_verified': 128, 'audit_response_files_verified': 104, 'primary_audit_seed_arm_rows': 24, 'S1_H1_seed_horizon_rows': 32, 'bootstrap_replicates': 20000, 'formal_label_sha256': sha(OUT / 'FORMAL_LABEL.json'), 'primary_freeze_sha256': sha(OUT / 'PRIMARY_FREEZE.json')}
(OUT / 'FINAL_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
