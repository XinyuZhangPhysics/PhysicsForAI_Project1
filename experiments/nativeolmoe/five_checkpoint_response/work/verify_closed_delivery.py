# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Read-only final completeness/immutability verification after T4 closure."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json
RUN = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
assert (RUN / 'T4_STAGE_SEAL.sha256').is_file()
histories = []
for name in ('P2D_20261008_081445', 'P2D_RECOVERY_20261008_085240', 'P2D_T1_CERTIFICATION_20261008_102758'):
    old = RUN.parent / name
    manifest = old / 'SHA256SUMS_P2D.txt'
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert sha(old / relative) == expected, (name, relative)
        checked += 1
    histories.append(dict(path=str(old), manifest_sha256=sha(manifest), files_checked=checked, status='PASS'))
required = 'T0_BANK_T_SEAL.json T0_BASIS_AUDIT.json T0_OPENED_OVERLAP_LEDGER.json T0_OPENED_OVERLAP_LEDGER.sha256 T0_BANK_T_SELECTION_TRACE.jsonl T1_41B_TOMOGRAPHY.npz T1_5033_MODE1_SCALARS.json T1_SCALAR_PREDICTIONS.json T2_5033_TOMOGRAPHY.npz T2_PREDICTION_SCORE.json T3_INTERMEDIATE_TOMOGRAPHY.npz T4_TOMOGRAPHY_SUMMARY.csv T4_SPLIT_HALF_STABILITY.csv T4_BOOTSTRAP_SUMMARY.csv T4_CROSS_PROBE_GEOMETRY.csv T4_CHANNEL_DECOMPOSITION.csv T4_AU_CACHE_CAPABILITIES.json P2D_TOMOGRAPHY_EXECUTION_REPORT.md'.split()
assert all(((RUN / name).is_file() for name in required))
report = (RUN / 'P2D_TOMOGRAPHY_EXECUTION_REPORT.md').read_text().splitlines()
expected_prefix = [f'{name} = 0' for name in ('NEW_TRAINING_STEP_COUNT', 'RESERVE_ACCESS_COUNT', 'NEW_NULL_BANK_COUNT', 'BASIS_REFIT_COUNT', 'OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT')]
expected_prefix += [f'T{i}_STATUS = PASS' for i in range(5)]
assert report[:10] == expected_prefix
output = dict(status='PASS', historical_runs=histories, required_outputs=[dict(path=str(RUN / n), sha256=sha(RUN / n)) for n in required], report_prefix_verified=True)
with (RUN / 'P2D_HISTORY_IMMUTABILITY_AND_DELIVERY_AUDIT.json').open('x') as f:
    json.dump(output, f, indent=2)
    f.write('\n')
print('HISTORICAL IMMUTABILITY AND REQUIRED DELIVERY PASS', [r['files_checked'] for r in histories])
