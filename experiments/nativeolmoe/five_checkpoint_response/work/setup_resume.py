# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, subprocess, shutil
RUN = Path(__file__).resolve().parents[1]
PRIOR = RUN.parent / 'P2D_RECOVERY_20261008_085240'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
checks = []
for line in (PRIOR / 'SHA256SUMS_P2D.txt').read_text().splitlines():
    h, n = line.split('  ', 1)
    assert sha(PRIOR / n) == h, n
    checks.append({'path': str(PRIOR / n), 'sha256': h})
assert len(checks) == 289
names = [p.name for p in PRIOR.glob('T0_*') if p.is_file()] + ['Formal_Bank_T_v1.jsonl', 'control_package']
for n in names:
    (RUN / n).symlink_to(PRIOR / n, target_is_directory=(PRIOR / n).is_dir())
for n in ['CODEX_HANDOFF_P2D_T1_NUMERICAL_CERTIFICATION.md', 'P2D_T1_NUMERICAL_IDENTITY_DISPOSITION.md']:
    shutil.copyfile(Path(_p1_path('path_27e4e1a80a62')) / n, RUN / 'work' / n)
pkg = RUN / 'control_package/Project1_P2D_Codex_Control_Package_v1p1'
p = subprocess.run(['python3', 'scripts/verify_package.py'], cwd=pkg, text=True, capture_output=True)
with (RUN / 'logs/package_verification.log').open('x') as f:
    f.write(p.stdout + p.stderr)
assert p.returncode == 0
with (RUN / 'T1_RESUME_PREREQUISITE_AUDIT.json').open('x') as f:
    json.dump({'status': 'PASS', 'prior_run_path': str(PRIOR), 'prior_289_hashes_verified': checks, 'package_verification': 'PASS_23_FILES', 'inherited_T0_seal_sha256': sha(PRIOR / 'T0_BANK_T_SEAL.json'), 'T0_inputs_reused_by_symlink_no_copies': names, 'authorities': [{'path': str(RUN / 'work' / n), 'source_path': str(Path(_p1_path('path_27e4e1a80a62')) / n), 'sha256': sha(RUN / 'work' / n)} for n in ['CODEX_HANDOFF_P2D_T1_NUMERICAL_CERTIFICATION.md', 'P2D_T1_NUMERICAL_IDENTITY_DISPOSITION.md']], 'MODEL_LOAD_COUNT': 0, 'RESERVE_ACCESS_COUNT': 0}, f, indent=2)
    f.write('\n')
print('RESUME PRECHECK PASS; 289 old files unchanged; package 23 PASS; T0 reused in place')
