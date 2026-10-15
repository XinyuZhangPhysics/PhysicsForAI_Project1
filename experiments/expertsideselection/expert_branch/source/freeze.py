# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
assert not (OUT / 'CALIBRATION_IMPLEMENTATION_CHECK_v284M.json').exists()
p = OUT / 'EXPERT_SIDE_SELECTION_PREREGISTRATION_v284M.md'
h = sha(p)
(OUT / 'EXPERT_SIDE_SELECTION_PREREGISTRATION_v284M.sha256').write_text(f'{h}  {p.name}\n')
assets = {str(p): h, str(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json'): sha(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json'), str(OUT / 'DIRECTION_BINDINGS_v284M.json'): sha(OUT / 'DIRECTION_BINDINGS_v284M.json')}
for p in (BASE / 'work/source').glob('*.py'):
    assets[str(p)] = sha(p)
for p in (OUT / 'banks').iterdir():
    assets[str(p)] = sha(p)
write('PROTOCOL_FREEZE_v284M.json', {'timestamp': stamp(), 'phase': 'BEFORE_ANY_NEW_DERIVATIVE', 'preregistration_sha256': h, 'assets': assets, 'closure_atol': 1e-10, 'closure_rtol': 1e-08, 'fd_atol': 1e-09, 'fd_rtol': 1e-06, 'structural_floor': 1e-10, 'amplitudes': [0.02, 0.01, 0.005], 'bootstrap_seed': 202610120284, 'bootstrap_draws': 20000})
print('FROZEN', h, flush=True)
