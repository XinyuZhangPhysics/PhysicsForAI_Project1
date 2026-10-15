# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import ast, csv, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
BASE = Path(_p1_path('path_65e87eb57756'))
ROOT = BASE / '2026-10-12/g-j'
OUT = ROOT / 'outputs/v285M_archive_completion'
H = BASE / '2026-10-08/jie/outputs'
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)
source = BASE / '2026-10-09/g-ju/outputs/P2P_NV1_FRESH_P2P_COMPLETE/historical/NV0_source/source/preflight.py'
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'CrossOperator']
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))
preds = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
seeds = torch.stack(torch.load(H.parent / 'work/p2h_private_seeds.pt', weights_only=True)).double()
cp = {(r['seed'], r['arm'], r['update']): r for r in json.loads((H / 'P2H_CHECKPOINT_INDEX.json').read_text())['checkpoints']}
rows = {(int(r['stream']), int(r['arm']), r['sector']): r for r in csv.DictReader((OUT / 'FINITE_TIME_THREE_WAY_CASES.csv').open())}
checks = []
start = time.monotonic()
for j in [0, 6]:
    caps = []
    for n in range(1, 33):
        with np.load(OUT / f'FINITE_TIME_RECOVERED_MAP_FACTORS/stream{j}_step{n:02}.npz') as z:
            caps.append(CrossOperator(*[torch.from_numpy(z[k].copy()).double() for k in ['h', 'phi', 'g']]))
    for arm in [0, 7, 14]:
        p = preds[arm]
        em = p['eta_M']
        eu = p['eta_U']
        am = 1 - em * p['mu']
        au = 1 - eu * p['lambda_']
        m = seeds[j].clone()
        u = torch.zeros(576, dtype=torch.float64)
        for n in range(32):
            cm = sum((op.c(m) for op in caps)) / 32
            ct = sum((op.ct(u) for op in caps)) / 32
            m, u = (am * m + em * ct, au * u + eu * cm)
        with np.load(cp[j, arm, 32]['path']) as z:
            om = torch.from_numpy(z['M'].copy()).double()
            ou = torch.from_numpy(z['U'].copy()).double()
        emerr = float(torch.linalg.vector_norm(m - om))
        euerr = float(torch.linalg.vector_norm(u - ou))
        dm = om - seeds[j]
        for sector, error, den in [('full_state', np.sqrt(emerr ** 2 + euerr ** 2), float(torch.sqrt(dm.square().sum() + ou.square().sum()))), ('expert_private', emerr, float(torch.linalg.vector_norm(dm))), ('router_private', euerr, float(torch.linalg.vector_norm(ou)))]:
            val = error / max(den, 1e-30)
            expected = float(rows[j, arm, sector]['seqmean_original_error'])
            assert np.isclose(val, expected, rtol=1e-09, atol=1e-12), (j, arm, sector, val, expected)
            checks.append(dict(stream=j, arm=arm, sector=sector, literal_mean_map_32_steps_error=val, accelerated_error=expected, absolute_difference=abs(val - expected), pass_check=True))
        print(f'Literal mean-map verification stream {j}, arm {arm}: PASS; {time.monotonic() - start:.1f}s', flush=True)
with (OUT / 'FINITE_TIME_LITERAL_MEAN_VERIFICATION.json').open('w') as f:
    json.dump(dict(status='PASS', method='32 literal mean-map applications in full coordinates; canonical CrossOperator actions; six cases across primary and replication at arm indices 0,7,14', checks=checks, elapsed_seconds=time.monotonic() - start), f, indent=2)
