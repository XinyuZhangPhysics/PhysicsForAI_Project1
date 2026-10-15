# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys, json, math, hashlib
from pathlib import Path
R = Path(__file__).resolve().parents[1]
B = R.parents[1]
A = B / '2026-10-12/g-j/outputs/v285M_archive_completion'
P = B / '2026-10-09/g-ju/outputs/P2P_NV1_FRESH_P2P_COMPLETE'
H = B / '2026-10-08/jie/outputs'
O = R / 'outputs/v286M_finite_time_mechanism'
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, torch, pandas as pd
from scipy.linalg import logm, expm
torch.set_num_threads(4)
DT = torch.float64
D = 576
T = 32
import ast
p = R / 'work/finite_time_v286.py'
nodes = [n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ['c', 'ct', 'gram', 'coefficients']]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'))
arms = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
cp = {(r['seed'], r['arm'], r['update']): r for r in json.loads((H / 'P2H_CHECKPOINT_INDEX.json').read_text())['checkpoints']}
seeds = torch.stack(torch.load(H.parent / 'work/p2h_private_seeds.pt', weights_only=True)).double()
checks = []
stability = []
for j in range(9):
    if j < 8:
        caps = []
        for n in range(1, 33):
            with np.load(A / f'FINITE_TIME_RECOVERED_MAP_FACTORS/stream{j}_step{n:02}.npz') as z:
                caps.append(tuple((torch.from_numpy(z[k].copy()).double() for k in ['h', 'phi', 'g'])))
    else:
        vs = [v for v in json.loads((P / 'P2P_BATCH_VIEW_SEAL.json').read_text())['views'] if v['group'] == 'primary']
        caps = []
        for v in vs:
            z = torch.load(P / f"captures/fresh_P2P/capture_{v['name']}.pt", weights_only=True)
            caps.append(tuple((z[k].double() for k in ['h', 'phi', 'g'])))
    for n, (h, p, g) in enumerate(caps):
        gram0 = h.T @ (g @ g.T * (p @ p.T)) @ h / 16
        ev = torch.linalg.eigvalsh(gram0).clamp_min(0)
        sv = ev.sqrt()
        for k, ar in enumerate(arms):
            em, eu = (ar['eta_M'], ar['eta_U'])
            am = 1 - em * ar['mu']
            au = 1 - eu * ar['lambda_']
            M = torch.zeros(D, 2, 2, dtype=DT)
            M[:, 0, 0] = am - 1
            M[:, 1, 1] = au - 1
            M[:, 0, 1] = em * sv
            M[:, 1, 0] = eu * sv
            aNorm = max(float(torch.linalg.svdvals(M).max()), abs(am - 1), abs(au - 1))
            lmin = float(((am + au) / 2 - torch.sqrt(((am - au) / 2) ** 2 + em * eu * ev)).min())
            assert aNorm < 1 and lmin > 0
            LL = np.array([[am, em * float(sv[-1])], [eu * float(sv[-1]), au]])
            kk = logm(LL)
            rr = np.linalg.norm(expm(kk) - LL)
            assert np.isrealobj(kk) and rr < 1e-12
            stability.append(dict(bank='sequence' if j < 8 else 'reference', stream=j if j < 8 else vs[n]['stream'], map=n + 1, arm=k, norm_L_minus_I=aNorm, min_real_eigenvalue=lmin, minimum_singular_value_lower_bound=1 - aNorm, condition_number_upper_bound=(1 + aNorm) / (1 - aNorm), log_reconstruction_2x2_absolute_residual=rr, log_series_12_term_operator_remainder_bound=aNorm ** 13 / (13 * (1 - aNorm))))
    print('map-log stability bank', j, 'PASS', flush=True)
    if j not in [0, 6]:
        continue
    GS = gram(caps)
    seed = seeds[j]
    for m in [1, 2, 4, 8, 16, 32]:
        src = R / f'work/subsets_stream{j}_m{m}.npz'
        if not src.exists():
            raise RuntimeError(f'Sampling output not ready: {src}')
        z = np.load(src)
        indices = z['indices']
        expected = z['errors']
        for draw in sorted(set([0, len(indices) - 1])):
            ids = indices[draw]
            bank = [caps[i] for i in ids]
            gm = GS[ids][:, :, ids, :].mean((0, 2))
            cm = sum((c(cap, seed) for cap in bank)) / m
            for k in [0, 7, 14]:
                ar = arms[k]
                am, au, p, q = coefficients(ar)
                v = cm.clone()
                r = torch.zeros(D, dtype=DT)
                u = r.clone()
                for n in range(16):
                    r += p[n] * v
                    u += q[n] * v
                    v = gm @ v
                predM = am ** 32 * seed + sum((ct(cap, r) for cap in bank)) / m
                with np.load(cp[j, k, 32]['path']) as zz:
                    targetM = torch.from_numpy(zz['M'].copy()).double()
                    targetU = torch.from_numpy(zz['U'].copy()).double()
                dn = float(torch.linalg.vector_norm(targetM - seed))
                un = float(torch.linalg.vector_norm(targetU))
                emerr = float(torch.linalg.vector_norm(predM - targetM))
                euerr = float(torch.linalg.vector_norm(u - targetU))
                actual = np.array([math.hypot(emerr, euerr) / math.hypot(dn, un), emerr / dn, euerr / un])
                reldiff = np.max(abs(actual - expected[draw, k]) / np.maximum(actual, 1e-300))
                assert np.allclose(actual, expected[draw, k], rtol=5e-06, atol=1e-11), (j, m, draw, k, actual, expected[draw, k])
                checks.append(dict(stream=j, m=m, draw=draw, arm=k, max_relative_anchored_error=reldiff))
                if m == 4 and draw == 0 and (k == 7):
                    M = seed.clone()
                    U = torch.zeros(D, dtype=DT)
                    for n in range(32):
                        M, U = (am * M + ar['eta_M'] * sum((ct(cap, U) for cap in bank)) / m, au * U + ar['eta_U'] * sum((c(cap, M) for cap in bank)) / m)
                    residual = math.hypot(float(torch.linalg.vector_norm(M - predM)), float(torch.linalg.vector_norm(U - u))) / float(torch.linalg.vector_norm(predM))
                    assert residual < 1e-12
                    checks.append(dict(stream=j, m=m, arm=k, literal_32_mean_steps_relative_error=residual))
pd.DataFrame(stability).to_csv(O / 'FINITE_TIME_MAP_LOG_STABILITY.csv', index=False, float_format='%.17g')
pd.DataFrame(checks).to_csv(O / 'SAMPLING_NUMERICAL_VERIFICATION.csv', index=False, float_format='%.17g')
(O / 'MAP_LOG_AND_SAMPLING_VERIFICATION.json').write_text(json.dumps(dict(status='PASS', map_arm_checks=len(stability), unique_maps=288, sampling_cases=sum(('draw' in q for q in checks)), literal_four_map_replays=2, max_log_reconstruction_residual=max((q['log_reconstruction_2x2_absolute_residual'] for q in stability)), maximum_map_norm_L_minus_I=max((q['norm_L_minus_I'] for q in stability)), minimum_map_eigenvalue=min((q['min_real_eigenvalue'] for q in stability)), maximum_log_series_remainder=max((q['log_series_12_term_operator_remainder_bound'] for q in stability))), indent=2))
print('ALL SECONDARY VERIFICATIONS PASS', flush=True)
