# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import ast, json, hashlib, csv, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import pyarrow.parquet as pq
R = Path(__file__).resolve().parents[2]
O = R / 'outputs/P2Q_RANDOM_COCYCLE_COMPLETE'
H = Path(_p1_path('path_08e09cd8aadb'))
P = Path(_p1_path('path_5e0fd824e039'))
torch.set_num_threads(8)
torch.use_deterministic_algorithms(True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(n):
    return json.loads((O / n).read_text())

def close(a, b, tol=2e-12):
    assert np.allclose(a, b, rtol=tol, atol=1e-14), (a, b)
c = load('P2Q_COMPLETION.json')
assert c['status'] == 'COMPLETE'
dec = load('P2Q_PRIMARY_DECISION.json')
dh = sha(O / 'P2Q_PRIMARY_DECISION.json')
assert dh == (O / 'P2Q_PRIMARY_DECISION.sha256').read_text().split()[0] == c['primary_decision_sha256'] == load('P2Q_REPLICATION.json')['primary_decision_sha256']
assert sha(O / 'source/run_p2q.py') == dec['source_sha256'] == load('P2Q_IMPLEMENTATION_CONVENTIONS.json')['source_sha256']
ev = load('P2Q_EVENT_LEDGER.json')
seal = next((r['time_ns'] for r in ev if r['action'] == 'PRIMARY_DECISION_HASH_SEALED'))
assert all((r['time_ns'] > seal for r in ev if r['action'] == 'TARGET_CHECKPOINT_OPEN' and r['stream'] >= 6 or (r['action'] == 'TARGET_SCALAR_ROWS_OPEN' and min(r['streams']) >= 6)))
access = load('P2Q_TARGET_ACCESS_AUDIT.json')
assert len(access) == 840 and all((r['opened_time_ns'] > seal for r in access if r['seed'] >= 6))
comp = list(csv.DictReader((O / 'P2Q_CHECKPOINT_COMPARISON.csv').open()))
gains = list(csv.DictReader((O / 'P2Q_FINITE_TIME_GAINS.csv').open()))
tab = pq.read_table(O / 'P2Q_COCYCLE_TRAJECTORIES.parquet')
tr = tab.to_pylist()
assert len(comp) == 1680 and len(gains) == 120 and (len(tr) == 7920)
assert len({(r['seed'], r['arm'], r['update'], r['model']) for r in tr}) == 7920
assert {int(r['update']) for r in comp} == {0, 1, 2, 4, 8, 16, 32}
assert len(load('P2Q_BATCH_SEQUENCE_SEAL.json')['views']) == 256
act = load('P2Q_OPERATOR_ACTION_AUDIT.json')['batches']
assert len(act) == 256 and max((max(r['C_relative_error'], r['CT_relative_error']) for r in act)) <= 1e-12
anchors = load('P2Q_P2P_ANCHOR_REPLAY.json')['anchors']
assert len(anchors) == 4 and all((all(r['capture_bitwise_equal'].values()) and r['s1_relative_error'] <= 1e-10 and (r['top4_left_projector_frobenius'] <= 1e-09) for r in anchors))
e = np.zeros((8, 15, 2))
cc = e.copy()
gam = np.zeros((8, 15, 3))
models = ['RTO_EXACT_BATCH_COCYCLE', 'F0_TOP4_FIXED_BASELINE']
for j in range(8):
    for k in range(15):
        for q, model in enumerate(models):
            rows = [r for r in comp if int(r['seed']) == j and int(r['arm']) == k and (r['model'] == model) and (int(r['update']) in [1, 2, 4, 8])]
            assert len(rows) == 4
            e[j, k, q] = np.median([float(r['displacement_error']) for r in rows])
            cc[j, k, q] = np.median([float(r['displacement_cosine']) for r in rows])
        gr = next((r for r in gains if int(r['seed']) == j and int(r['arm']) == k))
        gam[j, k] = [float(gr[n]) for n in ['Gamma_observed', 'Gamma_RTO', 'Gamma_F0top4']]
rat = e[:, :, 0] / np.maximum(e[:, :, 1], 1e-30)
npz = np.load(O / 'P2Q_BOOTSTRAP_RESAMPLES.npz', allow_pickle=False)
draws = np.random.default_rng(2026100901).integers(0, 6, (20000, 6))
assert np.array_equal(draws, npz['seed_identity_draws'])
computed = {}
for n, v in [('E_loc_RTO', e[:6, :, 0]), ('C_loc_RTO', cc[:6, :, 0]), ('R_loc', rat[:6])]:
    computed[n] = np.median(np.median(v, axis=1)[draws], axis=1)
a = np.median(gam[draws], axis=1)
obs = a[:, :, 0]
rto = a[:, :, 1]
f0 = a[:, :, 2]
den = np.maximum(np.linalg.norm(obs, axis=1), 1e-30)
er = np.linalg.norm(rto - obs, axis=1) / den
ef = np.linalg.norm(f0 - obs, axis=1) / den
center = lambda v: v - v.mean(axis=1, keepdims=True)
o = center(obs)
rr = center(rto)
ff = center(f0)
computed.update(E_Gamma_RTO=er, E_Gamma_F0top4=ef, R_Gamma=er / np.maximum(ef, 1e-30), rho_Gamma_RTO=(o * rr).sum(1) / (np.linalg.norm(o, axis=1) * np.linalg.norm(rr, axis=1)), rho_Gamma_F0top4=(o * ff).sum(1) / (np.linalg.norm(o, axis=1) * np.linalg.norm(ff, axis=1)))
for n, v in computed.items():
    close(v, npz[n])
    s = load('P2Q_BOOTSTRAP_SUMMARY.json')['statistics'][n]
    close(np.percentile(v, [2.5, 97.5]), [s['lower95'], s['upper95']])
s = dec['statistics']
local = [s['E_loc_RTO']['upper95'] <= 0.3, s['C_loc_RTO']['lower95'] >= 0.85, s['R_loc']['upper95'] <= 0.75, int((np.median(rat[:6], 1) < 1).sum()) >= 5]
seed_er = np.linalg.norm(gam[:6, :, 1] - gam[:6, :, 0], axis=1)
seed_ef = np.linalg.norm(gam[:6, :, 2] - gam[:6, :, 0], axis=1)
window = [s['E_Gamma_RTO']['upper95'] <= 0.5, s['rho_Gamma_RTO']['lower95'] >= 0.75, s['R_Gamma']['upper95'] <= 0.75, int((seed_er < seed_ef).sum()) >= 5]
assert list(dec['local_components'].values()) == local and list(dec['window_components'].values()) == window
expected = 'P2Q_P2H_WINDOW_COCYCLE_BRIDGE_SUPPORTED' if all(local) and all(window) else 'P2Q_LOCAL_COCYCLE_SUPPORTED_WINDOW_NOT_CLOSED' if all(local) else 'P2Q_COCYCLE_IMPROVEMENT_INSUFFICIENT_FOR_CLOSURE' if not (local[0] and local[1]) and local[2] and local[3] else 'P2Q_STOCHASTIC_TANGENT_BRIDGE_NOT_SUPPORTED_OR_UNRESOLVED'
assert expected == dec['label']
tree = ast.parse((P / 'historical/NV0_source/source/preflight.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CrossOperator']
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'inherited', 'exec'))
tree = ast.parse((O / 'source/run_p2q.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'batch_actions']
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actions', 'exec'))
seeds = torch.load(H.parent / 'work/p2h_private_seeds.pt', weights_only=True)
f = np.load(H / 'P2H_CROSS_HESSIAN_SPECTRUM.npz')
m = torch.stack([seeds[k % 8] for k in range(15)]).double()
u = torch.from_numpy(np.stack([f['u'][:, k % 4] for k in range(15)])).double()
errs = []
for ar in anchors:
    cap = torch.load(O / f"replay_subset/capture_{ar['batch']}.pt", weights_only=True)
    h, p, g = [cap[k].double() for k in ['h', 'phi', 'g']]
    cm, ct = batch_actions(h, p, g, m, u)
    op = CrossOperator(h, p, g)
    for k in range(15):
        for v, w in [(cm[k], op.c(m[k])), (ct[k], op.ct(u[k]))]:
            er = float(torch.linalg.vector_norm(v - w) / torch.linalg.vector_norm(w).clamp_min(1e-30))
            assert er <= 1e-12
            errs.append(er)
preds = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
fu = f['u'].astype('float64')
fv = f['v'].astype('float64').reshape(-1, 4)
fs = f['s'].astype('float64')
GU = fu.T @ fu
GV = fv.T @ fv
max_base = 0.0
lookup = {(r['seed'], r['arm'], r['update'], r['model']): r for r in tr}
for j in range(8):
    sm = seeds[j].double().numpy().reshape(-1)
    z = sm @ fv
    sm2 = sm @ sm
    for k, pr in enumerate(preds):
        d = 1.0
        b = np.zeros(4)
        a = np.zeros(4)
        for n in range(33):
            if n:
                b, a = ((1 - pr['eta_M'] * pr['mu']) * b + pr['eta_M'] * fs * (GU @ a), (1 - pr['eta_U'] * pr['lambda_']) * a + pr['eta_U'] * fs * (d * z + GV @ b))
                d *= 1 - pr['eta_M'] * pr['mu']
            amp = np.sqrt(d * d * sm2 + 2 * d * (b @ z) + b @ GV @ b + a @ GU @ a)
            record = lookup[j, k, n, models[1]]
            rel = abs(amp - record['amplitude']) / amp
            max_base = max(max_base, rel)
            assert rel <= 1e-12
cap = torch.load(O / 'replay_subset/capture_B0_1.pt', weights_only=True)
op = CrossOperator(*[cap[k].double() for k in ['h', 'phi', 'g']])
seed = seeds[0].double()
cm = op.c(seed)
first_errors = []
for k, pr in enumerate(preds):
    pm = (1 - pr['eta_M'] * pr['mu']) * seed
    pu = pr['eta_U'] * cm
    am = float(torch.sqrt(pm.square().sum() + pu.square().sum()))
    close(am, lookup[0, k, 1, models[0]]['amplitude'])
    obs = np.load(H / f'checkpoints/seed0_arm{k:02d}_n01.npz')
    om = torch.from_numpy(obs['M']).double()
    ou = torch.from_numpy(obs['U']).double()
    dm = om - seed
    pn = pm - seed
    er = float(torch.sqrt((pn - dm).square().sum() + (pu - ou).square().sum()) / torch.sqrt(dm.square().sum() + ou.square().sum()))
    cos = float(((pn * dm).sum() + (pu * ou).sum()) / torch.sqrt((pn.square().sum() + pu.square().sum()) * (dm.square().sum() + ou.square().sum())))
    saved = next((r for r in comp if int(r['seed']) == 0 and int(r['arm']) == k and (int(r['update']) == 1) and (r['model'] == models[0])))
    close([er, cos], [float(saved['displacement_error']), float(saved['displacement_cosine'])])
    first_errors.append(er)
result = dict(status='PASS', primary_seal_valid=True, replication_target_access_after_seal=True, source_freeze_unchanged=True, row_counts=dict(trajectory=7920, checkpoint_comparison=1680, gains=120), all_bootstrap_resamples_recomputed=True, decision_conjunctions_recomputed=True, all_arm_vectorized_action_checks=120, maximum_action_relative_error=max(errs), all_F0_amplitudes_independently_reproduced=3960, maximum_F0_amplitude_relative_error=max_base, RTO_first_update_arms_independently_reproduced=15)
(O / 'P2Q_FINAL_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
