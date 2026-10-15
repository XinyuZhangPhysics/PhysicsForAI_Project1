# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
import ast, sys, json, hashlib, time, platform, socket, csv, traceback, shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/P2Q_RANDOM_COCYCLE_COMPLETE'
P = Path(_p1_path('path_5e0fd824e039'))
H = Path(_p1_path('path_08e09cd8aadb'))
A = ROOT / 'work/PhysicsForAI_Project1_Migration_v209M'
SEED = H.parent / 'work/p2h_private_seeds.pt'
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def deny(*a, **k):
    raise RuntimeError('NETWORK_ACCESS_PROHIBITED')
socket.socket.connect = deny
socket.create_connection = deny
START = time.monotonic()
EVENTS = []
COMP = []
TRAJ = []
GAINS = []
CP_AUDIT = []
BATCH_AUDIT = []
MODELS = ['RTO_EXACT_BATCH_COCYCLE', 'F0_TOP4_FIXED_BASELINE']
TIMES = [0, 1, 2, 4, 8, 16, 32]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def tsha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def log(s):
    print(s, flush=True)

def event(action, **kw):
    EVENTS.append(dict(action=action, time_ns=time.time_ns(), **kw))
    write('P2Q_EVENT_LEDGER.json', EVENTS)
    log(action + ' ' + str(kw))

def gate(b, label='P2Q_PROVENANCE_FAIL', detail=''):
    if not b:
        raise RuntimeError(label + ': ' + detail)

def csvwrite(n, rows):
    with (OUT / n).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def norm(t):
    return float(torch.linalg.vector_norm(t))

def inherited():
    src = P / 'historical/NV0_source/source/preflight.py'
    tree = ast.parse(src.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'capture', 'CrossOperator'}]
    gate(len(nodes) == 2, detail='inherited definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec'), globals())

def audit():
    freeze = json.loads((A / '12_P2Q_RANDOM_COCYCLE/P2Q_PROTOCOL_FREEZE.json').read_text())
    manifest = A / '11_P2P_SCIENTIFIC_CLOSURE/SHA256SUMS_P2P.txt'
    gate(sha(manifest) == sha(P / 'SHA256SUMS_P2P.txt'), detail='P2P manifest identity')
    checked = []
    for line in manifest.read_text().splitlines():
        h, n = line.split(maxsplit=1)
        n = n.lstrip('*')
        p = P / n
        gate(p.is_file() and sha(p) == h, detail='P2P ' + n)
        checked.append(dict(path=str(p), sha256=h, size=p.stat().st_size))
    for n, h in freeze['p2h_target_hashes'].items():
        gate((H / n).is_file() and sha(H / n) == h, 'P2Q_HISTORICAL_TARGET_MISSING', n)
    idx = json.loads((H / 'P2H_CHECKPOINT_INDEX.json').read_text())
    checkpoints = {(r['seed'], r['arm'], r['update']): r for r in idx['checkpoints'] if r['update'] in TIMES}
    gate(len(checkpoints) == 840, 'P2Q_HISTORICAL_TARGET_MISSING', 'checkpoint count')
    for key, r in checkpoints.items():
        gate(Path(r['path']).is_file() and sha(r['path']) == r['sha256'], 'P2Q_HISTORICAL_TARGET_MISSING', str(key))
    pre = json.loads((H / 'P2H_PREFLIGHT_SEAL.json').read_text())
    gate(sha(H / 'P2H_PREFLIGHT_SEAL.json') == json.loads((A / '04_Q2_FORMATION/P2H_STOCHASTIC/P2H_PRIMARY_DECISION.json').read_text())['preflight_seal_sha256'], detail='preflight')
    for n in ['P2H_PREDICTIONS.json', 'P2H_DATA_BANK_SEAL.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz']:
        gate(sha(H / n) == pre['artifact_sha256'][n], detail=n)
    gate(sha(SEED) == pre['seed_file_sha256'], detail='private seeds')
    sourcezip = A / '04_Q2_FORMATION/P2H_STOCHASTIC/P2H_REPRODUCIBILITY_SOURCE.zip'
    hm = {n.lstrip('*'): h for h, n in (l.split(maxsplit=1) for l in (A / '04_Q2_FORMATION/P2H_STOCHASTIC/SHA256SUMS_P2H.txt').read_text().splitlines())}
    gate(sha(sourcezip) == hm[sourcezip.name], detail='historical source archive')
    bs = json.loads((H / 'P2H_DATA_BANK_SEAL.json').read_text())
    banks = {}
    for n, r in bs['banks'].items():
        p = P / 'inputs/banks' / f'{n}.npy'
        gate(sha(p) == r['file_sha256'], detail=n)
        ar = np.load(p, allow_pickle=False)
        gate(hashlib.sha256(ar.tobytes()).hexdigest() == r['token_array_sha256'], detail=n + ' values')
        banks[n] = torch.from_numpy(ar)
    env = json.loads((P / 'P2P_ASSET_AUDIT.json').read_text())['environment']
    versions = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, transformers=__import__('transformers').__version__)
    gate(versions == env['versions'] and sha(sys.executable) == env['python_executable_sha256'], detail='environment')
    preds = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
    gate(len(preds) == 15 and [(r['epsilon'], r['r_eta']) for r in preds] == [(e, r) for e in freeze['arms']['epsilon'] for r in freeze['arms']['r_eta']], detail='arm order')
    f0 = np.load(H / 'P2H_CROSS_HESSIAN_SPECTRUM.npz', allow_pickle=False)
    gate(f0['u'].shape == (576, 4) and f0['v'].shape == (576, 1536, 4), detail='F0 dimensions')
    seeds = torch.load(SEED, weights_only=True)
    gate(len(seeds) == 8 and all((t.shape == (576, 1536) for t in seeds)), detail='seeds dimensions')
    write('P2Q_ASSET_AUDIT.json', dict(status='PASS', P2P_package_in_place=str(P), P2P_verified_files=checked, historical_target_hashes=freeze['p2h_target_hashes'], checkpoints_byte_audited=840, checkpoint_content_opened_before_primary_seal_streams=[0, 1, 2, 3, 4, 5], historical_seed_sha256=sha(SEED), preflight_sha256=sha(H / 'P2H_PREFLIGHT_SEAL.json'), historical_source_sha256=sha(sourcezip), versions=versions, python=sys.executable, CPU_threads=8, capture_precision='float32', operator_precision='float64', optimizer=False, training_updates=0, network_guard='socket denied + offline local-only model loading', new_banks=0))
    views = [dict(name=f'B{j}_{n}', stream=j, update=n, start=4 * (n - 1), stop=4 * n, ids_sha256=tsha(banks[f'train_{j}'][4 * (n - 1):4 * n])) for j in range(8) for n in range(1, 33)]
    write('P2Q_BATCH_SEQUENCE_SEAL.json', dict(status='SEALED_BEFORE_SEQUENCE_OPERATOR_OUTCOMES', views=views, protocol_sha256=sha(A / '12_P2Q_RANDOM_COCYCLE/Project1_P2Q_TimeOrderedRandomCocycle_Bridge_Protocol_v1_FROZEN.md'), freeze_sha256=sha(A / '12_P2Q_RANDOM_COCYCLE/P2Q_PROTOCOL_FREEZE.json'), sequence_order='unchanged historical order', input_tokens_per_batch=1024))
    event('ASSETS_AND_SEQUENCE_SEALED')
    return (checkpoints, banks, preds, f0, seeds)

def anchors(model, dense, banks):
    rows = []
    for name in ['B0_1', 'B0_17', 'B3_17', 'B5_17']:
        j, n = map(int, name[1:].split('_'))
        hp = capture(model, dense, banks[f'train_{j}'][4 * (n - 1):4 * n])
        old = torch.load(P / f'captures/fresh_P2P/capture_{name}.pt', weights_only=True)
        eq = [torch.equal(t, old[k]) for t, k in zip(hp, ['h', 'phi', 'g'])]
        h, p, g = [t.double() for t in hp]
        gram = h.T @ (g @ g.T * (p @ p.T)) @ h / 16
        gram = (gram + gram.T) / 2
        e, u = torch.linalg.eigh(gram)
        s = e.flip(0).clamp_min(0).sqrt()
        u = u.flip(1)[:, :4]
        ref = np.load(P / f'singular_details/singular_{name}.npz', allow_pickle=False)
        sr = abs(float(s[0]) - float(ref['s'][0])) / float(ref['s'][0])
        ur = torch.from_numpy(ref['u'][:, :4])
        pe = norm(u @ u.T - ur @ ur.T)
        rows.append(dict(batch=name, capture_bitwise_equal=dict(zip(['h', 'phi', 'g'], eq)), capture_sha256=dict(zip(['h', 'phi', 'g'], [tsha(t) for t in hp])), s1_relative_error=sr, top4_left_projector_frobenius=pe))
        write('P2Q_P2P_ANCHOR_REPLAY.json', dict(status='IN_PROGRESS', anchors=rows, s1_tolerance=1e-10, projector_tolerance=1e-09))
        gate(all(eq) and sr <= 1e-10 and (pe <= 1e-09), 'P2Q_P2P_OPERATOR_REPLAY_FAIL', name)
        d = OUT / 'replay_subset'
        d.mkdir(exist_ok=True)
        torch.save(dict(zip(['h', 'phi', 'g'], hp)), d / f'capture_{name}.pt')
    write('P2Q_P2P_ANCHOR_REPLAY.json', dict(status='PASS', anchors=rows, s1_tolerance=1e-10, projector_tolerance=1e-09))
    event('ANCHOR_REPLAY_PASS')

def batch_actions(h, p, g, m, u):
    cm = []
    ct = []
    for start in range(0, 15, 3):
        mm = m[start:start + 3]
        uu = u[start:start + 3]
        response = torch.matmul(p, mm.transpose(1, 2))
        cm.append(-0.25 * ((response * g[None]).sum(-1) @ h))
        w = (uu @ h.T)[:, :, None] * p[None]
        ct.append(-0.25 * torch.matmul(g.T, w))
    return (torch.cat(cm), torch.cat(ct))

def checkpoint(j, n, ms, us, seed, checkpoints, preds):
    event('TARGET_CHECKPOINT_OPEN', stream=j, update=n)
    for k in range(15):
        r = checkpoints[j, k, n]
        ar = np.load(r['path'], allow_pickle=False)
        obsM = torch.from_numpy(ar['M']).double()
        obsU = torch.from_numpy(ar['U']).double()
        gate(tsha(torch.from_numpy(ar['M'])) == r['M_value_sha256'] and tsha(torch.from_numpy(ar['U'])) == r['U_value_sha256'], detail='checkpoint tensor hash')
        if n == 0:
            gate(torch.equal(obsM, seed) and torch.count_nonzero(obsU) == 0, detail='exact initial state')
        CP_AUDIT.append(dict(seed=j, arm=k, update=n, path=r['path'], sha256=r['sha256'], opened_time_ns=time.time_ns()))
        dm = obsM - seed
        du = obsU
        on = (dm.square().sum() + du.square().sum()).sqrt()
        for q in range(2):
            pm = ms[q][k] - seed
            pu = us[q][k]
            pn = (pm.square().sum() + pu.square().sum()).sqrt()
            er = float(((pm - dm).square().sum() + (pu - du).square().sum()).sqrt() / on.clamp_min(1e-30))
            co = float(((pm * dm).sum() + (pu * du).sum()) / (pn * on).clamp_min(1e-30))
            COMP.append(dict(seed=j, arm=k, epsilon=preds[k]['epsilon'], r_eta=preds[k]['r_eta'], update=n, model=MODELS[q], displacement_error=er, displacement_cosine=co, observed_displacement_norm=float(on), predicted_displacement_norm=float(pn), observed_amplitude=float((obsM.square().sum() + obsU.square().sum()).sqrt()), predicted_amplitude=float((ms[q][k].square().sum() + us[q][k].square().sum()).sqrt()), local_summary_included=n in [1, 2, 4, 8]))

def process(streams, model, dense, banks, preds, f0, seeds, checkpoints):
    if min(streams) >= 6:
        gate((OUT / 'P2Q_PRIMARY_DECISION.sha256').is_file() and sha(OUT / 'P2Q_PRIMARY_DECISION.json') == DECISION_HASH, detail='replication seal gate')
    event('TARGET_SCALAR_ROWS_OPEN', streams=list(streams), columns=['seed', 'arm', 'update', 'M_norm', 'U_norm'])
    tab = pq.read_table(H / 'P2H_TRAJECTORIES.parquet', columns=['seed', 'arm', 'update', 'M_norm', 'U_norm'], filters=[('seed', 'in', list(streams)), ('update', '<=', 32)])
    observed = {(r['seed'], r['arm'], r['update']): (r['M_norm'] ** 2 + r['U_norm'] ** 2) ** 0.5 for r in tab.to_pylist()}
    gate(len(observed) == len(streams) * 15 * 33, detail='filtered scalar rows')
    etaM = torch.tensor([r['eta_M'] for r in preds], dtype=torch.float64)
    etaU = torch.tensor([r['eta_U'] for r in preds], dtype=torch.float64)
    aM = 1 - etaM * torch.tensor([r['mu'] for r in preds], dtype=torch.float64)
    aU = 1 - etaU * torch.tensor([r['lambda_'] for r in preds], dtype=torch.float64)
    fu = torch.from_numpy(f0['u']).double()
    fv = torch.from_numpy(f0['v'].reshape(-1, 4)).double()
    fs = torch.from_numpy(f0['s']).double()
    for j in streams:
        seed = seeds[j].double()
        ms = [seed[None].repeat(15, 1, 1) for _ in range(2)]
        us = [torch.zeros(15, 576, dtype=torch.float64) for _ in range(2)]
        amps = {}
        for n in range(33):
            if n:
                ids = banks[f'train_{j}'][4 * (n - 1):4 * n]
                hp = capture(model, dense, ids)
                h, p, g = [t.double() for t in hp]
                cm, ct = batch_actions(h, p, g, ms[0], us[0])
                op = CrossOperator(h, p, g)
                errC = norm(cm[0] - op.c(ms[0][0])) / max(norm(cm[0]), 1e-30)
                errCT = norm(ct[0] - op.ct(us[0][0])) / max(norm(ct[0]), 1e-30)
                gate(max(errC, errCT) <= 1e-12, 'P2Q_MATRIX_FREE_ACTION_FAIL', f'B{j}_{n}')
                BATCH_AUDIT.append(dict(stream=j, update=n, capture_sha256=dict(zip(['h', 'phi', 'g'], [tsha(t) for t in hp])), C_relative_error=errC, CT_relative_error=errCT))
                ms[0], us[0] = (aM[:, None, None] * ms[0] + etaM[:, None, None] * ct, aU[:, None] * us[0] + etaU[:, None] * cm)
                cm0 = ms[1].reshape(15, -1) @ fv * fs @ fu.T
                ct0 = (us[1] @ fu * fs @ fv.T).reshape(15, 576, 1536)
                ms[1], us[1] = (aM[:, None, None] * ms[1] + etaM[:, None, None] * ct0, aU[:, None] * us[1] + etaU[:, None] * cm0)
            for q in range(2):
                an = (ms[q].square().sum((1, 2)) + us[q].square().sum(1)).sqrt().numpy()
                if n in [8, 32]:
                    amps[q, n] = an
                for k in range(15):
                    TRAJ.append(dict(seed=j, arm=k, epsilon=preds[k]['epsilon'], r_eta=preds[k]['r_eta'], update=n, model=MODELS[q], M_norm=norm(ms[q][k]), U_norm=norm(us[q][k]), amplitude=float(an[k]), observed_amplitude=observed[j, k, n]))
            if n in TIMES:
                checkpoint(j, n, ms, us, seed, checkpoints, preds)
            if n and n % 8 == 0:
                log(f'STREAM {j}: {n}/32; elapsed {time.monotonic() - START:.1f}s')
        for k in range(15):
            go = np.log(observed[j, k, 32] / observed[j, k, 8]) / 24
            GAINS.append(dict(seed=j, arm=k, epsilon=preds[k]['epsilon'], r_eta=preds[k]['r_eta'], Gamma_observed=float(go), Gamma_RTO=float(np.log(amps[0, 32][k] / amps[0, 8][k]) / 24), Gamma_F0top4=float(np.log(amps[1, 32][k] / amps[1, 8][k]) / 24)))
        event('STREAM_COMPLETE', stream=j)

def metrics(streams):
    e = np.empty((len(streams), 15, 2))
    c = e.copy()
    gam = np.empty((len(streams), 15, 3))
    for z, j in enumerate(streams):
        for k in range(15):
            for q in range(2):
                rows = [r for r in COMP if r['seed'] == j and r['arm'] == k and (r['model'] == MODELS[q]) and (r['update'] in [1, 2, 4, 8])]
                gate(len(rows) == 4, detail='local checkpoint count')
                e[z, k, q] = np.median([r['displacement_error'] for r in rows])
                c[z, k, q] = np.median([r['displacement_cosine'] for r in rows])
            r = next((r for r in GAINS if r['seed'] == j and r['arm'] == k))
            gam[z, k] = [r['Gamma_observed'], r['Gamma_RTO'], r['Gamma_F0top4']]
    ratio = e[:, :, 0] / np.maximum(e[:, :, 1], 1e-30)
    return (e, c, ratio, gam)

def gainstats(g):
    o = g[:, 0]
    r = g[:, 1]
    f = g[:, 2]
    den = max(np.linalg.norm(o), 1e-30)
    er = np.linalg.norm(r - o) / den
    ef = np.linalg.norm(f - o) / den
    return dict(E_Gamma_RTO=float(er), rho_Gamma_RTO=float(np.corrcoef(r, o)[0, 1]), R_Gamma=float(er / max(ef, 1e-30)), E_Gamma_F0top4=float(ef), rho_Gamma_F0top4=float(np.corrcoef(f, o)[0, 1]))

def primary():
    global DECISION_HASH
    e, c, ratio, gam = metrics(range(6))
    seedE = np.median(e[:, :, 0], 1)
    seedC = np.median(c[:, :, 0], 1)
    seedR = np.median(ratio, 1)
    rng = np.random.default_rng(2026100901)
    draws = rng.integers(0, 6, (20000, 6))
    samples = dict(E_loc_RTO=np.median(seedE[draws], 1), C_loc_RTO=np.median(seedC[draws], 1), R_loc=np.median(seedR[draws], 1))
    points = dict(E_loc_RTO=float(np.median(seedE)), C_loc_RTO=float(np.median(seedC)), R_loc=float(np.median(seedR)))
    points.update(gainstats(np.median(gam, 0)))
    for n in gainstats(np.median(gam, 0)):
        samples[n] = np.empty(20000)
    for b, ix in enumerate(draws):
        for n, v in gainstats(np.median(gam[ix], 0)).items():
            samples[n][b] = v
    summary = {n: dict(point=points[n], lower95=float(np.percentile(v, 2.5)), upper95=float(np.percentile(v, 97.5))) for n, v in samples.items()}
    np.savez(OUT / 'P2Q_BOOTSTRAP_RESAMPLES.npz', seed_identity_draws=draws, **samples)
    write('P2Q_BOOTSTRAP_SUMMARY.json', dict(seed=2026100901, resamples=20000, CI='percentile 95%', seed_level_E_loc_RTO=seedE.tolist(), seed_level_C_loc_RTO=seedC.tolist(), seed_level_R_loc=seedR.tolist(), statistics=summary))
    lo = lambda n: summary[n]['lower95']
    hi = lambda n: summary[n]['upper95']
    nb = int((seedR < 1).sum())
    ng = int(sum((gainstats(g)['E_Gamma_RTO'] < gainstats(g)['E_Gamma_F0top4'] for g in gam)))
    lc = dict(error_upper95_le_030=hi('E_loc_RTO') <= 0.3, cosine_lower95_ge_085=lo('C_loc_RTO') >= 0.85, ratio_upper95_le_075=hi('R_loc') <= 0.75, seeds_better_at_least_five=nb >= 5)
    wc = dict(error_upper95_le_050=hi('E_Gamma_RTO') <= 0.5, correlation_lower95_ge_075=lo('rho_Gamma_RTO') >= 0.75, ratio_upper95_le_075=hi('R_Gamma') <= 0.75, seeds_better_at_least_five=ng >= 5)
    local = all(lc.values())
    window = all(wc.values())
    improve = lc['ratio_upper95_le_075'] and lc['seeds_better_at_least_five']
    label = 'P2Q_P2H_WINDOW_COCYCLE_BRIDGE_SUPPORTED' if local and window else 'P2Q_LOCAL_COCYCLE_SUPPORTED_WINDOW_NOT_CLOSED' if local else 'P2Q_COCYCLE_IMPROVEMENT_INSUFFICIENT_FOR_CLOSURE' if not (lc['error_upper95_le_030'] and lc['cosine_lower95_ge_085']) and improve else 'P2Q_STOCHASTIC_TANGENT_BRIDGE_NOT_SUPPORTED_OR_UNRESOLVED'
    write('P2Q_PRIMARY_DECISION.json', dict(status='SEALED', label=label, local_criterion_pass=local, window_criterion_pass=window, local_components=lc, window_components=wc, primary_seeds_local_better=nb, primary_seeds_gain_better=ng, statistics=summary, primary_streams=list(range(6)), bootstrap_sha256=sha(OUT / 'P2Q_BOOTSTRAP_RESAMPLES.npz'), source_sha256=sha(__file__), Q3_guidance=False))
    DECISION_HASH = sha(OUT / 'P2Q_PRIMARY_DECISION.json')
    (OUT / 'P2Q_PRIMARY_DECISION.sha256').write_text(DECISION_HASH + '  P2Q_PRIMARY_DECISION.json\n')
    event('PRIMARY_DECISION_HASH_SEALED', sha256=DECISION_HASH, label=label)
    return label

def replication(label):
    e, c, ratio, gam = metrics([6, 7])
    sr = np.median(ratio, 1)
    cc = float(np.median(c[:, :, 0]))
    gs = gainstats(np.median(gam, 0))
    checks = dict(seed6_local_ratio_lt_one=bool(sr[0] < 1), seed7_local_ratio_lt_one=bool(sr[1] < 1), combined_cosine_ge_075=cc >= 0.75)
    if label == 'P2Q_P2H_WINDOW_COCYCLE_BRIDGE_SUPPORTED':
        checks.update(gain_RTO_better=gs['E_Gamma_RTO'] < gs['E_Gamma_F0top4'], gain_correlation_ge_060=gs['rho_Gamma_RTO'] >= 0.6)
    status = 'P2Q_REPLICATION_DIRECTIONALLY_CONSISTENT' if all(checks.values()) else 'P2Q_REPLICATION_MIXED' if any(checks.values()) else 'P2Q_REPLICATION_NOT_CONSISTENT'
    write('P2Q_REPLICATION.json', dict(status=status, checks=checks, seed_level_R_loc=sr.tolist(), combined_C_loc_RTO=cc, seed_level_E_loc_RTO=np.median(e[:, :, 0], 1).tolist(), gain_statistics=gs, per_seed_gain_statistics=[gainstats(g) for g in gam], primary_label_unchanged=label, primary_decision_sha256=DECISION_HASH))
    event('REPLICATION_COMPLETE', status=status)
    return status

def main():
    shutil.copy2(__file__, OUT / 'source/run_p2q.py')
    inherited()
    write('P2Q_IMPLEMENTATION_CONVENTIONS.json', dict(source_sha256=sha(__file__), simultaneous_update='both right-hand sides use x_n', capture='once per sequence batch; four individual sequences; inherited full-model float32 capture', matrix_free='vectorized across arms in chunks of three; no truncation', F0='exact historical float32 triplets cast to float64 without reorthogonalization', bootstrap='numpy default_rng; same six seed identities paired for every metric', significant_local_improvement='frozen local upper95 ratio <=0.75 and at least 5/6 seed median ratios <1', replication_mixed='some but not all applicable frozen component checks pass', zero_checkpoint='initial tensor equality check; n=0 excluded from local summaries', target_access='replication checkpoint bytes may be hash audited; tensor and scalar outcome reads require verified primary seal', no_Q3_guidance=True))
    cps, banks, preds, f0, seeds = audit()
    model = AutoModelForCausalLM.from_pretrained(str(P / 'inputs/model'), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    gate((model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers) == (576, 1536, 30), detail='architecture')
    dense = model.model.layers[20].mlp
    anchors(model, dense, banks)
    process(range(6), model, dense, banks, preds, f0, seeds, cps)
    label = primary()
    process([6, 7], model, dense, banks, preds, f0, seeds, cps)
    rep = replication(label)
    pq.write_table(pa.Table.from_pylist(TRAJ), OUT / 'P2Q_COCYCLE_TRAJECTORIES.parquet')
    csvwrite('P2Q_CHECKPOINT_COMPARISON.csv', COMP)
    csvwrite('P2Q_FINITE_TIME_GAINS.csv', GAINS)
    write('P2Q_OPERATOR_ACTION_AUDIT.json', dict(status='PASS', batches=BATCH_AUDIT, high_dimensional_cross_matrices_saved=0))
    write('P2Q_TARGET_ACCESS_AUDIT.json', CP_AUDIT)
    event('EXECUTION_COMPLETE', elapsed_seconds=time.monotonic() - START)
    write('P2Q_COMPLETION.json', dict(status='COMPLETE', primary_label=label, replication=rep, sequence_batches=256, sequence_full_model_forwards=1024, anchor_full_model_forwards=16, training_updates=0, optimizer=False, new_banks=0, network_downloads=0, elapsed_seconds=time.monotonic() - START, checkpoint_tensors_read=840, trajectory_rows=len(TRAJ), checkpoint_comparison_rows=len(COMP), finite_time_gain_rows=len(GAINS), primary_decision_sha256=DECISION_HASH))
    log('COMPLETE ' + label + ' ' + rep)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        write('P2Q_NONSCIENTIFIC_STOP.json', dict(status=str(e).split(':')[0] if str(e).startswith('P2Q_') else 'P2Q_TECHNICAL_EXECUTION_FAIL', error=str(e), scientific_verdict=None, elapsed_seconds=time.monotonic() - START))
        sys.exit(1)
