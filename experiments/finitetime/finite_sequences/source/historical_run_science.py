# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import importlib.util, json, time, csv, traceback, sys, hashlib
from pathlib import Path
import numpy as np
import torch
W = Path(__file__).resolve().parent
ROOT = W.parents[1]
OUT = W / 'results'
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('nv1_methods', ROOT / 'work/nv1/run_nv1.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.W = W
c.OUT = OUT
c.NV0 = W / 'nv0_source'
c.AUDITS = []
START = time.time()
EVENTS = []
NUMERICS = []

def event(action, **kw):
    EVENTS.append(dict(action=action, time_ns=time.time_ns(), **kw))
    c.write('P2P_EVENT_LEDGER.json', EVENTS)
    c.log(action)

def csvwrite(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def cap(view, banks, model, dense):
    ids = banks[view['bank']][view['start']:view['stop']]
    h, p, g = c.capture(model, dense, ids)
    replay = c.capture(model, dense, ids)
    eq = [torch.equal(a, b) for a, b in zip((h, p, g), replay)]
    c.gate(all(eq), 'P2O_DETERMINISTIC_REPLAY_FAIL', view['name'])
    torch.save(dict(h=h, phi=p, g=g), W / f"capture_{view['name']}.pt")
    return (h.double(), p.double(), g.double(), eq, [c.tsha(t) for t in (h, p, g)])

def singular(view, banks, model, dense):
    name = view['name']
    h, p, g, eq, hashes = cap(view, banks, model, dense)
    op = c.CrossOperator(h, p, g)
    ktok = g @ g.T * (p @ p.T)
    gram = h.T @ ktok @ h / 16
    gram = (gram + gram.T) / 2
    eig, u = torch.linalg.eigh(gram)
    ratio = float(eig[0] / eig[-1])
    c.gate(ratio >= -1e-08, 'P2O_SINGULAR_NUMERICS_FAIL', name + ' negative eigenvalue')
    s = eig.flip(0).clamp_min(0).sqrt()
    u = u.flip(1)[:, :8]
    v = torch.stack([op.ct(u[:, k]) / s[k] for k in range(8)], dim=-1)
    residual = []
    for k in range(8):
        a = c.norm(op.c(v[:, :, k]) - s[k] * u[:, k]) / float(s[k])
        b = c.norm(op.ct(u[:, k]) - s[k] * v[:, :, k]) / float(s[k])
        residual.append([a, b])
        if float(s[k] / s[0]) >= 0.001:
            c.gate(max(a, b) <= 0.001, 'P2O_SINGULAR_NUMERICS_FAIL', name + ' mode ' + str(k + 1))
    c.gate(max((max(r) for r in residual[:4])) <= 0.0002, 'P2O_SINGULAR_NUMERICS_FAIL', name + ' top four')
    orth = float(torch.linalg.matrix_norm(v.flatten(0, 1).T @ v.flatten(0, 1) - torch.eye(8, dtype=torch.float64)))
    np.save(W / f'right_{name}.npy', v.numpy())
    np.savez(W / f'singular_{name}.npz', s=s.numpy(), u=u.numpy(), residuals=residual, min_eigenvalue_ratio=ratio)
    rec = dict(batch=name, group=view['group'], capture_replay_exact=eq, capture_sha256=hashes, min_eigenvalue_ratio=ratio, residuals=residual, right_orthogonality_error=orth)
    NUMERICS.append(rec)
    c.write('P2P_SINGULAR_NUMERICS.json', NUMERICS)
    event('SCIENTIFIC_BATCH_COMPLETE', batch=name, group=view['group'])
    return dict(name=name, group=view['group'], s=s.numpy(), u=u.numpy(), vpath=W / f'right_{name}.npy', frob2=float((s * s).sum()), E={k: float((s[:k] ** 2).sum() / (s * s).sum()) for k in [1, 4, 8, 16]})

def right_gram(recs):
    vs = [np.load(r['vpath'], mmap_mode='r').reshape(-1, 8) for r in recs]
    n = len(recs) * 8
    gram = torch.zeros(n, n, dtype=torch.float64)
    for start in range(0, 884736, 16384):
        block = torch.from_numpy(np.concatenate([v[start:start + 16384] for v in vs], axis=1))
        gram += block.T @ block
    return (gram, vs)

def pooled(recs):
    us = torch.from_numpy(np.concatenate([r['u'] for r in recs], axis=1))
    su = us @ us.T / len(recs)
    eu, qu = torch.linalg.eigh(su)
    qu = qu[:, -24:].flip(1)
    gr, vs = right_gram(recs)
    er, z = torch.linalg.eigh(gr / len(recs))
    order = torch.arange(len(er) - 1, len(er) - 25, -1)
    weights = z[:, order] / torch.sqrt(len(recs) * er[order])[None, :]
    qm = np.lib.format.open_memmap(W / 'pooled_right.npy', mode='w+', dtype=np.float64, shape=(884736, 24))
    for start in range(0, 884736, 16384):
        block = torch.from_numpy(np.concatenate([v[start:start + 16384] for v in vs], axis=1))
        qm[start:start + len(block)] = (block @ weights).numpy()
    qm.flush()
    qmt = torch.from_numpy(np.asarray(qm))
    c.gate(float(torch.linalg.matrix_norm(qmt.T @ qmt - torch.eye(24, dtype=torch.float64))) < 1e-09, 'P2O_SINGULAR_NUMERICS_FAIL', 'pooled right orthonormality')
    np.savez(OUT / 'P2P_POOLED_SUBSPACE.npz', Q_U=qu.numpy(), Q_M=np.asarray(qm), left_pool_eigenvalues=eu.flip(0).numpy(), right_pool_eigenvalues=er.flip(0).numpy(), right_mode_gram=gr.numpy(), right_pool_weights=weights.numpy())
    event('PRIMARY_POOLED_BASIS_FROZEN', dimension=24)
    return (qu, qmt, gr)

def restriction(rec, qu, qm):
    a = torch.load(W / f"capture_{rec['name']}.pt", weights_only=True)
    op = c.CrossOperator(a['h'].double(), a['phi'].double(), a['g'].double())
    k = qu.T @ torch.stack([op.c(qm[:, i].reshape(576, 1536)) for i in range(24)], dim=1)
    vv = torch.from_numpy(np.load(rec['vpath']).reshape(-1, 8))
    uu = torch.from_numpy(rec['u'])
    gammaU = float(((qu.T @ uu) ** 2).sum() / 8)
    gammaM = float(((qm.T @ vv) ** 2).sum() / 8)
    tau = 1 - float((k * k).sum()) / rec['frob2']
    rec.update(K=k.numpy(), Gamma_U=gammaU, Gamma_M=gammaM, Gamma=(gammaU + gammaM) / 2, tau=tau)
    event('RESTRICTED_OPERATOR_COMPLETE', batch=rec['name'])

def intrinsic(a, b):
    top = a.T @ b - b.T @ a
    bot = a @ b.T - b @ a.T
    return float(np.sqrt((top * top).sum() + (bot * bot).sum()) / (4 * np.linalg.norm(a) * np.linalg.norm(b)))

def geometry(recs, gr=None):
    n = len(recs)
    D = np.zeros((n, n))
    DU = D.copy()
    DM = D.copy()
    CHI = D.copy()
    rows = []
    vs = None if gr is not None else [np.load(r['vpath'], mmap_mode='r').reshape(-1, 8) for r in recs]
    for i in range(n):
        for j in range(i + 1, n):
            ou = float(((recs[i]['u'][:, :4].T @ recs[j]['u'][:, :4]) ** 2).sum() / 4)
            ov = gr[8 * i:8 * i + 4, 8 * j:8 * j + 4].numpy() if gr is not None else vs[i][:, :4].T @ vs[j][:, :4]
            om = float((ov * ov).sum() / 4)
            du = 1 - ou
            dm = 1 - om
            d = (du + dm) / 2
            ch = intrinsic(recs[i]['K'], recs[j]['K'])
            for x, val in [(D, d), (DU, du), (DM, dm), (CHI, ch)]:
                x[i, j] = x[j, i] = val
            rows.append(dict(batch_a=recs[i]['name'], batch_b=recs[j]['name'], group=recs[i]['group'] + '_' + recs[j]['group'], overlap_U=ou, overlap_M=om, D_U=du, D_M=dm, D_pair=d, chi_C=ch))
    return (D, DU, DM, CHI, rows)

def rotation(ks, mean=None):
    mean = ks.mean(axis=0) if mean is None else mean
    u, s, vh = np.linalg.svd(mean, full_matrices=False)
    d = np.einsum('ia,bac,cj->bij', u[:, :8].T, ks, vh.T[:, :8])
    total = (d * d).sum()
    off = total - (np.diagonal(d, axis1=1, axis2=2) ** 2).sum()
    return (float(np.sqrt(max(off, 0) / total)), d, s, u, vh)

def rho(ks, mean=None):
    mean = ks.mean(axis=0) if mean is None else mean
    den = np.linalg.norm(mean)
    return float(np.sqrt(((ks - mean) ** 2).sum(axis=(1, 2)).mean()) / den) if den > 0 else float('inf')

def bootstrap(recs, D, DU, DM, CHI):
    ks = np.stack([r['K'] for r in recs])
    gamma = np.array([r['Gamma'] for r in recs])
    tau = np.array([r['tau'] for r in recs])
    energies = {k: np.array([r['E'][k] for r in recs]) for k in [1, 4, 8, 16]}
    pairs = np.triu_indices(32, 1)
    rng = np.random.default_rng(2026100814)
    names = ['D_pair', 'D_U', 'D_M', 'rho_Xi', 'Gamma24', 'tau24', 'R_off', 'chi_C'] + [f'E{k}' for k in energies]
    samples = {n: np.empty(20000, dtype=np.float64) for n in names}
    for b in range(20000):
        ix = rng.integers(0, 32, 32)
        i = ix[pairs[0]]
        j = ix[pairs[1]]
        selected = ks[ix]
        vals = dict(D_pair=np.median(D[i, j]), D_U=np.median(DU[i, j]), D_M=np.median(DM[i, j]), rho_Xi=rho(selected), Gamma24=np.median(gamma[ix]), tau24=np.median(tau[ix]), R_off=rotation(selected)[0], chi_C=np.median(CHI[i, j]))
        vals.update({f'E{k}': np.median(x[ix]) for k, x in energies.items()})
        for name, x in vals.items():
            samples[name][b] = x
        if (b + 1) % 5000 == 0:
            c.log(f'BOOTSTRAP {b + 1}/20000')
    points = dict(D_pair=np.median(D[pairs]), D_U=np.median(DU[pairs]), D_M=np.median(DM[pairs]), rho_Xi=rho(ks), Gamma24=np.median(gamma), tau24=np.median(tau), R_off=rotation(ks)[0], chi_C=np.median(CHI[pairs]))
    points.update({f'E{k}': np.median(x) for k, x in energies.items()})
    result = {n: dict(point=float(points[n]), lower95=float(np.percentile(samples[n], 2.5)), upper95=float(np.percentile(samples[n], 97.5))) for n in names}
    np.savez(OUT / 'P2P_BOOTSTRAP_RESAMPLES.npz', **samples)
    c.write('P2P_BOOTSTRAP_SUMMARY.json', dict(seed=2026100814, resamples=20000, confidence_interval='percentile 95%', sample_unit='32 primary batch identities', pair_rule='All 496 distinct positional pairs in each resampled 32-entry sample; repeated identities have zero self-dispersion and self-commutator', pooled_basis='fixed primary pooled basis', mean_basis='SVD of each resampled mean for R_off', statistics=result))
    return result

def classify(x):
    lo = lambda n: x[n]['lower95']
    hi = lambda n: x[n]['upper95']
    small = hi('D_pair') <= 0.1
    order = lo('D_pair') >= 0.3
    pert = hi('rho_Xi') <= 0.25
    nonpert = lo('rho_Xi') >= 0.5
    adequate = lo('Gamma24') >= 0.9 and hi('tau24') <= 0.1
    fixed = hi('R_off') <= 0.2
    rot = lo('R_off') >= 0.35
    negligible = hi('chi_C') <= 0.05
    material = lo('chi_C') >= 0.1
    below_material = hi('chi_C') < 0.1
    insufficient = hi('E4') < 0.9 or (not small and adequate and fixed)
    a = small and pert and negligible
    cc = not a and adequate and fixed and below_material and insufficient
    bb = order and nonpert and rot and material
    label = 'P2O_A_SMALL_NOISE_MEAN_OPERATOR' if a else 'P2O_C_MULTIMODE_OPERATOR' if cc else 'P2O_B_RANDOM_PROJECTOR_MULTIPLICATIVE' if bb else 'P2O_MIXED_OR_UNRESOLVED'
    return (label, dict(small_dispersion=small, order_one_dispersion=order, perturbative_noise=pert, nonperturbative_noise=nonpert, pooled_adequate=adequate, fixed_basis=fixed, strong_rotational_mixing=rot, negligible_commutator=negligible, material_commutator=material, commutator_C_upper_bound_below_material=below_material, leading_four_insufficient=insufficient, A_conjunction=a, B_conjunction=bb, C_conjunction=cc))

def anchored(recs, f0):
    fu = f0['u'].astype(np.float64)
    fv = f0['v'].astype(np.float64).reshape(-1, 4)
    for r in recs:
        uu = r['u'][:, :4]
        vv = np.load(r['vpath'], mmap_mode='r').reshape(-1, 8)[:, :4]
        r['F0_overlap_U'] = float(((fu.T @ uu) ** 2).sum() / 4)
        r['F0_overlap_M'] = float(((fv.T @ vv) ** 2).sum() / 4)
        r['F0_dispersion'] = 1 - (r['F0_overlap_U'] + r['F0_overlap_M']) / 2

def fe_replication(primary, fe, label):
    kp = np.stack([r['K'] for r in primary])
    kf = np.stack([r['K'] for r in fe])
    mean = kp.mean(0)
    rfe = rotation(kf, mean)[0]
    noisefe = rho(kf, mean)
    dispersion = []
    comm = []
    cross = []
    for e in fe:
        ve = np.load(e['vpath'], mmap_mode='r').reshape(-1, 8)[:, :4]
        ds = []
        for p in primary:
            vp = np.load(p['vpath'], mmap_mode='r').reshape(-1, 8)[:, :4]
            du = 1 - float(((e['u'][:, :4].T @ p['u'][:, :4]) ** 2).sum() / 4)
            dm = 1 - float(((ve.T @ vp) ** 2).sum() / 4)
            d = (du + dm) / 2
            ch = intrinsic(e['K'], p['K'])
            ds.append(d)
            comm.append(ch)
            cross.append(dict(batch_a=p['name'], batch_b=e['name'], group='primary_FE', overlap_U=1 - du, overlap_M=1 - dm, D_U=du, D_M=dm, D_pair=d, chi_C=ch))
        dispersion.append(float(np.median(ds)))
    u, s, vh = np.linalg.svd(mean, full_matrices=False)
    qu, qm = POOL
    mean_u = qu.numpy() @ u[:, :4]
    mean_v = qm.numpy() @ vh.T[:, :4]
    mean_d = []
    for e in fe:
        ve = np.load(e['vpath'], mmap_mode='r').reshape(-1, 8)[:, :4]
        mean_d.append(1 - (float(((mean_u.T @ e['u'][:, :4]) ** 2).sum() / 4) + float(((mean_v.T @ ve) ** 2).sum() / 4)) / 2)
    medgamma = float(np.median([e['Gamma'] for e in fe]))
    medtau = float(np.median([e['tau'] for e in fe]))
    medd = float(np.median(dispersion))
    medch = float(np.median(comm))
    if label.startswith('P2O_A_'):
        checks = dict(all_FE_F0_dispersion_below_020=all((e['F0_dispersion'] < 0.2 for e in fe)), all_FE_mean_dispersion_below_020=all((d < 0.2 for d in mean_d)), FE_noise_at_most_050=noisefe <= 0.5)
    elif label.startswith('P2O_B_'):
        checks = dict(FE_median_dispersion_at_least_020=medd >= 0.2, FE_rotation_at_least_025=rfe >= 0.25, FE_median_primary_commutator_at_least_0075=medch >= 0.075)
    elif label.startswith('P2O_C_'):
        checks = dict(FE_median_capture_at_least_085=medgamma >= 0.85, FE_median_tau_at_most_015=medtau <= 0.15, FE_rotation_at_most_030=rfe <= 0.3)
    else:
        checks = {}
    result = 'FE_REPLICATION_MIXED' if not checks else 'FE_REPLICATION_DIRECTIONALLY_CONSISTENT' if all(checks.values()) else 'FE_REPLICATION_CONTRADICTORY' if not any(checks.values()) else 'FE_REPLICATION_MIXED'
    c.write('P2P_FE_REPLICATION.json', dict(status=result, primary_label_unchanged=label, criteria=checks, FE_rotational_index_in_frozen_primary_mean_basis=rfe, FE_noise_relative_to_primary_mean=noisefe, FE_median_primary_dispersion=medd, FE_per_batch_primary_pair_median_dispersion=dispersion, FE_per_batch_F0_dispersion=[e['F0_dispersion'] for e in fe], FE_per_batch_primary_mean_dispersion=mean_d, FE_median_primary_commutator=medch, FE_median_Gamma24=medgamma, FE_median_tau24=medtau, unresolved_primary_rule='No directional branch exists to replicate; report MIXED with all frozen components' if not checks else None, primary_decision_sha256=DECISION_HASH))
    return (result, cross)

def commutator_output(recs, geometryrows, f0):
    rows = [dict(batch_a=r['batch_a'], batch_b=r['batch_b'], group=r['group'], kind='intrinsic', epsilon='', r='', frobenius_norm='', normalized=r['chi_C']) for r in geometryrows]
    pairs = [(i, j) for i in range(len(recs)) for j in range(i + 1, len(recs))]
    eye = np.eye(24)
    zero = np.zeros((24, 24))
    s0 = float(f0['s'][0])
    for eps in [-0.2, -0.08, 0.04, 0.12, 0.3]:
        for mob in [0.25, 1, 4]:
            aa = [np.block([[-eye, (a['K'] / s0).T], [mob * a['K'] / s0, -mob * (1 - eps) * eye]]) for a in recs]
            for i, j in pairs:
                com = aa[i] @ aa[j] - aa[j] @ aa[i]
                n = float(np.linalg.norm(com))
                den = 2 * np.linalg.norm(aa[i]) * np.linalg.norm(aa[j])
                rows.append(dict(batch_a=recs[i]['name'], batch_b=recs[j]['name'], group=recs[i]['group'] + '_' + recs[j]['group'], kind='historical_generator_secondary', epsilon=eps, r=mob, frobenius_norm=n, normalized=float(n / den)))
    csvwrite('P2P_COMMUTATOR_SUMMARY.csv', rows)

def output_tables(recs, rows, primary):
    spectra = []
    summary = []
    for a in recs:
        for i, s in enumerate(a['s']):
            spectra.append(dict(batch=a['name'], group=a['group'], mode=i + 1, singular_value=float(s), squared_value=float(s * s)))
        summary.append(dict(batch=a['name'], group=a['group'], s1=float(a['s'][0]), E1=a['E'][1], E4=a['E'][4], E8=a['E'][8], E16=a['E'][16], F0_overlap_U=a['F0_overlap_U'], F0_overlap_M=a['F0_overlap_M'], F0_D_pair=a['F0_dispersion'], Gamma24_U=a['Gamma_U'], Gamma24_M=a['Gamma_M'], Gamma24=a['Gamma'], tau24=a['tau']))
    csvwrite('P2P_SINGULAR_SPECTRA.csv', spectra)
    csvwrite('P2P_TOP8_PROJECTOR_SUMMARY.csv', summary)
    csvwrite('P2P_PAIRWISE_PROJECTOR_GEOMETRY.csv', rows)
    np.savez(OUT / 'P2P_COMMON_OPERATOR_MATRICES.npz', batch_names=np.array([r['name'] for r in recs]), groups=np.array([r['group'] for r in recs]), K=np.stack([r['K'] for r in recs]), primary_mean=np.stack([r['K'] for r in primary]).mean(0))
    ks = np.stack([r['K'] for r in primary])
    r, d, s, u, vh = rotation(ks)
    c.write('P2P_MEAN_NOISE_SUMMARY.json', dict(rho_Xi=rho(ks), R_off=r, mean_singular_values=s.tolist(), diagonal_coefficient_covariance=np.cov(np.diagonal(d, axis1=1, axis2=2), rowvar=False).tolist(), top8_singular_value_CV=(np.stack([a['s'][:8] for a in primary]).std(0) / np.stack([a['s'][:8] for a in primary]).mean(0)).tolist(), mean_operator_degenerate=bool(np.linalg.norm(ks.mean(0)) == 0)))

def main():
    global POOL, DECISION_HASH
    pp = ROOT / 'work/nv1/results/P2P_NV1_NUMERICAL_VALIDATION_PASS.json'
    expected = (pp.parent / 'P2P_NV1_PASS.sha256').read_text().split()[0]
    c.gate(c.sha(pp) == expected and json.loads(pp.read_text())['status'] == 'P2P_NV1_PASS', 'P2O_REQUIRED_ARTIFACT_OR_EXECUTION_FAIL', 'NV1 pass seal')
    event('FRESH_P2P_STARTED_AFTER_VERIFIED_NV1_PASS', pass_sha256=expected, source_sha256=c.sha(__file__))
    banks, views, f0 = c.audit()
    event('FRESH_ASSETS_REAUDITED_AND_40_VIEWS_RESEALED')
    model = c.load_model()
    dense = model.model.layers[20].mlp
    pr = c.probes(f0)
    for v in [views[0], views[32]]:
        c.validate(model, dense, banks[v['bank']][v['start']:v['stop']], v['name'], pr)
    c.write('P2P_CAPTURE_REPLAY_AUDIT.json', dict(status='PASS', audits=c.AUDITS, scientific_summaries_opened=False))
    primary = [singular(v, banks, model, dense) for v in views[:32]]
    qu, qm, gr = pooled(primary)
    POOL = (qu, qm)
    for rec in primary:
        restriction(rec, qu, qm)
    anchored(primary, f0)
    D, DU, DM, CHI, rows = geometry(primary, gr)
    ci = bootstrap(primary, D, DU, DM, CHI)
    label, checks = classify(ci)
    c.write('P2P_PRIMARY_DECISION.json', dict(status='PRIMARY_DECISION_SEALED_BEFORE_FE_SCIENCE', primary_label=label, batches=32, bootstrap_sha256=c.sha(OUT / 'P2P_BOOTSTRAP_SUMMARY.json'), protocol_freeze_sha256=c.sha(c.PKG / '08_P2O_PROTOCOL_FREEZE/P2O_PROTOCOL_FREEZE.json'), conjunctions=checks, statistics=ci, FE_scientific_batches_evaluated=0, NV1_values_used_as_scientific_evidence=False, sealed_at_ns=time.time_ns()))
    DECISION_HASH = c.sha(OUT / 'P2P_PRIMARY_DECISION.json')
    (OUT / 'P2P_PRIMARY_DECISION.sha256').write_text(DECISION_HASH + '  P2P_PRIMARY_DECISION.json\n')
    event('PRIMARY_DECISION_HASH_SEALED', sha256=DECISION_HASH, label=label)
    c.gate(c.sha(OUT / 'P2P_PRIMARY_DECISION.json') == DECISION_HASH, 'P2O_REQUIRED_ARTIFACT_OR_EXECUTION_FAIL', 'primary seal before FE')
    event('FE_SCIENCE_OPENED', primary_decision_sha256=DECISION_HASH)
    fe = [singular(v, banks, model, dense) for v in views[32:]]
    for e in fe:
        restriction(e, qu, qm)
    anchored(fe, f0)
    fresult, cross = fe_replication(primary, fe, label)
    _, _, _, _, ferows = geometry(fe)
    rows += cross + ferows
    output_tables(primary + fe, rows, primary)
    commutator_output(primary + fe, rows, f0)
    c.gate(c.sha(OUT / 'P2P_PRIMARY_DECISION.json') == DECISION_HASH, 'P2O_REQUIRED_ARTIFACT_OR_EXECUTION_FAIL', 'primary seal after FE')
    c.write('P2P_CAPTURE_REPLAY_AUDIT.json', dict(status='PASS', audits=c.AUDITS, scientific_batch_replays=NUMERICS, FE_numerical_validation_distinct_from_FE_science=True))
    event('P2P_SCIENTIFIC_EXECUTION_COMPLETE', label=label, FE_result=fresult, elapsed_seconds=time.time() - START)
    c.write('P2P_COMPLETION.json', dict(status='COMPLETE', primary_label=label, FE_result=fresult, elapsed_seconds=time.time() - START, scientific_batches=40, capture_sequences_including_replays=336, model_forward_calls=368, **c.COUNTS))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        label = str(e).split(':')[0] if str(e).startswith(('P2O_', 'P2P_NV1_')) else 'P2O_REQUIRED_ARTIFACT_OR_EXECUTION_FAIL'
        c.write('P2P_NONSCIENTIFIC_STOP.json', dict(status=label, scientific_verdict=None, error=str(e), type=type(e).__name__, elapsed_seconds=time.time() - START, completed_scientific_batches=len(NUMERICS), **c.COUNTS))
        sys.exit(1)
