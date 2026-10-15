# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""T4 analysis of sealed caches only; no torch import/model execution."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv, hashlib, json, sys, platform, time
from pathlib import Path
from collections import Counter
import numpy as np
RUN = Path(__file__).resolve().parents[1]
PKG = RUN / 'control_package/Project1_P2D_Codex_Control_Package_v1p1'
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
TIMES = (41, 146, 503, 1740, 5033)
LAYERS = (7, 15)
PROBES = ('local', 'objective')
BOOT_N = 2000
BOOT_SEED = 2026100701
PSD_RELATIVE_TOLERANCE = 1e-08

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def ident(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}

def read(p):
    return json.loads(Path(p).read_text())

def write(name, o):
    with (RUN / name).open('x') as f:
        json.dump(o, f, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
        f.write('\n')

def sealed(name):
    p = RUN / (name + '.json')
    assert sha(p) == (RUN / (name + '.sha256')).read_text().split()[0]
    d = read(p)
    for r in d.get('artifacts', {}).values():
        assert sha(r['path']) == r['sha256'], ('STAGE_ARTIFACT_HASH_MISMATCH', r['path'])
    return d

def cf(a, b):
    den = np.linalg.norm(a, axis=(-2, -1)) * np.linalg.norm(b, axis=(-2, -1))
    return np.divide(np.sum(a * b, axis=(-2, -1)), den, out=np.full(np.shape(den), np.nan), where=den > 0)

def deff(a):
    den = np.sum(a * a, axis=(-2, -1))
    tr = np.trace(a, axis1=-2, axis2=-1)
    return np.divide(tr * tr, den, out=np.full(np.shape(den), np.nan), where=den > 0)

def spectrum(g, v, hi):
    eig, u = np.linalg.eigh((g + g.T) / 2)
    eig = eig[::-1]
    u = u[:, ::-1]
    assert np.isfinite(eig).all() and eig[0] >= 0
    assert eig[-1] >= -PSD_RELATIVE_TOLERANCE * max(eig[0], np.finfo(float).tiny), ('PSD_NUMERICAL_GUARD', eig.tolist())
    raw = eig.copy()
    eig = np.maximum(eig, 0)
    A = float(np.trace(g))
    assert A > 0, ('ZERO_OPERATOR_SCALE', A)
    p = eig / eig.sum()
    ent = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    cum = np.cumsum(p)
    w = v.T @ hi @ u
    sentinel = np.abs(v[0] @ w) ** 2
    return {'trace_A': A, 'd_eff': float(deff(g)), 'd_ent': ent, 'd50': int(np.searchsorted(cum, 0.5) + 1), 'd80': int(np.searchsorted(cum, 0.8) + 1), 'd90': int(np.searchsorted(cum, 0.9) + 1), 'eigenvalues': raw, 'normalized_spectrum': p, 'sentinel_mode1_eigenloadings': sentinel, 'eigenvectors': u}

def geometry(a, b, k):
    _, ua = np.linalg.eigh((a + a.T) / 2)
    _, ub = np.linalg.eigh((b + b.T) / 2)
    ca = ua[:, -k:]
    cb = ub[:, -k:]
    s = np.linalg.svd(ca.T @ cb, compute_uv=False)
    angles = np.degrees(np.arccos(np.clip(s, 0, 1)))
    return {'k': k, 'principal_angles_degrees': angles.tolist(), 'projector_overlap_Qk': float(np.sum(s * s) / k)}

def bootstrap_geometry(a, b, k):
    _, ua = np.linalg.eigh((a + np.swapaxes(a, -1, -2)) / 2)
    _, ub = np.linalg.eigh((b + np.swapaxes(b, -1, -2)) / 2)
    ca = ua[:, :, -k:]
    cb = ub[:, :, -k:]
    ss = np.linalg.svd(np.swapaxes(ca, -1, -2) @ cb, compute_uv=False)
    return (np.degrees(np.arccos(np.clip(ss, 0, 1))), np.sum(ss * ss, axis=1) / k)

def stability(a, b):
    c = float(cf(a, b))
    x, y = (float(deff(a)), float(deff(b)))
    delta = abs(x - y) / ((x + y) / 2)
    return {'shape_cosine': c, 'd_eff_half0': x, 'd_eff_half1': y, 'relative_d_eff_difference': delta, 'status': 'STABLE' if c >= 0.9 and delta <= 0.25 else 'UNRESOLVED_GEOMETRY'}

def interval(a):
    assert np.isfinite(a).all()
    return {'bootstrap_mean': float(np.mean(a)), 'bootstrap_q025': float(np.quantile(a, 0.025)), 'bootstrap_q975': float(np.quantile(a, 0.975))}

def csvout(name, rr):
    rows = []
    for row in rr:
        r = dict(row, protocol_sha256=META['protocol']['sha256'], basis_sha256=META['basis']['sha256'], Bank_T_seal_sha256=META['Bank_T_seal_sha256'], source_provenance_path=str(RUN / 'T4_PROVENANCE_AUDIT.json'), source_provenance_sha256=PROVENANCE_SHA, checkpoint_identity_audit_path=str(RUN / 'T0_CHECKPOINT_ENVIRONMENT_AUDIT.json'), checkpoint_identity_audit_sha256=sha(RUN / 'T0_CHECKPOINT_ENVIRONMENT_AUDIT.json'), model_repository=META['model_repository'], checkpoint_revision=CP.get(row.get('tokens_B'), {}).get('revision_commit_sha'), source_paths_json=json.dumps(META['source_artifacts']), software_environment_json=json.dumps(META['software_environment']))
        for k, v in list(r.items()):
            if isinstance(v, (list, dict, np.ndarray)):
                r[k] = json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)
        rows.append(r)
    fields = list(dict.fromkeys((k for r in rows for k in r)))
    with (RUN / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    global META, PROVENANCE_SHA, CP
    seals = {n: sealed(n) for n in ('T0_BANK_T_SEAL', 'T1_NUMERICAL_IDENTITY_SEAL', 'T1_41B_COMPLETE_SEAL', 'T1_PREDICTION_SEAL', 'T2_STAGE_SEAL', 'T3_STAGE_SEAL')}
    META = {k: v for k, v in seals['T3_STAGE_SEAL'].items() if k in ('protocol', 'basis', 'Bank_T_seal', 'Bank_T_seal_sha256', 'source_artifacts', 'model_repository', 'software_environment', 'implementation_sources', 'technical_disposition', 'technical_certification', 'technical_certification_seal', 'canonical_joint_definition')}
    META['source_artifacts'] += [ident(RUN / n) for n in ('T1_41B_TOMOGRAPHY.npz', 'T2_5033_TOMOGRAPHY.npz', 'T3_INTERMEDIATE_TOMOGRAPHY.npz', 'T1_SCALAR_PREDICTIONS.json', 'T2_PREDICTION_SCORE.json')]
    META['analysis_environment'] = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(), 'executable': sys.executable}
    env = read(RUN / 'T0_CHECKPOINT_ENVIRONMENT_AUDIT.json')
    CP = {r['tokens_B']: r for r in env['checkpoints']}
    cache_files = []
    audit_values = []
    mode_only_files = 0
    implementation_checks = {}
    for name in ['T1_41B_41_CACHE_INDEX.json', 'T1_5033_MODE1_5033_CACHE_INDEX.json', 'T2_5033_5033_CACHE_INDEX.json'] + [f'T3_INTERMEDIATE_{t}_CACHE_INDEX.json' for t in (146, 503, 1740)]:
        ci = read(RUN / name)
        assert ci['Bank_T_seal_sha256'] == META['Bank_T_seal_sha256'] and len(ci['files']) == 448
        for r in ci['files']:
            assert sha(r['path']) == r['sha256']
            with np.load(r['path'], allow_pickle=False) as z:
                m = json.loads(str(z['metadata_json']))
                assert m['Bank_T_seal_sha256'] == META['Bank_T_seal_sha256']
                assert m['protocol']['sha256'] == META['protocol']['sha256'] and m['basis']['sha256'] == META['basis']['sha256']
                assert m['checkpoint']['revision_commit_sha'] == CP[m['tokens_B']]['revision_commit_sha']
                for rsrc in m['implementation_sources']:
                    if rsrc['path'] not in implementation_checks:
                        implementation_checks[rsrc['path']] = sha(rsrc['path'])
                    assert implementation_checks[rsrc['path']] == rsrc['sha256'], ('IMPLEMENTATION_SOURCE_CHANGED', rsrc['path'])
                if ci['scalar_only']:
                    assert m['scalar_only'] and (not any(('local_J' in k or 'objective_J' in k for k in z.files)))
                    assert z['L7_g'].shape == z['L15_g'].shape == (1,)
                    mode_only_files += 1
                else:
                    audit_values += m['cache_audit']
                    for l in LAYERS:
                        assert np.array_equal(z[f'L{l}_g'], z[f'L{l}_g_router'] + z[f'L{l}_g_gain']), ('CANONICAL_G_IDENTITY_CHANGED', r['path'], l)
                    payload = m.get('raw_component_response_artifact')
                    if payload:
                        assert sha(payload['path']) == payload['sha256']
                        with np.load(payload['path'], allow_pickle=False) as raw:
                            for l in LAYERS:
                                ur = raw[f'L{l}_u_R']
                                ug = raw[f'L{l}_u_G']
                                u = (ur + ug).astype(np.float64)
                                ui = np.einsum('ith,jth->ij', u, u, dtype=np.float64, optimize=False)
                                assert np.array_equal(ui, z[f'L{l}_u_inner_products']), ('CANONICAL_U_CACHE_CHANGED', r['path'], l)
            cache_files.append(r)
    events = [json.loads(l) for l in (RUN / 'logs/TOMOGRAPHY_EVENTS.jsonl').open()]
    pred_index = next((i for i, e in enumerate(events) if e['event'] == 'T1_PREDICTION_SEAL_SEALED'))
    t2_index = next((i for i, e in enumerate(events) if e['event'] == 'MODEL_LOAD_BEGIN' and e['stage'] == 'T2_5033'))
    assert pred_index < t2_index and mode_only_files == 448
    complete_41_index = next((i for i, e in enumerate(events) if e['event'] == 'T1_41B_COMPLETE_SEAL_SEALED'))
    terminal_scalar_load_index = next((i for i, e in enumerate(events) if e['event'] == 'MODEL_LOAD_BEGIN' and e['stage'] == 'T1_5033_MODE1'))
    assert complete_41_index < terminal_scalar_load_index
    cert = read(RUN / 'T1_NUMERICAL_IDENTITY_CERTIFICATION.json')
    assert cert['certification_branch'] == 'CERT-B'
    old_audit = read(RUN / 'T1_OLD_CACHE_FIELD_AUDIT_v2.json')
    assert old_audit['documents'] == 216
    assert len([e for e in events if e['event'] == 'DOCUMENT_DONE' and e.get('stage') == 'T1_41B']) == 448
    pred_sha = sha(RUN / 'T1_SCALAR_PREDICTIONS.json')
    assert pred_sha == seals['T1_PREDICTION_SEAL']['prediction_sha256']
    assert pred_sha == read(RUN / 'T2_PREDICTION_SCORE.json')['prediction_sha256']
    sourceaudit = read(RUN / 'T0_OPENED_SOURCE_PROVENANCE_AUDIT.json')
    rehash = []
    seen = {}
    recovered = read(RUN / 'T0_RECOVERED_SHARDS.json')
    for r in recovered['shards']:
        p = r['physical_absolute_path']
        actual = sha(p)
        assert actual == r['expected_sha256']
        rehash.append({'path': p, 'sha256': actual, 'status': 'PASS'})
    for cp in CP.values():
        for r in cp['files']:
            key = str(Path(r['path']).resolve())
            if key not in seen:
                seen[key] = sha(r['path'])
            assert seen[key] == r['sha256']
    for r in sourceaudit['source_artifacts']:
        assert sha(r['path']) == r['sha256']
    write('T4_PROVENANCE_AUDIT.json', dict(META, status='PASS', checkpoints=list(CP.values()), stage_seal_hashes={n: sha(RUN / (n + '.json')) for n in seals}, document_caches=cache_files, data_sources_independently_rehashed=rehash, all_checkpoint_weights_final_hash_PASS=True, opened_source_artifacts_final_hash_PASS=True, implementation_source_hashes_PASS=implementation_checks, prediction_sha256_unchanged=pred_sha, prediction_sealed_before_full_terminal_model_load=True, complete_41B_sealed_before_terminal_scalar=True, terminal_T1_mode1_only_cache_count=448, technical_certification_branch='CERT-B', technical_certification_result='DIRECT_JOINT_NUMERICAL_PATH_ONLY', diagnostic_base_forwards=5, diagnostic_doc216_repeat_count=3, old_valid_objective_and_adjoint_fields_reused_documents=216, missing_local_component_recovery_base_forwards=214, certified_component_vector_reuse_documents=[0, 215], raw_41B_component_response_payloads_verified=448, original_direct_guard_tolerances_unchanged=True, restored_FP64_input_conversion_audit=ident(RUN / 'T1_FP64_REDUCTION_CONVENTION_AUDIT.json'), all_old_blocked_attempts_preserved=True, NEW_TRAINING_STEP_COUNT=0, RESERVE_ACCESS_COUNT=0, NEW_NULL_BANK_COUNT=0, BASIS_REFIT_COUNT=0, OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT=0, model_load_count=sum((e['event'] == 'MODEL_LOADED' for e in events)), baseline_full_forward_count_from_completed_production=2686, production_cache_documents=sum((e['event'] == 'DOCUMENT_DONE' for e in events))))
    PROVENANCE_SHA = sha(RUN / 'T4_PROVENANCE_AUDIT.json')
    print('T4 final provenance audit PASS', flush=True)
    data = {}
    for t, name in ((41, 'T1_41B_TOMOGRAPHY.npz'), (5033, 'T2_5033_TOMOGRAPHY.npz')):
        with np.load(RUN / name, allow_pickle=False) as z:
            data[t] = {k: z[k] for k in z.files if k != 'metadata_json'}
    with np.load(RUN / 'T3_INTERMEDIATE_TOMOGRAPHY.npz', allow_pickle=False) as z:
        for t in (146, 503, 1740):
            data[t] = {k[len(f'T{t}_'):]: z[k] for k in z.files if k.startswith(f'T{t}_')}
    basis = {}
    hi = {}
    with np.load(PKG / 'basis/p2b_formala_corrected_6d_basis.npz', allow_pickle=False) as z:
        for l in LAYERS:
            basis[l] = z[f'L{l}_controls_exact_common_projected']
            hi[l] = data[41][f'L{l}_H_inverse_sqrt']
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, 64, size=(BOOT_N, 7, 64), dtype=np.int64) + np.arange(7)[None, :, None] * 64
    assert all((np.array_equal(data[t]['domain'], data[41]['domain']) and np.array_equal(data[t]['document_ids'], data[41]['document_ids']) for t in TIMES))
    assert list(data[41]['domain']) == [d for d in DOMAINS for _ in range(64)]
    summary = []
    split = []
    bootrows = []
    cross = []
    flow = []
    channels = []
    sentinel = []
    cache_summary = []
    operators = {}
    boots = {}
    spectra = {}
    cache_raw = {}
    for t in TIMES:
        for l in LAYERS:
            for p in PROBES:
                key = (t, l, p)
                prefix = f'L{l}_{p}_J'
                g = data[t][prefix + '_physical']
                operators[key] = g
                sp = spectrum(g, basis[l], hi[l])
                spectra[key] = sp
                summary.append(dict(tokens_B=t, layer=l, probe=p, **{k: v for k, v in sp.items() if k != 'eigenvectors'}))
                st = stability(data[t][prefix + '_half0_physical'], data[t][prefix + '_half1_physical'])
                split.append(dict(tokens_B=t, layer=l, probe=p, **st))
                raw = data[t][prefix][indices].mean(axis=(1, 2))
                bs = hi[l] @ raw @ hi[l]
                boots[key] = bs
                A = np.trace(bs, axis1=-2, axis2=-1)
                dd = deff(bs)
                for metric, val in [('trace_A', A), ('d_eff', dd)]:
                    bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=metric, **interval(val)))
                eig = np.linalg.eigvalsh(bs)[:, ::-1]
                eig = np.maximum(eig, 0)
                norm = eig / eig.sum(1, keepdims=True)
                entropy = np.exp(-np.sum(np.where(norm > 0, norm * np.log(np.maximum(norm, np.finfo(float).tiny)), 0), axis=1))
                bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric='d_ent', **interval(entropy)))
                for j in range(6):
                    bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=f'normalized_eigenvalue_{j + 1}', **interval(norm[:, j])))
                for threshold in (0.5, 0.8, 0.9):
                    dims = np.argmax(np.cumsum(norm, axis=1) >= threshold, axis=1) + 1
                    bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=f'd{int(threshold * 100)}', **interval(dims)))
                for j, val in enumerate(sp['sentinel_mode1_eigenloadings']):
                    sentinel.append(dict(tokens_B=t, layer=l, probe=p, eigenvector_rank=j + 1, sentinel_mode1_loading=float(val)))
                gg = {c: data[t][f'L{l}_{p}_{c}_physical'] for c in ('J', 'R', 'G', 'RG')}
                residual = gg['J'] - gg['R'] - gg['G'] - gg['RG'] - gg['RG'].T
                assert np.linalg.norm(residual) <= 1e-10 + 1e-05 * np.linalg.norm(gg['J'])
                channels.append(dict(tokens_B=t, layer=l, probe=p, J_trace=float(np.trace(gg['J'])), R_trace=float(np.trace(gg['R'])), G_trace=float(np.trace(gg['G'])), cross_trace_RG_plus_GR=float(2 * np.trace(gg['RG'])), R_trace_fraction=float(np.trace(gg['R']) / np.trace(gg['J'])), G_trace_fraction=float(np.trace(gg['G']) / np.trace(gg['J'])), cross_trace_fraction=float(2 * np.trace(gg['RG']) / np.trace(gg['J'])), component_closure_frobenius_absolute=float(np.linalg.norm(residual)), R_physical_operator=gg['R'], G_physical_operator=gg['G'], RG_physical_operator=gg['RG']))
                for component in ('R', 'G', 'RG'):
                    component_bs = hi[l] @ data[t][f'L{l}_{p}_{component}'][indices].mean(axis=(1, 2)) @ hi[l]
                    trace_bs = np.trace(component_bs, axis1=-2, axis2=-1) * (2 if component == 'RG' else 1)
                    bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=f'{component}_joint_trace_fraction', **interval(trace_bs / A)))
            g = data[t][f'L{l}_g']
            gR = data[t][f'L{l}_g_router']
            gG = data[t][f'L{l}_g_gain']
            a2 = data[t][f'L{l}_a_norm_sq']
            uip = data[t][f'L{l}_u_inner_products']
            u2 = np.diagonal(uip, axis1=-2, axis2=-1)
            den = a2[:, None] * u2
            cos2 = np.divide(g * g, den, out=np.full_like(g, np.nan), where=den > 0)
            assert np.all(g[den == 0] ** 2 <= 1e-10)
            assert np.all(cos2[den > 0] <= 1 + 1e-05)
            cache_raw[f'T{t}_L{l}_a_norm_sq'] = a2
            cache_raw[f'T{t}_L{l}_u_inner_products'] = uip
            cache_raw[f'T{t}_L{l}_g'] = g
            cache_raw[f'T{t}_L{l}_cos_sq_theta'] = cos2
            cache_raw[f'T{t}_L{l}_cos_sq_defined'] = den > 0
            for j in range(6):
                cache_summary.append(dict(tokens_B=t, layer=l, mode=j + 1, documents=448, a_norm_sq_mean=float(a2.mean()), u_norm_sq_mean=float(u2[:, j].mean()), g_squared_mean=float(np.mean(g[:, j] ** 2)), alignment_cos_sq_mean=float(np.nanmean(cos2[:, j])) if np.any(den[:, j] > 0) else None, alignment_undefined_zero_norm_count=int(np.sum(den[:, j] == 0)), susceptibility_reconstruction_abs_error=float(abs(np.sum(den[:, j][den[:, j] > 0] * cos2[:, j][den[:, j] > 0]) / 448 - np.mean(g[:, j] ** 2)))))
    print('T4 spectra, halves, bootstrap operators and component cache PASS', flush=True)
    for t in TIMES:
        for l in LAYERS:
            a, b = (operators[t, l, 'local'], operators[t, l, 'objective'])
            dist = cf(boots[t, l, 'local'], boots[t, l, 'objective'])
            ci = interval(dist)
            decision = 'COMMON_GEOMETRY_SUPPORTED' if ci['bootstrap_q025'] > 0.9 else 'PROBE_GEOMETRY_DISTINCT' if ci['bootstrap_q975'] < 0.9 else 'CROSS_PROBE_GEOMETRY_UNRESOLVED'
            for k in (1, 2, 3):
                cross.append(dict(tokens_B=t, layer=l, local_objective_shape_cosine=float(cf(a, b)), **ci, decision=decision, **geometry(a, b, k)))
                angles_b, q_b = bootstrap_geometry(boots[t, l, 'local'], boots[t, l, 'objective'], k)
                bootrows.append(dict(tokens_B=t, layer=l, probe='local_vs_objective', metric=f'projector_overlap_Q{k}', **interval(q_b)))
                for j in range(k):
                    bootrows.append(dict(tokens_B=t, layer=l, probe='local_vs_objective', metric=f'principal_angle_k{k}_rank{j + 1}_degrees', **interval(angles_b[:, j])))
            for p in PROBES:
                a, b = (operators[t, l, p], operators[41, l, p])
                ratio = float(np.trace(a) / np.trace(b))
                ratio_bs = np.trace(boots[t, l, p], axis1=-2, axis2=-1) / np.trace(boots[41, l, p], axis1=-2, axis2=-1)
                bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric='A_t_over_A_41', **interval(ratio_bs)))
                bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric='shape_cosine_vs41', **interval(cf(boots[t, l, p], boots[41, l, p]))))
                for k in (1, 2, 3):
                    flow.append(dict(tokens_B=t, layer=l, probe=p, A_t_over_A_41=ratio, d_eff=spectra[t, l, p]['d_eff'], d90=spectra[t, l, p]['d90'], shape_cosine_vs41=float(cf(a, b)), **geometry(a, b, k)))
                    angles_b, q_b = bootstrap_geometry(boots[t, l, p], boots[41, l, p], k)
                    bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=f'projector_overlap_Q{k}_vs41', **interval(q_b)))
                    for j in range(k):
                        bootrows.append(dict(tokens_B=t, layer=l, probe=p, metric=f'principal_angle_vs41_k{k}_rank{j + 1}_degrees', **interval(angles_b[:, j])))
    layer_ordering = []
    layer_shape = []
    for p in PROBES:
        prob = float(np.mean(deff(boots[41, 15, p]) < deff(boots[41, 7, p])))
        layer_ordering.append({'probe': p, 'tokens_B': 41, 'P_d_eff_L15_less_than_L7': prob, 'supported': prob >= 0.975, 'threshold': 0.975, 'interpretation': 'supported' if prob >= 0.975 else 'unresolved or non-transferring'})
        for t in TIMES:
            a = spectra[t, 7, p]['normalized_spectrum']
            b = spectra[t, 15, p]['normalized_spectrum']
            layer_shape.append({'tokens_B': t, 'probe': p, 'spectral_shape_cosine_L7_L15': float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 'comparison_space': 'ordered normalized spectra only; layer control spaces distinct; no cross-layer principal angles'})
    write('T4_LAYER_ORDERING_AND_SPECTRAL_COMPARISON.json', dict(META, checkpoints=list(CP.values()), baseline_layer_ordering=layer_ordering, layer_spectral_comparisons=layer_shape, bootstrap_seed=BOOT_SEED, bootstrap_replicates=BOOT_N, paired=True))
    write('T4_AU_CACHE_CAPABILITIES.json', dict(META, status='PASS', checkpoints=list(CP.values()), per_document_checkpoint_layer_mode_available=True, canonical_joint_definition=META['canonical_joint_definition'], local_component_recovery_forward_count=214, technical_certification_branch='CERT-B', available_fields=['a_norm_sq', 'u_norm_sq', 'g=a_dot_u', 'all_six_by_six_u_inner_products', 'router_gain_inner_products', 'router_gain_objective_contractions'], cache_only_auxiliary=True, new_probe_count=0, additional_model_pass_count=0, zero_norm_alignment_undefined_preserved=True, consistency_tolerance={'absolute': 1e-10, 'relative': 1e-05}, all_document_cache_audits=audit_values, summary=cache_summary, max_v160_Gram_vs_cache_relative_error=max((r['frobenius_relative'] or 0 for r in audit_values))))
    cache_raw['metadata_json'] = np.array(json.dumps(dict(META, status='PASS', checkpoints=list(CP.values()), alignment_NaN_means_zero_norm_undefined=True)))
    with (RUN / 'T4_AU_SUFFICIENT_STATISTICS.npz').open('xb') as f:
        np.savez_compressed(f, **cache_raw)
    for name, rr in [('T4_TOMOGRAPHY_SUMMARY.csv', summary), ('T4_SPLIT_HALF_STABILITY.csv', split), ('T4_BOOTSTRAP_SUMMARY.csv', bootrows), ('T4_CROSS_PROBE_GEOMETRY.csv', cross), ('T4_CHANNEL_DECOMPOSITION.csv', channels), ('T4_TEMPORAL_SPECTRAL_FLOW.csv', flow), ('T4_SENTINEL_LOADINGS.csv', sentinel), ('T4_AU_CACHE_SUMMARY.csv', cache_summary)]:
        csvout(name, rr)
    with (RUN / 'T4_BOOTSTRAP_DRAWS.npz').open('xb') as f:
        np.savez_compressed(f, indices=indices, seed=BOOT_SEED, replicates=BOOT_N, metadata_json=np.array(json.dumps(dict(META, bootstrap_generator='numpy.default_rng PCG64', paired_across='layers/probes/checkpoints', domain_draws_each=64))))
    csvout('T4_LAYER_SPECTRAL_SHAPE.csv', layer_shape)
    artifacts = [p for p in sorted(RUN.glob('T4_*')) if p.is_file()]
    write('T4_STAGE_SEAL.json', dict(META, status='SEALED', T4_STATUS='PASS', artifacts={p.name: ident(p) for p in artifacts}, bootstrap_replicates=BOOT_N, bootstrap_seed=BOOT_SEED, checkpoints=list(CP.values())))
    with (RUN / 'T4_STAGE_SEAL.sha256').open('x') as f:
        f.write(sha(RUN / 'T4_STAGE_SEAL.json') + '  T4_STAGE_SEAL.json\n')
    print('T4 SEALED', flush=True)
    report = ['NEW_TRAINING_STEP_COUNT = 0', 'RESERVE_ACCESS_COUNT = 0', 'NEW_NULL_BANK_COUNT = 0', 'BASIS_REFIT_COUNT = 0', 'OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT = 0', 'T0_STATUS = PASS', 'T1_STATUS = PASS', 'T2_STATUS = PASS', 'T3_STATUS = PASS', 'T4_STATUS = PASS', '', '# P2D restricted-6D tomography execution report', '', 'All frozen T0–T4 stages completed. Stage PASS means execution and provenance passed; negative and unresolved scientific decisions remain unchanged. No training, reserve access, new null bank, basis refit or outcome-dependent protocol change occurred.', '', f"Protocol SHA-256: `{META['protocol']['sha256']}`", f"Corrected basis SHA-256: `{META['basis']['sha256']}`", f"Bank-T seal SHA-256: `{META['Bank_T_seal_sha256']}`", '', 'Bank T contains exactly 448 documents, 64 per domain, in deterministic acceptance order, with a 32/32 half split. Complete historical overlap replay and safe non-reserve Q3 source mapping passed; source-ID and token-hash overlap counts are both zero.', '', 'CERT-B authorized composed joint responses without relaxing the old direct-joint guard: uJ is the FP32 component sum and gJ is the FP64 sum of component contractions. The direct joint remains a non-blocking numerical audit. Valid old objective/adjoint fields were reused for 216 documents; missing raw local component vectors were recovered with 214 local-only base forwards and two diagnostic vector reuses. The first cache-replay attempt was preserved; an explicit FP64 input cast restored the exact historical contraction convention without changing any tolerance. Local tomography uses the unchanged certified v160 FP32 fixed-support Gram and FP64 reductions. Objective susceptibility uses document-mean next-token NLL, the downstream adjoint and the inherited exact FP32 tangent. All spectra use the frozen nonorthogonal control metric H; no controls were refit or whitened. BF16 MPS eager forward, exact checkpoint commits, batch size 1 and disabled CPU fallback match the historical certified backend.', '', 'The complete 448-document 41B stage was sealed before the 5033B mode-1 scalar stage began. The T1 terminal cache contains only mode-1 scalar endpoints. The scalar prediction was hashed and sealed before the full 5033B checkpoint stage began. T2 scores the unchanged prediction and thresholds; intermediate checkpoints were evaluated only after the T2 seal.', '', '| Layer | Probe | Scale error | Shape cosine | Frozen decision |', '|---|---|---:|---:|---|']
    for r in read(RUN / 'T2_PREDICTION_SCORE.json')['scores']:
        report.append(f"| {r['layer']} | {r['probe']} | {r['epsilon_scale']:.6g} | {r['shape_cosine']:.6g} | {r['decision']} |")
    report += ['', 'Baseline fresh-bank layer-concentration test:']
    for r in layer_ordering:
        report.append(f"- {r['probe']}: P(d_eff L15 < L7) = {r['P_d_eff_L15_less_than_L7']:.6g}; {r['interpretation']}, with the frozen 0.975 threshold.")
    counts = Counter((r['status'] for r in split))
    report += ['', f'Split-half stability: {dict(counts)} across the 20 layer/checkpoint/probe operators.', '', 'All cross-probe decisions, normalized spectra, effective dimensions, fixed k=1/2/3 principal angles/projector overlaps, temporal scale ratios, sentinel loadings, router/gain/cross decomposition and auxiliary a/u sufficient-statistic audits are saved in the T4 artifacts. Cross-layer comparisons use ordered spectral shapes only.', '', 'Bootstrap: 2000 balanced-domain paired replicates, seed 2026100701, PCG64; identical document draws across all layers/probes/checkpoints. Raw draws are persisted. The auxiliary cache adds no model passes or scientific probe; zero-norm alignment remains undefined.', '', f"Dependency recovery: {recovered['SHARDS_REUSED_WITHOUT_DOWNLOAD']} shards reused in place and {recovered['SHARDS_NEWLY_DOWNLOADED']} recovered into a content-addressed cache. The previous blocked run is unchanged.", f"New complete shard payload bytes: {recovered['COMPLETE_SHARD_PAYLOAD_BYTES_NEWLY_DOWNLOADED']}; retained incomplete payload bytes: {recovered['INCOMPLETE_CACHE_PAYLOAD_BYTES_RETAINED']}. These are payload accounting totals, not exact network-interface transfer counters.", '', 'Final provenance independently rehashed all seven source shards, every frozen checkpoint weight and all opened-source authorities; all stage/document hashes passed. Full paths, source hashes, checkpoint identities and environment details are embedded in JSON/NPZ outputs and bound to every CSV through the hashed provenance audit.', '', 'Scientific scope is restricted to the declared six-control family and post-41B checkpoints. These measurements do not establish a microscopic pre-41B formation time.', '']
    with (RUN / 'P2D_TOMOGRAPHY_EXECUTION_REPORT.md').open('x') as f:
        f.write('\n'.join(report))
    write('P2D_EXECUTION_STATE.json', dict(META, NEW_TRAINING_STEP_COUNT=0, RESERVE_ACCESS_COUNT=0, NEW_NULL_BANK_COUNT=0, BASIS_REFIT_COUNT=0, OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT=0, **{n + '_STATUS': 'PASS' for n in ('T0', 'T1', 'T2', 'T3', 'T4')}))
    print('FINAL REPORT WRITTEN; ready for delivery hash finalizer', flush=True)
if __name__ == '__main__':
    main()
