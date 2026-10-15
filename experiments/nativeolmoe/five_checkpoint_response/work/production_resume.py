# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""CERT-B canonical composition; preserve valid cached objective/adjoint fields."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import gc, sys, traceback
ATOL = 1e-10
RTOL = 1e-05
E = None

def norm_check(a, b):
    return np.linalg.norm(a - b) <= ATOL + RTOL * np.linalg.norm(b)

def inner(u, v):
    return np.einsum('ith,jth->ij', u, v, dtype=np.float64, optimize=False)

def contract(u, a):
    return np.einsum('ith,th->i', u, a, dtype=np.float64, optimize=False)

def verify_cert():
    p = RUN / 'T1_NUMERICAL_IDENTITY_SEAL.json'
    assert sha(p) == (RUN / 'T1_NUMERICAL_IDENTITY_SEAL.sha256').read_text().split()[0]
    s = read(p)
    for r in s['artifacts'].values():
        assert sha(r['path']) == r['sha256']
    c = read(RUN / 'T1_NUMERICAL_IDENTITY_CERTIFICATION.json')
    assert c['certification_branch'] == 'CERT-B'
    assert c['T1_TECHNICAL_CERTIFICATION'] == 'DIRECT_JOINT_NUMERICAL_PATH_ONLY'
    return c

def audit_old_components():
    prior = read(PRIOR / 'T1_PARTIAL_CACHE_AUDIT.json')
    rr = []
    assert len(prior['files']) == 216
    for index, ref in enumerate(prior['files']):
        assert sha(ref['path']) == ref['sha256']
        with np.load(ref['path'], allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json']))
            row = E.ROWS[index]
            assert meta['source_id'] == row['source_id'] and meta['token_sha256'] == row['token_sha256']
            assert meta['Bank_T_seal_sha256'] == E.META['Bank_T_seal_sha256'] and meta['basis']['sha256'] == E.META['basis']['sha256']
            assert meta['checkpoint']['revision_commit_sha'] == E.CP[41]['revision_commit_sha']
            for src in meta['implementation_sources']:
                assert sha(src['path']) == src['sha256']
            for l in (7, 15):
                for k, shape in [('g_router', (6,)), ('g_gain', (6,)), ('u_router_inner_products', (6, 6)), ('u_gain_inner_products', (6, 6)), ('u_router_gain_inner_products', (6, 6)), ('a_norm_sq', ())]:
                    assert z[f'L{l}_' + k].shape == shape and np.isfinite(z[f'L{l}_' + k]).all()
                gr = z[f'L{l}_g_router']
                gg = z[f'L{l}_g_gain']
                for c, b in [('R', np.outer(gr, gr)), ('G', np.outer(gg, gg)), ('RG', np.outer(gr, gg))]:
                    assert np.array_equal(z[f'L{l}_objective_{c}'], b)
                for c in ('R', 'G', 'RG'):
                    assert z[f'L{l}_local_{c}'].shape == (6, 6) and np.isfinite(z[f'L{l}_local_{c}']).all()
            rr.append({'document_index': index, 'old_cache': ref, 'source_hash_bindings_PASS': True, 'component_sufficient_statistics_PASS': True, 'valid_objective_component_fields_reused': True, 'valid_a_norm_sq_and_baseline_NLL_reused': True, 'missing_raw_response_fields': ['L7_u_R', 'L7_u_G', 'L15_u_R', 'L15_u_G'], 'recovery_plan': 'docs0/215 reuse certified vectors; remaining 214 docs local-only forward/JVP recovery; no adjoint rerun'})
    write(RUN / 'T1_OLD_CACHE_FIELD_AUDIT.json', dict(E.META, status='PASS_FOR_VALID_FIELDS', documents=216, rows=rr, raw_response_fields_not_falsely_claimed_present=True, canonical_joint_fields_regenerated_before_aggregation=True))
    return prior['files']

def component_payload(index, cp, stage, arrays, meta):
    folder = RUN / 'component_response_cache' / stage / str(cp)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f'{index:03d}.npz'
    add_npz(p, dict(arrays, metadata_json=np.array(json.dumps(meta))), True)
    return ident(p)

def canonicalize_old(model, row, index, ref):
    torch = E.torch
    with np.load(ref['path'], allow_pickle=False) as z:
        out = {k: z[k].copy() for k in z.files if k != 'metadata_json'}
    raw = {}
    audit = []
    pending = []
    if index in (0, 215):
        diag = np.load(RUN / 'T1_NUMERICAL_IDENTITY_DIAGNOSTICS.npz', allow_pickle=False)
        caps = None
    else:
        ids = torch.tensor(row['token_ids'], device='mps').unsqueeze(0)
        caps, kw, last = E.capture(model, ids)
        with torch.no_grad():
            base = E.logits(model, last)
        ell = float(E.loss_from_logits(base, ids))
        pending.append(('OLD_BASELINE_NLL_REPLAY', ell == float(out['baseline_NLL'])))
    for l in (7, 15):
        if caps is None:
            prefix = f'D{index}_REPEAT1_L{l}'
            ur = diag[prefix + '_u_R_engine']
            ug = diag[prefix + '_u_G_engine']
            for suffix, oldname in [('g_R', 'g_router'), ('g_G', 'g_gain')]:
                pending.append(('CERT_COMPONENT_CONTRACTION_REPLAY', np.array_equal(diag[prefix + '_' + suffix], out[f'L{l}_{oldname}'])))
            pending.append(('CERT_ADJOINT_NORM_REPLAY', float(np.sum(diag[prefix + '_shared_adjoint'] ** 2)) == float(out[f'L{l}_a_norm_sq'])))
        else:
            local = E.make_local(model, caps, l)
            v = E.BASIS[l]
            vr = v.copy()
            vr[:, :64] = 0
            vg = v.copy()
            vg[:, 64:] = 0
            ur = np.stack([E.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vr])
            ug = np.stack([E.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vg])
            del local
        raw[f'L{l}_u_R'] = ur
        raw[f'L{l}_u_G'] = ug
        for suffix, got in [('router', inner(ur, ur)), ('gain', inner(ug, ug)), ('router_gain', inner(ur, ug))]:
            expected = out[f'L{l}_u_{suffix}_inner_products']
            pending.append(('RECOVERED_COMPONENT_INNER_PRODUCTS', norm_check(got, expected)))
        gr = out[f'L{l}_g_router']
        gg = out[f'L{l}_g_gain']
        g = gr + gg
        u = ur + ug
        uip = inner(u, u)
        out[f'L{l}_g_direct_original'] = out[f'L{l}_g']
        out[f'L{l}_u_inner_products_direct_original'] = out[f'L{l}_u_inner_products']
        out[f'L{l}_g_direct_audit'] = out[f'L{l}_g']
        out[f'L{l}_u_direct_inner_products_audit'] = out[f'L{l}_u_inner_products']
        out[f'L{l}_g'] = g
        out[f'L{l}_u_inner_products'] = uip
        out[f'L{l}_objective_J'] = np.outer(g, g)
        bj = sum((out[f'L{l}_local_{c}'] for c in ('R', 'G', 'RG'))) + out[f'L{l}_local_RG'].T
        out[f'L{l}_local_J'] = bj
        diff = float(np.linalg.norm(bj - uip / (256 * 2048)))
        den = float(np.linalg.norm(bj))
        pending.append(('V160_GRAM_CANONICAL_U_CONSISTENCY', diff <= ATOL + RTOL * den))
        audit.append({'layer': l, 'v160_gram_vs_cached_u_gram_frobenius_abs': diff, 'frobenius_relative': diff / den if den else None, 'objective_component_max_abs': 0.0, 'local_component_max_abs': float(np.max(np.abs(bj - out[f'L{l}_local_R'] - out[f'L{l}_local_G'] - out[f'L{l}_local_RG'] - out[f'L{l}_local_RG'].T))), 'canonical_joint_regenerated': True, 'old_valid_component_fields_preserved': True})
    metadata = dict(E.META, stage='T1_41B', tokens_B=41, checkpoint=E.CP[41], document_id=row['document_id'], source_id=row['source_id'], source_shard=row['source_shard'], source_physical_path=row['source_physical_path'], source_sha256=row['source_sha256'], token_sha256=row['token_sha256'], scalar_only=False, cache_audit=audit, canonical_joint_definition='uJ=FP32(uR+uG); gJ=FP64(gR)+FP64(gG)', inherited_cache=ref, valid_expensive_fields_reused=True, old_adjoint_not_rerun=True, certified_diagnostic_vectors_reused=index in (0, 215), canonicalization_checks=[{'check': n, 'PASS': bool(ok)} for n, ok in pending])
    metadata['raw_component_response_artifact'] = component_payload(index, 41, 'T1_41B', raw, metadata)
    out['metadata_json'] = np.array(json.dumps(metadata))
    if caps is None:
        diag.close()
    else:
        del ids, caps, kw, last, base
    gc.collect()
    torch.mps.empty_cache()
    return (out, pending)

def measure(model, row, index, cp, stage, scalar_only=False):
    torch = E.torch
    ids = torch.tensor(row['token_ids'], device='mps').unsqueeze(0)
    caps, kw, last = E.capture(model, ids)
    with torch.no_grad():
        base = E.logits(model, last)
    bh = hashlib.sha256(base.detach().cpu().double().numpy().tobytes()).hexdigest()
    ell0 = float(E.loss_from_logits(base, ids))
    out = {}
    audit = []
    pending = []
    raw = {}
    direct_diagnostics = []
    for l in (7, 15):
        local = E.make_local(model, caps, l)
        a, ell = E.adjoint(model, caps[l]['layer_output'], l, kw, ids, bh)
        pending.append(('SUFFIX_NLL_EXACT', ell == ell0))
        v = E.BASIS[l][:1 if scalar_only else 6]
        vr = v.copy()
        vr[:, :64] = 0
        vg = v.copy()
        vg[:, 64:] = 0
        ur = np.stack([E.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vr])
        ug = np.stack([E.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in vg])
        ud = np.stack([E.continuous_fixed_support_jvp_fp32(local, x).detach().cpu().numpy() for x in v])
        u = ur + ug
        gr = contract(ur, a)
        gg = contract(ug, a)
        g = gr + gg
        gd = contract(ud, a)
        uip = inner(u, u)
        raw[f'L{l}_u_R'] = ur
        raw[f'L{l}_u_G'] = ug
        old_limit = ATOL + RTOL * np.maximum(np.abs(gd), np.abs(gr) + np.abs(gg))
        direct_diagnostics.append({'layer': l, 'direct_g': gd.tolist(), 'canonical_g': g.tolist(), 'absolute_residual': np.abs(gd - g).tolist(), 'old_guard_limits': old_limit.tolist(), 'old_guard_pass': (np.abs(gd - g) <= old_limit).tolist(), 'direct_u_vs_composed_relative': float(np.linalg.norm(ud.astype(np.float64) - u.astype(np.float64)) / max(np.linalg.norm(u.astype(np.float64)), 1e-30)), 'blocking': False, 'authority': 'CERT-B'})
        out[f'L{l}_g'] = g
        if scalar_only:
            out[f'L{l}_u_inner_products'] = uip
            out[f'L{l}_local_mode1_power'] = np.array(uip[0, 0] / (256 * 2048))
            out[f'L{l}_objective_mode1_power'] = np.array(g[0] ** 2)
        else:
            gram = E.exact_soft_gram(local)
            br = vr @ gram @ vr.T
            bg = vg @ gram @ vg.T
            brg = vr @ gram @ vg.T
            bj = br + bg + brg + brg.T
            diff = float(np.linalg.norm(bj - uip / (256 * 2048)))
            den = float(np.linalg.norm(bj))
            pending.append(('V160_GRAM_CANONICAL_U_CONSISTENCY', diff <= ATOL + RTOL * den))
            out[f'L{l}_u_inner_products'] = uip
            out[f'L{l}_a_norm_sq'] = np.array(np.sum(a * a, dtype=np.float64))
            out[f'L{l}_g_router'] = gr
            out[f'L{l}_g_gain'] = gg
            out[f'L{l}_g_direct_audit'] = gd
            out[f'L{l}_u_direct_inner_products_audit'] = inner(ud, ud)
            for probe, blocks in [('local', (bj, br, bg, brg)), ('objective', (np.outer(g, g), np.outer(gr, gr), np.outer(gg, gg), np.outer(gr, gg)))]:
                for name, b in zip(('J', 'R', 'G', 'RG'), blocks):
                    out[f'L{l}_{probe}_{name}'] = b
            for name, b in [('router', inner(ur, ur)), ('gain', inner(ug, ug)), ('router_gain', inner(ur, ug))]:
                out[f'L{l}_u_{name}_inner_products'] = b
            audit.append({'layer': l, 'v160_gram_vs_cached_u_gram_frobenius_abs': diff, 'frobenius_relative': diff / den if den else None, 'objective_component_max_abs': 0.0, 'local_component_max_abs': float(np.max(np.abs(bj - br - bg - brg - brg.T))), 'direct_objective_guard_failures_nonblocking': int(np.sum(np.abs(gd - g) > old_limit))})
        del local, a, ur, ug, ud, u
        torch.mps.empty_cache()
    if not scalar_only:
        out['baseline_NLL'] = np.array(ell0)
    pending.append(('FINITE_ENDPOINTS', all((np.isfinite(x).all() for x in out.values()))))
    meta = dict(E.META, stage=stage, tokens_B=cp, checkpoint=E.CP[cp], document_id=row['document_id'], source_id=row['source_id'], source_shard=row['source_shard'], source_physical_path=row['source_physical_path'], source_sha256=row['source_sha256'], token_sha256=row['token_sha256'], scalar_only=scalar_only, cache_audit=audit, canonical_joint_definition='uJ=FP32(uR+uG); gJ=FP64(gR)+FP64(gG)', direct_joint_nonblocking_audit=direct_diagnostics, checks=[{'check': n, 'PASS': bool(ok)} for n, ok in pending])
    if cp == 41 or not all((ok for _, ok in pending)):
        meta['raw_component_response_artifact'] = component_payload(index, cp, stage, raw, meta)
    out['metadata_json'] = np.array(json.dumps(meta))
    del caps, kw, last, base, ids, raw
    gc.collect()
    torch.mps.empty_cache()
    return (out, pending)

def gather(folder, cp, stage, scalar_only):
    files = sorted(folder.glob('*.npz'))
    assert len(files) == 448
    data = {}
    index = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            for k in z.files:
                if k == 'metadata_json' or 'direct_original' in k:
                    continue
                data.setdefault(k, []).append(z[k])
        index.append(ident(p))
    data = {k: np.stack(v) for k, v in data.items()}
    assert all((len(v) == 448 for v in data.values()))
    data.update(document_ids=np.array([r['document_id'] for r in E.ROWS]), domain=np.array([r['domain'] for r in E.ROWS]), half=np.array([r['half'] for r in E.ROWS]))
    write(RUN / (stage + '_' + str(cp) + '_CACHE_INDEX.json'), dict(E.META, stage=stage, tokens_B=cp, checkpoint=E.CP[cp], scalar_only=scalar_only, files=index))
    return data

def run_checkpoint(cp, stage, scalar_only=False):
    torch = E.torch
    if scalar_only:
        s = E.sealed('T1_41B_COMPLETE_SEAL')
        assert s['documents'] == 448
    old = audit_old_components() if cp == 41 else None
    folder = RUN / 'document_cache' / stage / str(cp)
    folder.mkdir(parents=True, exist_ok=True)
    E.event('MODEL_LOAD_BEGIN', tokens_B=cp, stage=stage, revision=E.CP[cp]['revision_commit_sha'], allowed_modes=1 if scalar_only else 6)
    model = E.load_checkpoint(E.CP[cp], E.CACHE)
    E.event('MODEL_LOADED', tokens_B=cp, stage=stage, parameter_gradients_disabled=True)
    try:
        for i, row in enumerate(E.ROWS):
            if old is not None and i < 216:
                out, pending = canonicalize_old(model, row, i, old[i])
                action = 'REUSE_VALID_FIELDS_RECOVER_MISSING_LOCAL_RESPONSES'
            else:
                out, pending = measure(model, row, i, cp, stage, scalar_only)
                action = 'NEW_CERTIFIED_MEASUREMENT'
            p = folder / f'{i:03d}.npz'
            E.save(p, **out)
            bad = [n for n, ok in pending if not ok]
            if bad:
                raise RuntimeError(('PERSISTED_NUMERICAL_GUARD_FAILURE', cp, i, bad, str(p)))
            E.event('DOCUMENT_DONE', stage=stage, tokens_B=cp, index=i, cache=ident(p), action=action)
            if i % 8 == 0 or i == 447:
                print(stage, cp, 'B', i + 1, '/448', action, flush=True)
    finally:
        del model
        gc.collect()
        torch.mps.empty_cache()
        E.event('MODEL_UNLOADED', tokens_B=cp, stage=stage)
    return gather(folder, cp, stage, scalar_only)

def t1():
    a = E.aggregate(run_checkpoint(41, 'T1_41B'))
    E.scientific_npz('T1_41B_TOMOGRAPHY.npz', a, 'T1', [41])
    E.seal('T1_41B_COMPLETE_SEAL', [RUN / 'T1_41B_TOMOGRAPHY.npz', RUN / 'T1_41B_41_CACHE_INDEX.json', RUN / 'T1_OLD_CACHE_FIELD_AUDIT.json'], documents=448, tokens_B=41)
    print('COMPLETE 448-DOCUMENT 41B SEAL EXISTS; terminal mode1 now authorized', flush=True)
    b = run_checkpoint(5033, 'T1_5033_MODE1', scalar_only=True)
    scalars = []
    prediction = []
    for l in (7, 15):
        for p in ('local', 'objective'):
            vals = b[f'L{l}_{p}_mode1_power']
            q5033 = float(E.balanced(vals))
            q41 = float(a[f'L{l}_{p}_J_mean_B'][0, 0])
            assert q41 > 0 and q5033 >= 0
            c = q5033 / q41
            scalars.append({'layer': l, 'probe': p, 'Q5033_mode1': q5033, 'per_document_mode1_power': vals.tolist()})
            prediction.append({'layer': l, 'probe': p, 'Q41_mode1': q41, 'Q5033_mode1': q5033, 'scalar_rescaling_c': c, 'predicted_physical_operator': (c * a[f'L{l}_{p}_J_physical']).tolist(), 'decision_thresholds': {'epsilon_scale_max': 0.2, 'shape_cosine_min': 0.95, 'split_shape_cosine_min': 0.9, 'split_d_eff_relative_difference_max': 0.25}})
    write(RUN / 'T1_5033_MODE1_SCALARS.json', dict(E.META, status='COMPLETE', checkpoint=E.CP[5033], permitted_terminal_modes=[1], scalars=scalars))
    write(RUN / 'T1_SCALAR_PREDICTIONS.json', dict(E.META, status='FROZEN', checkpoints=[E.CP[41], E.CP[5033]], prediction_rule='Gtilde_pred5033 = (Q5033(v1)/Q41(v1))*Gtilde41', predictions=prediction, full_5033_6D_computed=False, full_5033_6D_read=False))
    E.seal('T1_PREDICTION_SEAL', [RUN / n for n in ('T1_41B_TOMOGRAPHY.npz', 'T1_5033_MODE1_SCALARS.json', 'T1_SCALAR_PREDICTIONS.json', 'T1_41B_41_CACHE_INDEX.json', 'T1_5033_MODE1_5033_CACHE_INDEX.json')], T1_STATUS='PASS', prediction_sha256=sha(RUN / 'T1_SCALAR_PREDICTIONS.json'), held_out_information_barrier_preserved=True)
    print('T1 PREDICTION SEALED', flush=True)

def main(stage):
    global E
    cert = verify_cert()
    E = engine(stage)
    E.META.update(technical_certification=ident(RUN / 'T1_NUMERICAL_IDENTITY_CERTIFICATION.json'), technical_certification_seal=ident(RUN / 'T1_NUMERICAL_IDENTITY_SEAL.json'), canonical_joint_definition='uJ=FP32(uR+uG); gJ=FP64(gR)+FP64(gG); unchanged direct-JVP guard retained as non-blocking CERT-B audit', scientific_protocol_change=False)
    E.META['implementation_sources'] += [ident(Path(__file__))]
    E.run_checkpoint = run_checkpoint
    E.measure = measure
    {'T1': t1, 'T2': E.t2, 'T3': E.t3}[stage]()
if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception as exc:
        if E:
            E.event('EXECUTION_BLOCKED', stage=sys.argv[1], exact_error=str(exc), traceback=traceback.format_exc())
        raise
