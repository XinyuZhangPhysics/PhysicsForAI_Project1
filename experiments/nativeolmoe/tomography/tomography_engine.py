# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen staged tomography. Each entry verifies its predecessor seal before imports/loading."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import gc, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
RUN = Path(__file__).resolve().parents[1]
PKG = RUN / 'control_package/Project1_P2D_Codex_Control_Package_v1p1'
SRC = Path(_p1_path('path_97e245237253'))
H = Path(_p1_path('path_65e87eb57756'))
CACHE = H / '2026-10-04/jie/work/hf-cache'
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
LAYERS = (7, 15)
NUMERICAL_ATOL = 1e-10
NUMERICAL_RTOL = 1e-05

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

def write(p, o):
    with Path(p).open('x') as f:
        json.dump(o, f, indent=2, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
        f.write('\n')

def event(name, **kw):
    with (RUN / 'logs/TOMOGRAPHY_EVENTS.jsonl').open('a') as f:
        f.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': name, **kw}) + '\n')
        f.flush()

def sealed(name):
    p = RUN / (name + '.json')
    assert sha(p) == (RUN / (name + '.sha256')).read_text().split()[0], ('STAGE_SEAL_HASH_MISMATCH', name)
    d = read(p)
    for r in d.get('artifacts', {}).values():
        assert sha(r['path']) == r['sha256'], ('SEALED_ARTIFACT_HASH_MISMATCH', r['path'])
    return d

def save(p, **a):
    assert not Path(p).exists(), ('IMMUTABLE_OUTPUT_EXISTS', str(p))
    with Path(p).open('xb') as f:
        np.savez_compressed(f, **a)

def seal(name, paths, **kw):
    p = RUN / (name + '.json')
    write(p, dict(META, status='SEALED', artifacts={Path(r).name: ident(r) for r in paths}, **kw))
    with (RUN / (name + '.sha256')).open('x') as f:
        f.write(sha(p) + '  ' + p.name + '\n')
    event(name + '_SEALED', sha256=sha(p))
    return sha(p)

def cf(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / den) if den > 0 else None

def deff(a):
    den = np.sum(a * a)
    return float(np.trace(a) ** 2 / den) if den > 0 else None

def stability(a, b):
    c = cf(a, b)
    da, db = (deff(a), deff(b))
    delta = abs(da - db) / ((da + db) / 2) if da is not None and db is not None else None
    return {'cosine': c, 'd_eff_half0': da, 'd_eff_half1': db, 'relative_d_eff_difference': delta, 'status': 'STABLE' if c is not None and delta is not None and (c >= 0.9) and (delta <= 0.25) else 'UNRESOLVED_GEOMETRY'}

def physical(b, l):
    return HI[l] @ b @ HI[l]

def balanced(a, half=None):
    select = np.ones(448, dtype=bool) if half is None else np.array([r['half'] == half for r in ROWS])
    return np.mean([a[select & np.array([r['domain'] == d for r in ROWS])].mean(0) for d in DOMAINS], axis=0)

def capture(model, ids):
    caps = {l: {} for l in LAYERS}
    kwargs = {}
    handles = []

    def mlp(l):

        def f(m, a, out):
            caps[l].update(mlp_input=a[0].detach().clone(), mlp_output=out[0].detach().clone())
        return f

    def pre(l):

        def f(m, a):
            caps[l]['residual'] = a[0].detach().clone()
        return f

    def post(l):

        def f(m, a, out):
            caps[l]['layer_output'] = out[0].detach().clone()
        return f

    def clone(v):
        if isinstance(v, torch.Tensor):
            return v.detach().clone()
        if isinstance(v, tuple):
            return tuple((clone(x) for x in v))
        return v

    def kw(m, a, k):
        kwargs.update({n: clone(v) for n, v in k.items()})
    try:
        for l in LAYERS:
            lay = model.model.layers[l]
            handles.extend([lay.mlp.register_forward_hook(mlp(l)), lay.post_attention_layernorm.register_forward_pre_hook(pre(l)), lay.register_forward_hook(post(l))])
        handles.append(model.model.layers[8].register_forward_pre_hook(kw, with_kwargs=True))
        with torch.no_grad():
            last = model.model(input_ids=ids, use_cache=False, output_router_logits=False).last_hidden_state.detach().clone()
    finally:
        for h in handles:
            h.remove()
    return (caps, kwargs, last)

def suffix(model, y, l, kw):
    h = y.reshape(1, -1, 2048).to(torch.bfloat16)
    for layer in model.model.layers[l + 1:]:
        h = layer(h, **kw)[0]
    return model.model.norm(h)

def logits(model, h):
    return model.lm_head(h)[0]

def loss_from_logits(z, ids):
    x = z[:-1].cpu().double()
    target = ids[0, 1:].cpu()
    return (torch.logsumexp(x, dim=-1) - x.gather(1, target[:, None]).squeeze(1)).mean()

def make_local(model, caps, l):
    c = caps[l]
    local = LocalMoE(model.model.layers[l].mlp, c['mlp_input'], c['residual'])
    local.validate_baseline(c['mlp_output'])
    assert torch.equal(local.native_residual_output()[0].reshape_as(c['layer_output']), c['layer_output']), ('ZERO_BASELINE_LAYER_MISMATCH', l)
    return local

def adjoint(model, y, l, kw, ids, baseline_logits_hash):
    leaf = y.float().detach().clone().requires_grad_(True)
    out = logits(model, suffix(model, leaf, l, kw))
    assert hashlib.sha256(out.detach().cpu().double().numpy().tobytes()).hexdigest() == baseline_logits_hash, ('SUFFIX_BASELINE_LOGITS_MISMATCH', l)
    loss = loss_from_logits(out, ids)
    a = torch.autograd.grad(loss, leaf)[0].detach()
    return (a.reshape(-1, 2048).cpu().double().numpy(), float(loss.detach()))

def consistent(a, b, scale=None):
    scale = np.maximum(np.abs(a), np.abs(b)) if scale is None else scale
    return bool(np.all(np.abs(a - b) <= NUMERICAL_ATOL + NUMERICAL_RTOL * scale))

def measure(model, row, index, cp, stage, scalar_only=False):
    ids = torch.tensor(row['token_ids'], device='mps').unsqueeze(0)
    caps, kw, last = capture(model, ids)
    with torch.no_grad():
        base = logits(model, last)
    basehash = hashlib.sha256(base.detach().cpu().double().numpy().tobytes()).hexdigest()
    ell0 = float(loss_from_logits(base, ids))
    out = {}
    cache_audit = []
    for li, l in enumerate(LAYERS):
        local = make_local(model, caps, l)
        a, ell = adjoint(model, caps[l]['layer_output'], l, kw, ids, basehash)
        assert ell == ell0, ('SUFFIX_NLL_MISMATCH', l, index)
        n = 1 if scalar_only else 6
        u = np.stack([continuous_fixed_support_jvp_fp32(local, v).detach().cpu().double().numpy() for v in BASIS[l][:n]])
        g = np.einsum('ith,th->i', u, a, dtype=np.float64, optimize=False)
        ug = np.einsum('ith,jth->ij', u, u, dtype=np.float64, optimize=False)
        out[f'L{l}_g'] = g
        out[f'L{l}_u_inner_products'] = ug
        out[f'L{l}_a_norm_sq'] = np.array(np.sum(a * a, dtype=np.float64))
        if scalar_only:
            out[f'L{l}_local_mode1_power'] = np.array(ug[0, 0] / (256 * 2048))
            out[f'L{l}_objective_mode1_power'] = np.array(g[0] ** 2)
        else:
            gram = exact_soft_gram(local)
            vj = BASIS[l]
            vr = vj.copy()
            vr[:, :64] = 0
            vg = vj.copy()
            vg[:, 64:] = 0
            ur = np.stack([continuous_fixed_support_jvp_fp32(local, v).detach().cpu().double().numpy() for v in vr])
            ugn = np.stack([continuous_fixed_support_jvp_fp32(local, v).detach().cpu().double().numpy() for v in vg])
            gr = np.einsum('ith,th->i', ur, a, dtype=np.float64, optimize=False)
            gg = np.einsum('ith,th->i', ugn, a, dtype=np.float64, optimize=False)
            assert consistent(g, gr + gg, np.maximum(np.abs(g), np.abs(gr) + np.abs(gg))), ('OBJECTIVE_COMPONENT_IDENTITY_FAILED', cp, l, index)
            bj = vj @ gram @ vj.T
            br = vr @ gram @ vr.T
            bg = vg @ gram @ vg.T
            brg = vr @ gram @ vg.T
            assert consistent(bj, br + bg + brg + brg.T), ('LOCAL_COMPONENT_IDENTITY_FAILED', cp, l, index)
            diff = np.linalg.norm(bj - ug / (256 * 2048))
            den = np.linalg.norm(bj)
            assert diff <= NUMERICAL_ATOL + NUMERICAL_RTOL * den, ('V160_GRAM_DIRECT_TANGENT_INCONSISTENCY', cp, l, index, diff, den)
            for probe, blocks in [('local', (bj, br, bg, brg)), ('objective', (np.outer(g, g), np.outer(gr, gr), np.outer(gg, gg), np.outer(gr, gg)))]:
                for name, b in zip(('J', 'R', 'G', 'RG'), blocks):
                    out[f'L{l}_{probe}_{name}'] = b
            out[f'L{l}_g_router'] = gr
            out[f'L{l}_g_gain'] = gg
            for name, b in [('router', np.einsum('ith,jth->ij', ur, ur, optimize=False)), ('gain', np.einsum('ith,jth->ij', ugn, ugn, optimize=False)), ('router_gain', np.einsum('ith,jth->ij', ur, ugn, optimize=False))]:
                out[f'L{l}_u_{name}_inner_products'] = b
            cache_audit.append({'layer': l, 'v160_gram_vs_cached_u_gram_frobenius_abs': float(diff), 'frobenius_relative': float(diff / den) if den > 0 else None, 'objective_component_max_abs': float(np.max(np.abs(g - gr - gg))), 'local_component_max_abs': float(np.max(np.abs(bj - br - bg - brg - brg.T)))})
            del ur, ugn
        del local, a, u
        torch.mps.empty_cache()
    assert all((np.isfinite(v).all() for v in out.values())), ('NONFINITE_DOCUMENT_ENDPOINT', cp, index)
    out.update(baseline_NLL=np.array(ell0), metadata_json=np.array(json.dumps(dict(META, stage=stage, tokens_B=cp, checkpoint=CP[cp], document_id=row['document_id'], source_id=row['source_id'], source_shard=row['source_shard'], source_physical_path=row['source_physical_path'], source_sha256=row['source_sha256'], token_sha256=row['token_sha256'], scalar_only=scalar_only, cache_audit=cache_audit))))
    del caps, kw, last, base, ids
    gc.collect()
    torch.mps.empty_cache()
    return out

def run_checkpoint(cp, stage, scalar_only=False):
    folder = RUN / 'document_cache' / stage / str(cp)
    folder.mkdir(parents=True, exist_ok=True)
    event('MODEL_LOAD_BEGIN', tokens_B=cp, stage=stage, revision=CP[cp]['revision_commit_sha'], allowed_modes=1 if scalar_only else 6)
    model = load_checkpoint(CP[cp], CACHE)
    event('MODEL_LOADED', tokens_B=cp, stage=stage, parameter_gradients_disabled=not any((p.requires_grad for p in model.parameters())))
    try:
        for i, row in enumerate(ROWS):
            p = folder / f'{i:03d}.npz'
            assert not p.exists(), ('UNEXPECTED_EXISTING_SCIENTIFIC_CACHE', str(p))
            out = measure(model, row, i, cp, stage, scalar_only)
            save(p, **out)
            event('DOCUMENT_DONE', stage=stage, tokens_B=cp, index=i, cache=ident(p))
            if i % 8 == 0 or i == 447:
                print(stage, cp, 'B', i + 1, '/448', flush=True)
    finally:
        del model
        gc.collect()
        torch.mps.empty_cache()
        event('MODEL_UNLOADED', tokens_B=cp, stage=stage)
    files = list(sorted(folder.glob('*.npz')))
    assert len(files) == 448
    keys = None
    data = {}
    index = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            kk = [k for k in z.files if k != 'metadata_json']
            if keys is None:
                keys = kk
                data = {k: [] for k in kk}
            assert kk == keys
            for k in keys:
                data[k].append(z[k])
        index.append(ident(p))
    data = {k: np.stack(v) for k, v in data.items()}
    data['document_ids'] = np.array([r['document_id'] for r in ROWS])
    data['domain'] = np.array([r['domain'] for r in ROWS])
    data['half'] = np.array([r['half'] for r in ROWS])
    write(RUN / (stage + '_' + str(cp) + '_CACHE_INDEX.json'), dict(META, stage=stage, tokens_B=cp, checkpoint=CP[cp], scalar_only=scalar_only, files=index))
    return data

def aggregate(d):
    for l in LAYERS:
        d[f'L{l}_H'] = BASIS[l] @ BASIS[l].T
        d[f'L{l}_H_inverse_sqrt'] = HI[l]
        for p in ('local', 'objective'):
            for c in ('J', 'R', 'G', 'RG'):
                b = d[f'L{l}_{p}_{c}']
                d[f'L{l}_{p}_{c}_mean_B'] = balanced(b)
                d[f'L{l}_{p}_{c}_physical'] = physical(balanced(b), l)
                for h in (0, 1):
                    d[f'L{l}_{p}_{c}_half{h}_physical'] = physical(balanced(b, h), l)
    return d

def scientific_npz(name, data, stage, cp):
    data['metadata_json'] = np.array(json.dumps(dict(META, stage=stage, checkpoints=[CP[t] for t in cp], numerical_consistency_tolerance={'absolute': NUMERICAL_ATOL, 'relative': NUMERICAL_RTOL}, cache_component_axes='gain first64; router second64; no independent mode renormalization', local_gram='unchanged v160 exact_soft_gram; FP32 selected-expert contraction, FP64 reductions')))
    save(RUN / name, **data)

def initialize(stage):
    global torch, transformers, LocalMoE, load_checkpoint, continuous_fixed_support_jvp_fp32, exact_soft_gram, META, ROWS, BASIS, HI, CP
    t0 = sealed('T0_BANK_T_SEAL')
    assert t0['T0_STATUS'] == 'PASS' and t0['documents'] == 448
    if stage == 'T2':
        sealed('T1_PREDICTION_SEAL')
    if stage == 'T3':
        sealed('T2_STAGE_SEAL')
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(PKG / 'reference_impl'))
    import torch as _torch, transformers as _transformers
    torch = _torch
    transformers = _transformers
    from project1_r2.model_io import load_checkpoint as _load, device_guard
    from project1_r2.expert_internal import LocalMoE as _local
    from project1_r2.shadow_oracle_repair import continuous_fixed_support_jvp_fp32 as _jvp
    from v160_contract import exact_soft_gram as _gram, require_certified
    LocalMoE = _local
    load_checkpoint = _load
    continuous_fixed_support_jvp_fp32 = _jvp
    exact_soft_gram = _gram
    device_guard('mps')
    certp = H / '2026-10-05/jie/outputs/outputs_v160/backend_equivalence_certification_v160.json'
    historical_checks = {}
    for line in (certp.parent / 'SHA256SUMS_v160.txt').read_text().splitlines():
        h, n = line.split(None, 1)
        historical_checks[n.strip().lstrip('*')] = h
    assert sha(certp) == historical_checks[certp.name]
    cert = read(certp)
    cfg = dict(device='mps', compute_dtype='BF16 forward; FP32 clean tangent; FP64 CPU reductions', kernel='transformers eager / frozen torch topk', loading='frozen load_checkpoint exact SHA', torch=torch.__version__, transformers=transformers.__version__, checkpoint='e5642d2d7b37c138775ddf59f685f91788e173b7')
    require_certified(cert, cfg)
    sap = certp.parent / 'implementation_source_audit_v160.json'
    assert sha(sap) == historical_checks[sap.name]
    sourceaudit = read(sap)
    for name, expected in sourceaudit['key_files'].items():
        assert sha(SRC.parent / name) == expected, ('HISTORICAL_CERTIFIED_SOURCE_HASH', name)
    env = read(RUN / 'T0_CHECKPOINT_ENVIRONMENT_AUDIT.json')
    CP = {r['tokens_B']: r for r in env['checkpoints']}
    ROWS = [json.loads(r) for r in (RUN / 'Formal_Bank_T_v1.jsonl').read_text().splitlines()]
    assert len(ROWS) == 448
    BASIS = {}
    HI = {}
    with np.load(PKG / 'basis/p2b_formala_corrected_6d_basis.npz', allow_pickle=False) as z:
        for l in LAYERS:
            v = z[f'L{l}_controls_exact_common_projected']
            BASIS[l] = v
            vals, vec = np.linalg.eigh(v @ v.T)
            assert vals.min() > 0
            HI[l] = vec * vals ** (-0.5) @ vec.T
    META = {'protocol': ident(PKG / 'protocol/Project1_P2C_Restricted6D_Tomography_Protocol_v1p1_FROZEN.md'), 'basis': ident(PKG / 'basis/p2b_formala_corrected_6d_basis.npz'), 'Bank_T_seal': ident(RUN / 'T0_BANK_T_SEAL.json'), 'Bank_T_seal_sha256': sha(RUN / 'T0_BANK_T_SEAL.json'), 'source_artifacts': [ident(RUN / 'Formal_Bank_T_v1.jsonl'), ident(RUN / 'T0_OPENED_OVERLAP_LEDGER.json'), ident(RUN / 'T0_RECOVERED_SHARDS.json'), ident(RUN / 'T0_CHECKPOINT_ENVIRONMENT_AUDIT.json')], 'model_repository': 'allenai/OLMoE-1B-7B-0924', 'software_environment': dict(env['software_environment'], torch=torch.__version__, transformers=transformers.__version__, device='mps', forward_dtype='BF16', tangent_dtype='FP32', reductions='FP64 CPU', batch_size=1, MPS_fallback=False), 'implementation_sources': [ident(Path(__file__)), ident(PKG / 'reference_impl/v160_contract.py'), ident(certp)] + [ident(SRC / 'project1_r2' / n) for n in ('model_io.py', 'expert_internal.py', 'shadow_oracle_repair.py', 'hard_routing_diagnostic.py')], 'NEW_TRAINING_STEP_COUNT': 0, 'RESERVE_ACCESS_COUNT': 0, 'NEW_NULL_BANK_COUNT': 0, 'BASIS_REFIT_COUNT': 0, 'OUTCOME_DEPENDENT_PROTOCOL_CHANGE_COUNT': 0}
    event('STAGE_GATE_PASS', stage=stage, T0_seal_sha256=META['Bank_T_seal_sha256'], certified_backend_same=True)

def t1():
    a = aggregate(run_checkpoint(41, 'T1_41B'))
    scientific_npz('T1_41B_TOMOGRAPHY.npz', a, 'T1', [41])
    b = run_checkpoint(5033, 'T1_5033_MODE1', scalar_only=True)
    scalars = []
    prediction = []
    for l in LAYERS:
        for p in ('local', 'objective'):
            vals = b[f'L{l}_{p}_mode1_power']
            q5033 = float(balanced(vals))
            q41 = float(a[f'L{l}_{p}_J_mean_B'][0, 0])
            assert q41 > 0 and q5033 >= 0
            c = q5033 / q41
            scalars.append({'layer': l, 'probe': p, 'Q5033_mode1': q5033, 'per_document_mode1_power': vals.tolist()})
            prediction.append({'layer': l, 'probe': p, 'Q41_mode1': q41, 'Q5033_mode1': q5033, 'scalar_rescaling_c': c, 'predicted_physical_operator': (c * a[f'L{l}_{p}_J_physical']).tolist(), 'decision_thresholds': {'epsilon_scale_max': 0.2, 'shape_cosine_min': 0.95, 'split_shape_cosine_min': 0.9, 'split_d_eff_relative_difference_max': 0.25}})
    write(RUN / 'T1_5033_MODE1_SCALARS.json', dict(META, status='COMPLETE', checkpoint=CP[5033], permitted_terminal_modes=[1], scalars=scalars))
    write(RUN / 'T1_SCALAR_PREDICTIONS.json', dict(META, status='FROZEN', checkpoints=[CP[41], CP[5033]], prediction_rule='Gtilde_pred5033 = (Q5033(v1)/Q41(v1))*Gtilde41', predictions=prediction, full_5033_6D_computed=False, full_5033_6D_read=False))
    seal('T1_PREDICTION_SEAL', [RUN / n for n in ('T1_41B_TOMOGRAPHY.npz', 'T1_5033_MODE1_SCALARS.json', 'T1_SCALAR_PREDICTIONS.json', 'T1_41B_41_CACHE_INDEX.json', 'T1_5033_MODE1_5033_CACHE_INDEX.json')], T1_STATUS='PASS', prediction_sha256=sha(RUN / 'T1_SCALAR_PREDICTIONS.json'), held_out_information_barrier_preserved=True)
    print('T1 prediction SEALED', flush=True)

def t2():
    prediction = read(RUN / 'T1_SCALAR_PREDICTIONS.json')
    observed = aggregate(run_checkpoint(5033, 'T2_5033'))
    scientific_npz('T2_5033_TOMOGRAPHY.npz', observed, 'T2', [5033])
    score = []
    with np.load(RUN / 'T1_41B_TOMOGRAPHY.npz', allow_pickle=False) as baseline:
        for pr in prediction['predictions']:
            l, p = (pr['layer'], pr['probe'])
            prefix = f'L{l}_{p}_J'
            got = observed[prefix + '_physical']
            pred = np.array(pr['predicted_physical_operator'])
            eps = float(np.linalg.norm(got - pred) / np.linalg.norm(got)) if np.linalg.norm(got) > 0 else None
            c = cf(got, pred)
            s41 = stability(baseline[prefix + '_half0_physical'], baseline[prefix + '_half1_physical'])
            s5033 = stability(observed[prefix + '_half0_physical'], observed[prefix + '_half1_physical'])
            adequate = eps is not None and c is not None and (eps <= 0.2) and (c >= 0.95)
            stable = s41['status'] == s5033['status'] == 'STABLE'
            label = ('SCALAR_OPERATOR_APPROXIMATION_ADEQUATE' if adequate else 'OPERATOR_RESHAPING_RESOLVED') if stable else 'SCALAR_VS_RESHAPING_UNRESOLVED'
            score.append({'layer': l, 'probe': p, 'epsilon_scale': eps, 'shape_cosine': c, 'baseline_split_stability': s41, 'terminal_split_stability': s5033, 'decision': label, 'prediction_sha256': sha(RUN / 'T1_SCALAR_PREDICTIONS.json'), 'thresholds_unchanged': True})
    write(RUN / 'T2_PREDICTION_SCORE.json', dict(META, status='COMPLETE', checkpoint=CP[5033], scores=score, prediction_sha256=sha(RUN / 'T1_SCALAR_PREDICTIONS.json')))
    seal('T2_STAGE_SEAL', [RUN / n for n in ('T2_5033_TOMOGRAPHY.npz', 'T2_PREDICTION_SCORE.json', 'T2_5033_5033_CACHE_INDEX.json')], T2_STATUS='PASS', scores=score)
    print('T2 SEALED', [(r['layer'], r['probe'], r['decision']) for r in score], flush=True)

def t3():
    result = {}
    for cp in (146, 503, 1740):
        data = aggregate(run_checkpoint(cp, 'T3_INTERMEDIATE'))
        for k, v in data.items():
            result[f'T{cp}_{k}'] = v
    scientific_npz('T3_INTERMEDIATE_TOMOGRAPHY.npz', result, 'T3', [146, 503, 1740])
    seal('T3_STAGE_SEAL', [RUN / 'T3_INTERMEDIATE_TOMOGRAPHY.npz'] + [RUN / f'T3_INTERMEDIATE_{cp}_CACHE_INDEX.json' for cp in (146, 503, 1740)], T3_STATUS='PASS')
    print('T3 SEALED', flush=True)
if __name__ == '__main__':
    stage = sys.argv[1]
    try:
        assert stage in ('T1', 'T2', 'T3')
        initialize(stage)
        {'T1': t1, 'T2': t2, 'T3': t3}[stage]()
    except Exception as e:
        event('EXECUTION_BLOCKED', stage=stage, exact_error=str(e), traceback=traceback.format_exc())
        raise
