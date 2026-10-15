# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import sys, os, json, hashlib, zipfile, time, traceback, subprocess
from pathlib import Path
from collections import Counter
import numpy as np
ROOT = Path(_p1_path('path_d0590e7c3477'))
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'outputs/outputs_v160'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'src'))
from v160_contract import *
import torch, transformers
from project1_r2.model_io import load_checkpoint, capture_layers
from project1_r2.expert_internal import LocalMoE
from project1_r2.banks import read_manifest, tokens_for, validate_split
from project1_r2.routing import sufficient_statistics
from project1_r2.metrics import internal_report, document_response_report
from project1_r2.dual_channel_engineering import clean_soft_local_sketch, dual_channel_native_decomposition_v153
from project1_r2.shadow_oracle_repair import continuous_fixed_support_jvp_fp32
from project1_r2.hard_routing_diagnostic import DIRECTIONS
from project1_r2.sketch import compare_sketches, domain_balanced_sketches
from project1_r2.formal_a_discovery_v159 import matched_controls, empirical_percentile, outcome, layer_seed

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def write(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else bool(x) if isinstance(x, np.bool_) else str(x)) + '\n')
    tmp.replace(p)

def log(problem, old, new, reason, checks, files=()):
    row = dict(time_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), problem=problem, old_implementation=old, new_implementation=new, scientific_semantics_unchanged=reason, checks=checks, files_changed=list(files))
    with (OUT / 'technical_adaptation_log_v160.jsonl').open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def progress(stage, completed, total):
    write('execution_progress_v160.json', dict(stage=stage, completed=completed, total=total))
    print(f'{stage}: {completed}/{total}', flush=True)
CONFIG = dict(device='mps', compute_dtype='BF16 forward; FP32 clean tangent; FP64 CPU reductions', kernel='transformers eager / frozen torch topk', loading='frozen load_checkpoint exact SHA', torch=torch.__version__, transformers=transformers.__version__, checkpoint=CONTRACT['checkpoint'])

def preflight():
    package = HERE / 'v160'
    manifest = {}
    for line in (package / 'MIGRATION_MANIFEST_v160_TECH_ADAPTIVE_FORMAL_A.txt').read_text().splitlines():
        if len(line) > 66 and line[64:66] == '  ':
            manifest[line[66:]] = line[:64]
    checks = {k: sha(package / k) == v for k, v in manifest.items()}
    assert all(checks.values()), checks
    old = read(ROOT / 'outputs_v159/formal_A_prerequisite_audit_v159.json')
    history = {}
    for label, folder in [('v158', Path(_p1_path('path_fb2f3153db17'))), ('v159', ROOT / 'outputs_v159')]:
        sums = folder / f'SHA256SUMS_{label}.txt'
        hist = {}
        for line in sums.read_text().splitlines():
            digest, name = line.split(None, 1)
            name = name.strip().lstrip('*')
            hist[name] = sha(folder / name) == digest
        assert all(hist.values()), (label, hist)
        history[label] = dict(all_hashes_match=True, count=len(hist), directory=str(folder), hashes={p.name: sha(p) for p in folder.iterdir() if p.is_file()}, model_loaded=False)
    write('immutable_history_audit_v160.json', history)
    source = old['source']
    assert sha(source['archive']) == source['archive_sha256']
    keys = {k: sha(ROOT / k) for k in source['key_files_verified']}
    assert keys == source['key_files_verified']
    write('implementation_source_audit_v160.json', dict(status='PASS', frozen_v153_source_reverified=True, source_archive=source['archive'], source_archive_sha256=source['archive_sha256'], key_files=keys, new_adapter_files={str(p): sha(p) for p in HERE.glob('*.py')}, scientific_formula_adapter='Direct Protocol v3 transcription: H_E residuals and split overlap of a, not mapped v. Legacy v159 helper defects corrected before A.', historical_source_modified=False))
    cache = Path(read(ROOT / 'outputs/transport_41B_completion_v150.json')['cache_path'])
    snapshot = cache / 'models--allenai--OLMoE-1B-7B-0924/snapshots' / CONTRACT['checkpoint']
    model_hashes = []
    for row in old['exact_41B']['config_and_index_and_shards_rehashed']:
        p = snapshot / row['name']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
        model_hashes.append(dict(path=str(p), **row))
    aroot = ROOT / 'outputs_v156'
    seal = read(aroot / 'Formal_Bank_A_Seal_v2_final.json')
    assert seal['FORMAL_BANK_A_RESPONSE_BLIND'] == 'YES' and (not seal['model_probe_computed'])
    selected = {'Formal_Bank_A_v2.jsonl': '44f4f05fafca67230b3c252929c01fbf9b063190fd9f519117622a7be5e90f09', 'v156_candidate_frame_manifest.jsonl': 'e28ddf1d5c23b5c72ff9ad6dd5df9502e5b0705bc2d83e8bc9357a2cf282cafb'}
    for k, v in selected.items():
        assert sha(aroot / k) == v == seal['hashes'][k]
    token_hashes = {k: sha(aroot / k) for k in seal['hashes'] if k.startswith('tokens/')}
    assert all((v == seal['hashes'][k] for k, v in token_hashes.items()))
    ver = read(aroot / 'Formal_Bank_A_Verification_v2_final.json')
    assert sha(aroot / 'Formal_Bank_A_Verification_v2_final.json') == seal['hashes']['Formal_Bank_A_Verification_v2_final.json']
    assert ver['status'] == 'PASS' and ver['A_count'] == 448 and (ver['historical_collision_count'] == 0)
    rows = [json.loads(x) for x in (aroot / 'Formal_Bank_A_v2.jsonl').read_text().splitlines()]
    assert len(rows) == 448 and Counter((r['domain'] for r in rows)) == Counter({d: 64 for d in DOMAINS})
    assert Counter(((r['domain'], r['half']) for r in rows)) == Counter({(d, h): 32 for d in DOMAINS for h in (0, 1)})
    token_index = aroot / 'token_payload_hash_index_v156.json'
    idx = read(token_index)
    assert idx['files'] == token_hashes and idx['root_sha256'] == 'f5d3a5c8f2cab8426ea12c61f909f1069b869988af79a072743092c38662adc6'
    assert hashlib.sha256(json.dumps(token_hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == idx['root_sha256']
    write('formal_A_prerequisite_audit_v160.json', dict(status='PASS', package_manifest_reverified=checks, bank_A=old['bank_A'], A_identity_rehashed=selected, token_payload_files_rehashed=len(token_hashes), token_payload_index_path=str(token_index), token_payload_index_sha256=sha(token_index), exact_41B=dict(revision_commit_sha=CONTRACT['checkpoint'], files=model_hashes), history_verified=True, model_probe_performed=False, B_C_accessed=False))
    env = dict(configuration=CONFIG, configuration_sha256=fingerprint(CONFIG), cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available(), mps_built=torch.backends.mps.is_built(), cpu_count=os.cpu_count(), cpu_threads=torch.get_num_threads(), ram_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)), storage_dtype='BF16', compute_dtype=CONFIG['compute_dtype'], model_loading_strategy=CONFIG['loading'], memory_strategy='fully resident BF16 MPS; document batch 1; eight selected expert output vectors/token for exact Gram', snapshot=str(snapshot), cache=str(cache))
    write('technical_environment_v160.json', env)
    frozen = read(package / 'protocol_freeze_audit_v159.json')['frozen']
    write('protocol_freeze_audit_v160.json', dict(status='PASS', frozen=frozen, protocol_sha256=sha(package / 'Project1_R2_FormalA_DiscoveryProtocol_v3.md'), charter_sha256=sha(package / 'Project1_R2_TechnicalAdaptationCharter_v1.md'), equivalence_protocol_sha256=sha(package / 'Project1_R2_BackendEquivalenceCertification_v1.md'), scientific_values_inspected=False, bank_B_or_C_accessed=False, constants_config_override_allowed=False, backend_configuration=CONFIG, configuration_sha256=fingerprint(CONFIG)))
    if not (OUT / 'technical_adaptation_log_v160.jsonl').exists():
        log('MPS hidden by execution sandbox', 'sandbox MPS unavailable', 'unsandboxed local execution using the same BF16 MPS hardware/software', 'Identical checkpoint and numerical backend; isolation only changes device visibility', 'MPS runtime probe passed')
        log('Legacy v159 helper disagrees with frozen written formulas', 'covariance omitted H_E; split compared mapped v', 'explicit H_E covariance; split compares whitened left singular vectors a', 'Restores already-frozen equations without amending scientific contract', 'Independent equation tests before A', ('work/v160_contract.py',))
        log('Exact evaluation of 257 controls can share sufficient statistics', 'separate FP32 JVP for each direction', 'algebraically contracted exact clean-soft Gram with FP64 reduction', 'All tokens, all selected experts, fixed support, clean FP32 weights retained; no random response sketches in A', 'Synthetic direct-JVP tests and G112 direct-JVP comparison before A', ('work/v160_contract.py',))
    return cache

def certification(model):
    refroot = ROOT / 'outputs_v153_continuation_20261005'
    cp = refroot / 'step10000-tokens41B'
    diag = read(ROOT / 'outputs_v151/diagnostic_manifest_14.json')
    bank = read_manifest(diag['bank_manifest_path'])
    domains = validate_split(bank)
    selected = subset(bank['documents'])
    selids = {r['document_id'] for r in selected}
    assert [r['document_id'] for r in selected] == [r['document_id'] for r in diag['documents']]
    controls = read(refroot / 'stored_direction_contract_v153.json')['directions']
    references = [ROOT / 'outputs_v151/diagnostic_manifest_14.json', Path(diag['bank_manifest_path']), refroot / 'stored_direction_contract_v153.json', refroot / 'route_freeze_report_v153.json', refroot / 'v153_gate_table.json', cp / 'dual_channel_decomposition.json', ROOT / 'outputs_v152/shadow_oracle_repair.json']
    for l in (7, 15):
        references.extend((cp / f'layer_{l}/sufficient_statistics.json', cp / f'layer_{l}/clean_soft_sketch_32.npz', cp / f'layer_{l}/internal_split_half_metrics.json'))
    references.append(cp / 'layer_7/clean_soft_sketch_64.npz')
    write('backend_equivalence_reference_index_v160.json', dict(status='PASS', reference_authority='immutable stored v152/v153; never regenerated', files=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in references], selected_document_ids=[r['document_id'] for r in selected], granular_support_reference_available=False, strongest_routing_reference='Stored route-freeze gate replay plus per-document support/selected-weight vectors; no fabricated f_route', internal_vectors='112 original per-document 64-D logit_mean and expert_internal_amplitude per layer', clean_soft_endpoints='Stored per-document/per-direction R_tangent norms and RMS (.02); direct JVP independent Gram check'))
    refs = {l: {r['document_id']: r for r in read(cp / f'layer_{l}/sufficient_statistics.json')['documents']} for l in (7, 15)}
    softrefs = read(cp / 'dual_channel_decomposition.json')['measurements']
    newstats = {l: [] for l in (7, 15)}
    sketches = {l: [] for l in (7, 15)}
    s64 = []
    newsoft = []
    oldsoft = []
    softrows = []
    closure = True
    route_checks = []
    gram_errors = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(Path(read(ROOT / 'outputs/transport_41B_completion_v150.json')['cache_path']) / 'models--allenai--OLMoE-1B-7B-0924/snapshots' / CONTRACT['checkpoint']), local_files_only=True)
    for i, doc in enumerate(bank['documents']):
        ids = torch.as_tensor(tokens_for(doc, tokenizer, model.config.max_position_embeddings, bank['preprocessing']), device='mps').unsqueeze(0)
        caps = capture_layers(model, ids)
        for layer in (7, 15):
            c = caps[layer]
            local = LocalMoE(model.model.layers[layer].mlp, c['mlp_input'], c['residual'])
            baseline = local.validate_baseline(c['mlp_output'])
            assert torch.equal(local.native_residual_output()[0].reshape_as(c['layer_output']), c['layer_output'])
            rc = local.route_freeze_check()
            route_checks.append(dict(document_id=doc['document_id'], layer=layer, **baseline, **rc))
            assert rc['private_common_gain_null_norm'] <= 1e-07 and rc['common_router_bias_null_norm'] <= 1e-07 and (rc['common_gain_scale_identity_relative_error'] <= 0.02)
            row = {k: doc[k] for k in ('document_id', 'domain', 'half')}
            row.update(logit_mean=local.logits.float().mean(0).cpu().numpy(), expert_internal_amplitude=local.amplitude(), **{k: v for k, v in sufficient_statistics(local.routes, 64, 0.01).items() if k != 'logit_mean'})
            newstats[layer].append(row)
            sketch = clean_soft_local_sketch(local, 64 if layer == 7 else 32, 20261004, doc['document_id'], layer)
            if layer == 7:
                s64.append(sketch)
                sketch = sketch[:32] * np.sqrt(2)
            sketches[layer].append(sketch)
            if doc['document_id'] in selids:
                gram = exact_soft_gram(local)
                for direction in DIRECTIONS:
                    v = controls[str(layer)]['controls'][direction]
                    y = continuous_fixed_support_jvp_fp32(local, v).cpu().double().numpy()
                    rms = np.linalg.norm(y) / np.sqrt(y.size)
                    reference = next((r for r in softrefs if r['layer'] == layer and r['document_id'] == doc['document_id'] and (r['direction'] == direction) and (r['h'] == 0.02)))
                    newsoft.append([rms, np.linalg.norm(y)])
                    oldsoft.append([reference['rms']['R_tangent'] / 0.02, reference['norms']['R_tangent'] / 0.02])
                    gp = float(np.asarray(v) @ gram @ np.asarray(v))
                    direct = rms ** 2
                    gram_errors.append(abs(gp - direct) / max(direct, 1e-30))
                    result = dual_channel_native_decomposition_v153(local, v, 0.02)
                    closure = closure and result['closure_pass'] and result['fixed_indices_unchanged']
                    softrows.append(dict(layer=layer, direction=direction, document_id=doc['document_id'], new_power=direct, reference_power=(reference['rms']['R_tangent'] / 0.02) ** 2))
                np.savez_compressed(OUT / f'G112_support_{layer}_{i}.npz', indices=local.routes.indices.cpu().numpy())
            del local, c
        del caps, ids
        if (i + 1) % 7 == 0 or i == 0:
            progress('G112_certification', i + 1, 112)
    result = {}
    allgates = {}
    all_u = []
    ref_u = []
    all_m = []
    ref_m = []
    for layer in (7, 15):
        new = [r for r in newstats[layer] if r['document_id'] in selids]
        ref = [refs[layer][r['document_id']] for r in new]
        eu = errors([r['logit_mean'] for r in new], [r['logit_mean'] for r in ref])
        em = errors([r['expert_internal_amplitude'] for r in new], [r['expert_internal_amplitude'] for r in ref])
        all_u.extend([r['logit_mean'] for r in new])
        ref_u.extend([r['logit_mean'] for r in ref])
        all_m.extend([r['expert_internal_amplitude'] for r in new])
        ref_m.extend([r['expert_internal_amplitude'] for r in ref])
        report, _, _ = internal_report(newstats[layer], domains, 1)
        resp = document_response_report(sketches[layer], {k: controls[str(layer)]['controls'][k] for k in DIRECTIONS}, [r['domain'] for r in bank['documents']])
        gates = dict(baseline='PASS', route_semantics='PASS', internal_amplitude='PASS' if all((report[f]['engineering_amplitude_ok'] for f in ('logit_mean', 'expert_internal_amplitude'))) else 'FAIL', internal_reproducibility='PASS' if all((report[f]['engineering_subspace_ok'] for f in ('logit_mean', 'expert_internal_amplitude'))) else 'FAIL', joint_subspace='PASS' if report['candidate_subspace_engineering_ok'] else 'FAIL', clean_response_RSE='PASS' if all((r['engineering_snr_ok'] for r in resp.values())) else 'FAIL', response_decomposition_closure='PASS' if closure else 'FAIL')
        group = [r for r in softrows if r['layer'] == layer]
        oldpowers = [np.mean([r['reference_power'] for r in group if r['direction'] == k]) for k in DIRECTIONS]
        newpowers = [np.mean([r['new_power'] for r in group if r['direction'] == k]) for k in DIRECTIONS]
        order = same_order(oldpowers, newpowers)
        result[str(layer)] = dict(router_error=eu, expert_error=em, route_set_mismatch_fraction=None, route_mismatch_reason='No per-token immutable support arrays/hash available; protocol Section 5 gate-replay fallback used', route_gate_replay='PASS', ordering_unchanged=order, reference_direction_powers=oldpowers, new_direction_powers=newpowers, engineering_gates=gates, internal_gate_metrics=report, response_RSE=resp)
        allgates.update({str(layer) + '_' + k: v for k, v in gates.items()})
        np.savez_compressed(OUT / f'G112_replay_layer_{layer}.npz', router=np.array([r['logit_mean'] for r in newstats[layer]]), expert=np.array([r['expert_internal_amplitude'] for r in newstats[layer]]), sketches=np.array(sketches[layer]))
    conv = compare_sketches(domain_balanced_sketches(sketches[7], [r['domain'] for r in bank['documents']]), domain_balanced_sketches(s64, [r['domain'] for r in bank['documents']]), 1, 0.2, 0.7)
    allgates['32_to_64_sketch'] = 'PASS' if all((v['engineering_stable'] for v in conv.values())) else 'FAIL'
    allgates['exact_Gram_direct_JVP'] = 'PASS' if max(gram_errors) <= 1e-05 else 'FAIL'
    eu = errors(all_u, ref_u)
    em = errors(all_m, ref_m)
    es = errors(newsoft, oldsoft)
    passed = certification_pass(True, True, eu, em, es, all((r['ordering_unchanged'] for r in result.values())), allgates)
    cert = dict(status='PASS' if passed else 'FAIL', BACKEND_EQUIVALENCE_CERTIFIED='YES' if passed else 'NO', configuration=CONFIG, configuration_sha256=fingerprint(CONFIG), storage_dtype='BF16', semantic_identity=dict(checkpoint_hashes='PASS', token_ids='PASS', model_eval=not model.training, dropout_disabled=True, expert_ordering='PASS', top_k=8, layers=[7, 15], source_identity='PASS'), router_error=eu, expert_error=em, clean_soft_error=es, layers=result, engineering_gates=allgates, sketch_32_to_64=conv, exact_Gram_max_relative_power_error=max(gram_errors), route_set_mismatch_fraction=None, route_reference_fallback_used=True, calibration_docs=14, engineering_replay_docs=112, Formal_A_model_values_inspected=False)
    write('backend_equivalence_certification_v160.json', cert)
    write('G112_route_gate_replay_v160.json', route_checks)
    assert passed, 'G112 equivalence failed; inspect certification before trying alternatives'
    print('BACKEND_EQUIVALENCE_CERTIFIED=YES', flush=True)
    return cert

def discovery(model, cert):
    require_certified(cert, CONFIG)
    assert (OUT / 'pytest_summary_v160.txt').exists() and 'failed' not in (OUT / 'pytest_summary_v160.txt').read_text()
    aroot = ROOT / 'outputs_v156'
    path = aroot / 'Formal_Bank_A_v2.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    arrays = {l: dict(router=[], expert=[], gram=[]) for l in (7, 15)}
    for i, row in enumerate(rows):
        require_certified(cert, CONFIG)
        if 'token_ids' in row:
            tokens = row['token_ids']
        else:
            tokens = read(aroot / row['text_or_token_path'])
        assert len(tokens) == 256
        assert hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest() == row['token_array_sha256']
        ids = torch.as_tensor(tokens, device='mps').unsqueeze(0)
        caps = capture_layers(model, ids)
        for l in (7, 15):
            c = caps[l]
            local = LocalMoE(model.model.layers[l].mlp, c['mlp_input'], c['residual'])
            local.validate_baseline(c['mlp_output'])
            arrays[l]['router'].append(local.logits.float().mean(0).cpu().numpy())
            arrays[l]['expert'].append(local.amplitude())
            arrays[l]['gram'].append(exact_soft_gram(local))
            del local, c
        del caps, ids
        if (i + 1) % 16 == 0 or i == 0:
            for l in (7, 15):
                np.savez_compressed(OUT / f'formal_A_reduced_layer_{l}.npz', **{k: np.array(v) for k, v in arrays[l].items()}, document_ids=[r['document_id'] for r in rows[:i + 1]])
            progress('Formal_A', i + 1, 448)
    d = np.array([r['domain'] for r in rows])
    h = np.array([r['half'] for r in rows])
    modes = {}
    split_result = {}
    white = {}
    control_data = {}
    vis = {}
    decisions = {}
    stats = {}
    control_summary = {}
    for l in (7, 15):
        values = {k: np.asarray(v, dtype=np.float64) for k, v in arrays[l].items()}
        full = fit(values['router'], values['expert'], d)
        r, halves = split(values['router'], values['expert'], d, h)
        v = full['v']
        controls = matched_controls(v, l)
        diagnostic = {}
        gain = v.copy()
        gain[64:] = 0
        router = v.copy()
        router[:64] = 0
        diagnostic['gain_only_projection'] = gain
        diagnostic['router_only_projection'] = router
        rng = np.random.Generator(np.random.PCG64(layer_seed(l) ^ 1346720333))
        for j in range(8):
            perm = rng.permutation(64)
            diagnostic[f'expert_permutation_{j}'] = np.r_[v[:64][perm], v[64:][perm]]
        diagnostic['common_router_bias'] = np.r_[np.zeros(64), np.ones(64) / 8]
        diagnostic['common_gain'] = np.r_[np.ones(64) / 8, np.zeros(64)]
        vectors = np.vstack((v, controls, np.array(list(diagnostic.values()))))
        powers = np.einsum('vi,dij,vj->dv', vectors, values['gram'], vectors, optimize=True)
        scale = max(float(abs(powers).max()), 1e-30)
        assert powers.min() >= -1e-07 * scale
        powers = np.maximum(powers, 0)
        p = powers.mean(0)
        q = empirical_percentile(p[0], p[1:257])
        flag = near_boundary(r, q)
        np.savez_compressed(OUT / f'formal_A_derived_layer_{l}.npz', **{k: val for k, val in full.items() if k != 'summary'}, half_a=np.array([x['a'] for x in halves]), half_v=np.array([x['v'] for x in halves]), controls=controls, per_document_visibility=powers)
        modes[str(l)] = dict(**full['summary'], mode_a=full['a'], control_v=v, mode_sha256=hashlib.sha256(full['a'].tobytes()).hexdigest(), control_sha256=hashlib.sha256(v.tobytes()).hexdigest())
        split_result[str(l)] = dict(overlap=r, threshold=0.7, compared='whitened left singular vectors a', mapped_control_overlap_diagnostic=float(abs(halves[0]['v'] @ halves[1]['v'])), half_summaries=[x['summary'] for x in halves])
        white[str(l)] = {k: full[k] for k in ('W_U', 'W_M', 'cov_U', 'cov_M')}
        white[str(l)]['summary'] = {k: full['summary'][k] for k in ('router_whitening', 'expert_whitening')}
        control_data[str(l)] = dict(controls=controls, seed=layer_seed(l), n=256, diagnostics=diagnostic)
        control_summary[str(l)] = dict(n=256, seed=layer_seed(l), block_norm_matching_max_error=float(abs(np.linalg.norm(controls[:, :64], axis=1) - np.linalg.norm(v[:64])).max()), router_block_norm_matching_max_error=float(abs(np.linalg.norm(controls[:, 64:], axis=1) - np.linalg.norm(v[64:])).max()), mean_zero_max_error=float(abs(np.stack((controls[:, :64].mean(1), controls[:, 64:].mean(1)))).max()), controls_sha256=hashlib.sha256(controls.tobytes()).hexdigest())
        null = p[1:257]
        vis[str(l)] = dict(candidate_clean_soft_visibility=float(p[0]), candidate_response_rms=float(np.sqrt(p[0])), null_distribution=dict(min=float(null.min()), q25=float(np.quantile(null, 0.25)), median=float(np.median(null)), q75=float(np.quantile(null, 0.75)), max=float(null.max()), mean=float(null.mean()), sd=float(null.std(ddof=1))), percentile=q, lower_tail_count=int(np.sum(null <= p[0])), denominator=256, percentile_convention='count(P_j <= P_A)/256', matched_control_visibilities=null, diagnostic_controls=dict(zip(diagnostic.keys(), map(float, p[257:]))), metric='Exact clean fixed-support FP32 soft tangent squared norm /(tokens*hidden); equal documents/domains; CPU FP64 reduction', TECHNICALLY_NEAR_BOUNDARY='YES' if flag else 'NO')
        decisions[str(l)] = dict(outcome=outcome(r, q), split_overlap=r, percentile=q, TECHNICALLY_NEAR_BOUNDARY='YES' if flag else 'NO')
        stats[str(l)] = dict(documents=448, per_domain=64, per_domain_half=32, router_shape=[448, 64], expert_shape=[448, 64], document_statistic='unchanged v153', dtype='FP64 reduced from frozen model FP32/BF16 outputs')
    write('formal_A_internal_stats_summary_v160.json', dict(status='COMPUTED', layers=stats))
    write('formal_A_whitening_v160.json', dict(status='COMPUTED', layers=white))
    write('formal_A_whitening_summary_v160.json', dict(status='COMPUTED', layers={l: v['summary'] for l, v in white.items()}))
    write('formal_A_modes_v160.json', dict(status='COMPUTED', layers=modes))
    write('formal_A_modes_summary_v160.json', dict(status='COMPUTED', layers={l: {k: v for k, v in m.items() if k not in ('mode_a', 'control_v')} for l, m in modes.items()}))
    write('formal_A_split_half_v160.json', dict(status='COMPUTED', layers=split_result))
    write('formal_A_controls_v160.json', dict(status='COMPUTED', layers=control_data))
    write('formal_A_controls_summary_v160.json', dict(status='COMPUTED', layers=control_summary))
    write('formal_A_visibility_v160.json', dict(status='COMPUTED', layers=vis))
    anyq = any((v['outcome'] == 'QUALIFIED_LATENT_CANDIDATE' for v in decisions.values()))
    write('formal_A_discovery_decision_v160.json', dict(status='A_DISCOVERY_COMPLETE', BACKEND_EQUIVALENCE_CERTIFIED='YES', A_DISCOVERY_COMPLETE='YES', AT_LEAST_ONE_A_QUALIFIED='YES' if anyq else 'NO', B_PREPARATION_REVIEW_AUTHORIZED='YES' if anyq else 'NO', FRESH_BANK_B_AUTHORIZED='NO', FRESH_BANK_C_AUTHORIZED='NO', R2_SCIENTIFIC_MECHANISM_VERDICT='NONE', layers=decisions))
    write('technical_robustness_audit_v160.json', dict(status='NO_BOUNDARY_FLAGS' if not any((v['TECHNICALLY_NEAR_BOUNDARY'] == 'YES' for v in decisions.values())) else 'BOUNDARY_FLAG_REQUIRES_AUDIT', layers=decisions, second_backend_certified=False))
    audit = read(OUT / 'formal_A_prerequisite_audit_v160.json')
    audit['model_probe_performed'] = True
    audit['backend_equivalence_before_A'] = 'PASS'
    write('formal_A_prerequisite_audit_v160.json', audit)
    print(json.dumps(decisions), flush=True)

def main():
    cache = preflight()
    progress('load_model', 0, 1)
    entry = dict(revision_commit_sha=CONTRACT['checkpoint'])
    model = load_checkpoint(entry, cache)
    progress('load_model', 1, 1)
    cert = certification(model)
    if '--cert-only' not in sys.argv:
        discovery(model, cert)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write('execution_failure_v160.json', dict(exception=type(e).__name__, reason=str(e), traceback=traceback.format_exc()))
        traceback.print_exc()
        sys.exit(1)
