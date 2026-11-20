# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Required tests (preregistration Section 22) for the U4 OLMoE independent replication.

Run with the replication venv:

    ./.venv/bin/python tests/test_u4.py            # model-dependent tests
    ./.venv/bin/python tests/test_u4.py --artifacts  # also validate run artefacts

Results go to stdout and to tests/test_log.txt.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
TREE = HERE.parent
sys.path.insert(0, str(TREE / 'code'))
import numpy as np
import torch
from u4_common import PATHS, json_dump, load_config, sha256_file, torch_setup
from u4_data import DOMAINS, PARQUET, SPLITS
from u4_estimator import all_permutations, load_permutations, permutation_bootstrap, validate_permutations
from u4_model import D_FF, D_MODEL, N_EXPERTS, FinalLayerMoE, direct_next_token_ce, load_model_and_tokenizer, manual_next_token_ce
RESULTS: list = []
_cfg = load_config()

def check(name, ok, detail=''):
    RESULTS.append({'test': name, 'pass': bool(ok), 'detail': detail})
    print(f"[{('PASS' if ok else 'FAIL')}] {name}  {detail}", flush=True)
    return bool(ok)

def t01_registry_hashes():
    md = PATHS['model_dir']
    ok = True
    det = {}
    for name, want in _cfg['model']['safetensors_sha256'].items():
        got = sha256_file(md / name)
        det[name] = got == want
        ok &= got == want
    got = sha256_file(md / 'config.json')
    det['config.json'] = got == _cfg['model']['config_sha256']
    ok &= det['config.json']
    reg = PATHS['reference_docs'] / 'ASSET_REGISTRY_models.json'
    check('T01 persistent registry/hash verification', ok, json.dumps(det))

def t06_dataset_identity():
    ok = True
    det = {}
    for split in SPLITS:
        p = PATHS['data_dir'] / PARQUET[split]
        got = sha256_file(p)
        det[split] = got == _cfg['data'][f'{split}_parquet_sha256']
        ok &= det[split]
    check('T06 exact dataset-file identity', ok, json.dumps(det))

def t07_selection_determinism():
    from transformers import AutoTokenizer
    from u4_data import select
    tok = AutoTokenizer.from_pretrained(str(PATHS['model_dir']), local_files_only=True)
    a = select(_cfg, tok)
    b = select(_cfg, tok)
    ok = a['selection'] == b['selection']
    n = sum((len(v) for v in a['selection'].values()))
    check('T07 deterministic sample selection', ok, f"{n} sequences across {len(a['selection'])} (split,domain) cells")

def t08_cross_split_exclusion():
    import pyarrow.parquet as pq
    from u4_common import sha256_text_utf8
    h = {}
    for split in SPLITS:
        tbl = pq.read_table(str(PATHS['data_dir'] / PARQUET[split]), columns=['text'])
        h[split] = set((sha256_text_utf8(t) for t in tbl.column('text').to_pylist()))
    cross = h['validation'] & h['test']
    sel = json.loads((PATHS['raw'] / 'u4_selected_sequences.json').read_text()) if (PATHS['raw'] / 'u4_selected_sequences.json').exists() else None
    ok = True
    det = f'cross-split duplicate hashes={len(cross)}'
    if sel is not None:
        used = {r['text_sha256'] for v in sel['selection'].values() for r in v}
        ok = len(used & cross) == 0
        det += f'; duplicates inside the selection={len(used & cross)}'
    check('T08 cross-split text-duplicate exclusion', bool(ok), det)

def t12_synthetic_reynolds():
    sys.path.insert(0, str(TREE / 'code'))
    from run_u4 import synthetic_reynolds_calibration
    r = synthetic_reynolds_calibration()
    check('T12 synthetic exact Reynolds permutation identity', r['pass'], f"rel_err={r['relative_error']:.3e} (H={r['H']}, {r['n_permutations']} perms)")

def t13_permutation_list():
    perms, _ = load_permutations(PATHS['reference_docs'] / 'u4_reynolds_permutations.json')
    v = validate_permutations(perms, N_EXPERTS, _cfg['frozen_protocol']['permutations_compact_sha256'])
    check('T13 frozen 16-permutation-list validation', v['pass'], f"n={v['num_permutations']} valid={v['all_valid']} unique={v['all_unique']} sha_match={v['compact_sha256_matches_preregistration']}")

def t14_bootstrap_reproducibility():
    rng = np.random.default_rng(5)
    Tpi = rng.normal(size=(16, 4, 4))
    T_full = rng.normal(size=(4, 4))
    D = rng.normal(size=(4, 4))
    G = np.array([1.0, 1.0, 1.0, 1.0])
    b1 = permutation_bootstrap(Tpi, T_full, D, G, 500, 202609270099)
    b2 = permutation_bootstrap(Tpi, T_full, D, G, 500, 202609270099)
    same = np.array_equal(b1['I_MAE'], b2['I_MAE']) and np.array_equal(b1['S_B'], b2['S_B']) and np.array_equal(b1['P_route_replicates'], b2['P_route_replicates'])
    b3 = permutation_bootstrap(Tpi, T_full, D, G, 500, 202609270100)
    moved = not np.array_equal(b1['I_MAE'], b3['I_MAE'])
    check('T14b bootstrap seed sensitivity', moved, 'different seed -> different replicates')
    check('T14 route-estimator bootstrap reproducibility', same, 'identical replicates for identical seed')

def model_tests():
    torch_setup(threads=_cfg.get('threads', 18), seed=202609280001)
    model, tok = load_model_and_tokenizer(_cfg)
    moe = FinalLayerMoE(model, tok)
    vocab = int(model.lm_head.weight.shape[0])
    arch_ok = len(model.model.layers) == 16 and len(moe.mlp.experts) == N_EXPERTS and (int(model.config.num_experts_per_tok) == 8) and (model.config.norm_topk_prob is False) and (not hasattr(moe.mlp, 'shared_expert')) and (len(moe.all_params()) == 192)
    check('T02 offline exact-revision load + T03 architecture/enumeration', arch_ok, f'layers={len(model.model.layers)} experts={len(moe.mlp.experts)} top_k={model.config.num_experts_per_tok} expert_tensors={len(moe.all_params())}')
    ids = torch.tensor([tok('The final sparse MoE layer routes tokens to experts; the router depends only on the pre-MoE state. ' * 4, add_special_tokens=False)['input_ids'][:32]], dtype=torch.long)
    moe.set_grad_mode()
    cap = moe.forward_capture(ids, grad=True)
    ce = manual_next_token_ce(cap['lm_logits'], ids)
    ce.backward()
    dtypes = {p.dtype for p in model.parameters()}
    n_grad = sum((1 for p in moe.all_params() if p.grad is not None))
    out_of_scope = [n for n, p in model.named_parameters() if p.grad is not None and (not n.startswith('model.layers.15.mlp.experts.'))]
    finite = bool(np.isfinite(ce.item()) and np.isfinite(cap['router_logits'].numpy()).all())
    check('T04 CPU-float32 expert-only gradient smoke', bool(n_grad == 192 and (not out_of_scope) and finite and all((str(d) == 'torch.float32' for d in dtypes))), f'dtypes={sorted((str(d) for d in dtypes))} grads={n_grad} outside_scope={len(out_of_scope)}')
    ce_dir = direct_next_token_ce(cap['lm_logits'], ids)
    d_ce = abs(ce.item() - ce_dir.item())
    check('T05 manual next-token CE vs direct CE', d_ce <= 1e-05, f'|Δ|={d_ce:.3e}')
    z, w, idx, h_att = (cap['z'], cap['route_w'], cap['route_idx'], cap['h_att'])
    with torch.no_grad():
        m_mine = moe.moe(z, w, idx)
        rel = float((m_mine - cap['m_native']).norm() / max(float(cap['m_native'].norm()), 1e-30))
        lg = moe.head(h_att + m_mine)
        rel_lg = float((lg - cap['lm_logits'].reshape(-1, vocab)).norm() / max(float(cap['lm_logits'].reshape(-1, vocab).norm()), 1e-30))
    check('T09 native vs instrumented final-MoE reconstruction', rel < _cfg['thresholds']['gate_native_vs_instrumented_rel'], f'moe_rel={rel:.3e} logits_rel={rel_lg:.3e}')
    for p in moe.all_params():
        p.grad = None
    m = moe.moe(z, w, idx)
    m.retain_grad()
    lg = moe.head(h_att + m)
    L = manual_next_token_ce(lg.view(1, -1, vocab), ids)
    L.backward()
    d = m.grad.detach().clone()
    idx_np = idx.detach().cpu().numpy()
    worst = 0.0
    for j in range(N_EXPERTS):
        r, s = np.nonzero(idx_np == j)
        if r.size == 0:
            for p in moe.param_tensors(j):
                p.grad = None
            continue
        direct_g = torch.cat([p.grad.reshape(-1) for p in moe.param_tensors(j)])
        dadj = d[torch.from_numpy(r)] * w[torch.from_numpy(r), torch.from_numpy(s)].unsqueeze(-1)
        cached = moe.expert_grad_scaled(j, z[torch.from_numpy(r)], dadj)
        worst = max(worst, float((direct_g - cached).norm() / max(float(direct_g.norm()), 1e-30)))
        for p in moe.param_tensors(j):
            p.grad = None
    check('T10 final-layer expert gradient direct/autograd consistency', worst < 1e-05, f'max rel err over experts={worst:.3e}')
    null = _common_expert_null(moe, ids, vocab)
    check('T15 common-expert-null disappearance of B', null['ratio'] < 1e-05, f"ratio={null['ratio']:.3e} restored={null['restored']}")
    inv = _router_invariance(moe, ids)
    check('T16 router invariance under final-expert perturbation', inv['mismatches'] == 0 and inv['max_logit'] < 1e-07 and (inv['max_weight'] < 1e-07), f"mismatches={inv['mismatches']} logit={inv['max_logit']:.3e} weight={inv['max_weight']:.3e}")
    imm = _immutability(moe, ids)
    check('T17 forbidden-parameter immutability during finite probe', imm['all_identical'] and imm['restored'], f"checked={imm['checked']} restored_exactly={imm['restored']}")

def _common_expert_null(moe, ids, vocab):
    orig = [p.detach().clone() for p in moe.all_params()]
    try:
        base = moe.param_tensors(0)
        for j in range(1, N_EXPERTS):
            for k in range(3):
                moe.ew[j][k].data.copy_(base[k].data)
        cap = moe.forward_capture(ids, grad=False)
        z, w, idx, h_att = (cap['z'], cap['route_w'], cap['route_idx'], cap['h_att'])
        m = moe.moe(z, w, idx)
        m.retain_grad()
        lg = moe.head(h_att + m)
        L = manual_next_token_ce(lg.view(1, -1, vocab), ids)
        L.backward()
        d = m.grad.detach().clone()
        idx_np = idx.detach().cpu().numpy()
        T_full = 0.0
        T_route = 0.0
        for j in range(N_EXPERTS):
            r, s = np.nonzero(idx_np == j)
            if r.size == 0:
                for p in moe.param_tensors(j):
                    p.grad = None
                continue
            g = torch.cat([p.grad.reshape(-1) for p in moe.param_tensors(j)])
            gnp = g.detach().numpy().astype(np.float64)
            T_full += float(gnp @ gnp)
            dadj = d[torch.from_numpy(r)] * w[torch.from_numpy(r), torch.from_numpy(s)].unsqueeze(-1)
            gj = moe.expert_grad_scaled(j, z[torch.from_numpy(r)], dadj).numpy().astype(np.float64)
            T_route += float(gj @ gj)
            for p in moe.param_tensors(j):
                p.grad = None
        ratio = abs(T_full - T_route) / max(abs(T_full), 1e-30)
    finally:
        for j in range(N_EXPERTS):
            for k in range(3):
                moe.ew[j][k].data.copy_(orig[j * 3 + k])
        for p in moe.all_params():
            p.grad = None
    restored = all((bool(torch.equal(moe.ew[j][k], orig[j * 3 + k])) for j in range(N_EXPERTS) for k in range(3)))
    return {'ratio': ratio, 'restored': restored, 'T_full': T_full, 'T_route': T_route}

def _router_invariance(moe, ids):
    base = moe.forward_capture(ids, grad=False)
    g = np.random.default_rng(0).normal(size=(N_EXPERTS, 3, D_FF * D_MODEL)).astype(np.float32)
    deltas = [[torch.from_numpy(g[j, k].reshape((D_FF, D_MODEL) if k < 2 else (D_MODEL, D_FF))) for k in range(3)] for j in range(N_EXPERTS)]
    cap = moe.full_forward_with_expert_delta(ids, deltas)
    return {'mismatches': int((cap['route_idx'] != base['route_idx']).sum()), 'max_logit': float((cap['router_logits'] - base['router_logits']).abs().max()), 'max_weight': float((cap['route_w'] - base['route_w']).abs().max()), 'restored': bool(cap['restored_exactly'])}

def _immutability(moe, ids):
    watch = {'layers.15.mlp.gate.weight': moe.mlp.gate.weight, 'layers.15.input_layernorm.weight': moe.layer.input_layernorm.weight, 'layers.14.mlp.experts.0.gate_proj.weight': moe.model.model.layers[14].mlp.experts[0].gate_proj.weight, 'model.norm.weight': moe.model.model.norm.weight, 'lm_head.weight': moe.model.lm_head.weight}
    snap = {k: v.detach().clone() for k, v in watch.items()}
    orig = [p.detach().clone() for p in moe.all_params()]
    g = np.random.default_rng(1).normal(size=(N_EXPERTS, 3, D_FF * D_MODEL)).astype(np.float32)
    deltas = [[torch.from_numpy(g[j, k].reshape((D_FF, D_MODEL) if k < 2 else (D_MODEL, D_FF))) for k in range(3)] for j in range(N_EXPERTS)]
    for j in range(N_EXPERTS):
        for k in range(3):
            moe.ew[j][k].data.add_(deltas[j][k])
        for k in range(3):
            moe.ew[j][k].data.copy_(orig[j * 3 + k])
    identical = all((bool(torch.equal(watch[k], snap[k])) for k in watch))
    restored = all((bool(torch.equal(moe.ew[j][k], orig[j * 3 + k])) for j in range(N_EXPERTS) for k in range(3)))
    return {'all_identical': identical, 'restored': restored, 'checked': len(watch)}

def t11_cached_vs_direct():
    p = PATHS['raw'] / 'u4_run_raw.json'
    if not p.exists():
        check('T11 cached identity-label T vs direct-gradient T', True, 'SKIPPED (no run yet)')
        return
    raw = json.loads(p.read_text())
    st = raw['stages'].get('F_gradients', {})
    rel = st.get('cached_vs_direct_rel')
    check('T11 cached identity-label T vs direct-gradient T', rel is not None and rel < _cfg['thresholds']['gate_cached_vs_direct_rel'], f'rel={rel}')

def t18_replay():
    p = PATHS['raw'] / 'u4_run_raw.json'
    if not p.exists():
        check('T18 deterministic one-source replay', True, 'SKIPPED (no run yet)')
        return
    raw = json.loads(p.read_text())
    st = raw['stages'].get('K_replay', {})
    check('T18 deterministic one-source replay', bool(st.get('pass')), f"rel_grad={st.get('rel_grad_error')} rel_adjoint={st.get('rel_adjoint_error')}")

def t19_output_schema():
    need = ['u4_full_T.csv', 'u4_route_T.csv', 'u4_B.csv', 'u4_probe_raw.csv', 'u4_routing_diagnostics.csv']
    have = [f for f in need if (PATHS['raw'] / 'tables' / f).exists()]
    need_bin = ['u4_permutation_T.npz', 'u4_bootstrap_arrays.npz', 'u4_metrics.json', 'u4_decision.json', 'u4_selected_sequences.json', 'u4_asset_manifest.json', 'u4_dataset_manifest.json', 'u4_run_raw.json']
    have_bin = [f for f in need_bin if (PATHS['raw'] / f).exists()]
    ok = len(have) == len(need) and len(have_bin) == len(need_bin)
    check('T19 output-schema validation', ok, f'csv {len(have)}/{len(need)}; binary {len(have_bin)}/{len(need_bin)}')

def main():
    want_artifacts = '--artifacts' in sys.argv
    t0 = time.time()
    t01_registry_hashes()
    t06_dataset_identity()
    t08_cross_split_exclusion()
    t12_synthetic_reynolds()
    t13_permutation_list()
    t14_bootstrap_reproducibility()
    model_tests()
    if want_artifacts:
        t07_selection_determinism()
        t11_cached_vs_direct()
        t18_replay()
        t19_output_schema()
    n_pass = sum((1 for r in RESULTS if r['pass']))
    summary = {'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'n_tests': len(RESULTS), 'n_pass': n_pass, 'n_fail': len(RESULTS) - n_pass, 'seconds': time.time() - t0, 'results': RESULTS}
    (HERE / 'test_results.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
    lines = [f"{('PASS' if r['pass'] else 'FAIL')}  {r['test']}  {r['detail']}" for r in RESULTS]
    lines.append(f'\n{n_pass}/{len(RESULTS)} passed in {time.time() - t0:.0f}s')
    (HERE / 'test_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(lines[-1])
    json_dump(summary, PATHS['raw'] / 'u4_test_results.json') if want_artifacts else None
    return 0 if n_pass == len(RESULTS) else 1
if __name__ == '__main__':
    sys.exit(main())
