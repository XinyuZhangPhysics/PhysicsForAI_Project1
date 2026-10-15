# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""U4 OLMoE final-layer sparse-MoE — scientific run (independent replication).

Order of operations follows the frozen preregistration:

  A. asset / data / architecture gates          (Sec. 4, 5, 21)
  B. synthetic exact Reynolds calibration       (Sec. 19)      <- before any scientific metric
  C. frozen permutation-list validation         (Sec. 19)
  D. deterministic sample construction          (Sec. 6)
  E. layer-15 cache + reconstruction gate       (Sec. 9)
  F. direct full gradients, cached reproduction (Sec. 8, 9)
  G. common-expert null                         (Sec. 18)
  H. permutation Reynolds estimator             (Sec. 10)
  I. finite-difference probe                    (Sec. 12)
  J. routing-stability audit                    (Sec. 13)
  K. bootstrap, C1-C5, node decision            (Sec. 11, 14-17)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from u4_common import PATHS, ensure_dirs, json_dump, log, load_config, now_utc, sha256_file, torch_setup, write_csv
from u4_data import DOMAINS, PARQUET, SPLITS, select
from u4_model import D_FF, D_MODEL, N_EXPERTS, FinalLayerMoE, load_model_and_tokenizer, manual_next_token_ce
from u4_estimator import OFFDIAG, SET_KEYS, TEST_KEYS, VAL_KEYS, all_permutations, compact_permutation_sha256, load_permutations, percentile_ci, permutation_bootstrap, validate_permutations
SMOKE_TEXT = 'The quick brown fox jumps over the lazy dog. Attention is all you need; experts route tokens to specialists. ' * 8
VOCAB = None

def _rows_slots(route_idx, n_experts):
    """For every label c: (token rows routed to c, slot position in the top-k list)."""
    idx = route_idx.detach().cpu().numpy()
    out = []
    for c in range(n_experts):
        r, s = np.nonzero(idx == c)
        out.append((torch.from_numpy(r.astype(np.int64)), torch.from_numpy(s.astype(np.int64))))
    return out

def _expert_vector(moe, j, z, d, w, rows, slot):
    dadj = d[rows] * w[rows, slot].unsqueeze(-1)
    return moe.expert_grad_scaled(j, z[rows], dadj)

def _flat_blocks(moe, j):
    """Concatenated expert-j parameter gradient.

    An expert that is never routed on this distribution does not enter the autograd
    graph, so its gradient is exactly zero (p.grad is None).
    """
    parts = []
    for p in moe.param_tensors(j):
        parts.append(p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=p.dtype))
    return torch.cat(parts)

class SyntheticMoE:
    """Tiny H-expert SwiGLU MoE used only for the exact Reynolds calibration (Sec. 19)."""

    def __init__(self, H, d_model, d_ff, seed=7):
        g = np.random.default_rng(seed)
        self.H = H
        self.d_model = d_model
        self.d_ff = d_ff
        sc = 1.0 / np.sqrt(d_model)
        self.W = [[torch.tensor(g.normal(0, sc, (d_ff, d_model)), dtype=torch.float64), torch.tensor(g.normal(0, sc, (d_ff, d_model)), dtype=torch.float64), torch.tensor(g.normal(0, sc, (d_model, d_ff)), dtype=torch.float64)] for _ in range(H)]
        for j in range(H):
            for k in range(3):
                self.W[j][k].requires_grad_(True)

    def param_tensors(self, j):
        return self.W[j]

    def out(self, j, z):
        Wg, Wu, Wd = self.W[j]
        return torch.nn.functional.silu(z @ Wg.t()) * (z @ Wu.t()) @ Wd.t()

    def expert_grad_scaled(self, j, z, dadj):
        ps = self.W[j]
        out = self.out(j, z)
        s = (out * dadj).sum()
        gs = torch.autograd.grad(s, ps)
        return torch.cat([g.reshape(-1) for g in gs]).detach()

def synthetic_reynolds_calibration():
    """Enumerate all H! permutations and compare with the analytic mean-kernel route response."""
    torch.manual_seed(11)
    H, d_model, d_ff, n_tok, k = (4, 6, 5, 8, 2)
    syn = SyntheticMoE(H, d_model, d_ff)
    z = torch.randn(n_tok, d_model, dtype=torch.float64)
    d = torch.randn(n_tok, d_model, dtype=torch.float64)
    logits = torch.randn(n_tok, H, dtype=torch.float64)
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    a = torch.zeros(n_tok, H, dtype=torch.float64)
    a.scatter_(1, idx, w)
    P = 3 * d_ff * d_model
    U = np.zeros((H, n_tok, P), dtype=np.float64)
    for j in range(H):
        for r in range(n_tok):
            zr = z[r:r + 1]
            dr = d[r:r + 1]
            out = syn.out(j, zr)
            s = (out * dr).sum()
            gs = torch.autograd.grad(s, syn.param_tensors(j))
            U[j, r] = torch.cat([g.reshape(-1) for g in gs]).detach().numpy()
    C = a.numpy() @ a.numpy().T
    analytic = 0.0
    for j in range(H):
        G = U[j] @ U[j].T
        analytic += float((C * G).sum())
    analytic /= n_tok ** 2 * H
    perms = all_permutations(H)
    sets = [{'z': z, 'd': d, 'route_w': w, 'route_idx': idx, 'N': float(n_tok)}]
    rs = _rows_slots(idx, H)
    N = float(n_tok)
    acc = []
    for pi in perms:
        inv = np.argsort(pi)
        tot = 0.0
        for j in range(H):
            c = int(inv[j])
            rows, slot = rs[c]
            g = syn.expert_grad_scaled(j, z[rows], d[rows] * w[rows, slot].unsqueeze(-1))
            gj = g.numpy().astype(np.float64) / N
            tot += float(gj @ gj)
        acc.append(tot)
    exact = float(np.mean(acc))
    rel = abs(exact - analytic) / max(abs(analytic), 1e-300)
    return {'H': H, 'd_model': d_model, 'd_ff': d_ff, 'n_tokens': n_tok, 'top_k': k, 'n_permutations': int(perms.shape[0]), 'exact_permutation_average': exact, 'analytic_mean_kernel_route': analytic, 'relative_error': rel, 'threshold': 1e-10, 'pass': bool(rel < 1e-10), 'definition': "analytic = (1/(N^2 H)) sum_j sum_{r,r'} C[r,r'] <u_j(r), u_j(r')>, C = A A^T, u_j(r) = J_j(z_r)^T d_r"}

def main() -> None:
    ensure_dirs()
    cfg = load_config()
    thr = cfg['thresholds']
    torch_setup(threads=cfg.get('threads', 18), seed=202609280001)
    t_start = time.time()
    raw: dict = {'generated_utc': now_utc(), 'stages': {}}
    grad_dir = PATHS['raw'] / 'grads'
    grad_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PATHS['raw'] / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    md = PATHS['model_dir']
    asset = {}
    for name, want in cfg['model']['safetensors_sha256'].items():
        got = sha256_file(md / name)
        asset[name] = {'sha256': got, 'match': got == want, 'bytes': (md / name).stat().st_size}
    got = sha256_file(md / 'config.json')
    asset['config.json'] = {'sha256': got, 'match': got == cfg['model']['config_sha256'], 'bytes': (md / 'config.json').stat().st_size}
    for f in ('generation_config.json', 'model.safetensors.index.json', 'tokenizer.json'):
        p = md / f
        if p.exists():
            asset[f] = {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
    dmanifest = {}
    for split in SPLITS:
        p = PATHS['data_dir'] / PARQUET[split]
        dg = sha256_file(p)
        dmanifest[split] = {'path': str(p), 'file': PARQUET[split], 'sha256': dg, 'match': dg == cfg['data'][f'{split}_parquet_sha256'], 'bytes': p.stat().st_size}
    raw['stages']['A_asset_data'] = {'model': asset, 'data': dmanifest, 'asset_ok': all((v.get('match', True) for v in asset.values())), 'data_ok': all((v['match'] for v in dmanifest.values())), 'pass': bool(all((v.get('match', True) for v in asset.values())) and all((v['match'] for v in dmanifest.values())))}
    json_dump({'model': cfg['model'], 'files': asset}, PATHS['raw'] / 'u4_asset_manifest.json')
    json_dump({'dataset': cfg['data'], 'files': dmanifest}, PATHS['raw'] / 'u4_dataset_manifest.json')
    log('stage A done')
    perms, inv_perms = load_permutations(PATHS['reference_docs'] / 'u4_reynolds_permutations.json')
    perm_val = validate_permutations(perms, N_EXPERTS, cfg['frozen_protocol']['permutations_compact_sha256'])
    synth = synthetic_reynolds_calibration()
    raw['stages']['B_calibration'] = {'permutations': perm_val, 'synthetic_reynolds': synth, 'pass': bool(perm_val['pass'] and synth['pass'])}
    log(f"stage B done: synthetic rel err {synth['relative_error']:.3e} perm_ok={perm_val['pass']}")
    model, tok = load_model_and_tokenizer(cfg)
    moe = FinalLayerMoE(model, tok)
    global VOCAB
    VOCAB = int(model.lm_head.weight.shape[0])
    log(f'vocab={VOCAB}')
    sel = select(cfg, tok)
    sel_replay = select(cfg, tok)
    same = sel['selection'] == sel_replay['selection']
    selection = sel['selection']
    json_dump({'domains': DOMAINS, 'splits': SPLITS, 'selection': selection, 'diagnostics': sel['diagnostics'], 'deterministic_replay_identical': bool(same)}, PATHS['raw'] / 'u4_selected_sequences.json')
    alt = _alt_dedup_selection(cfg, tok)
    alt_same = all((alt[f'{s}|{d}'][i]['token_ids'] == selection[f'{s}|{d}'][i]['token_ids'] for s in SPLITS for d in DOMAINS for i in range(16)))
    raw['stages']['D_data'] = {'diagnostics': sel['diagnostics'], 'deterministic_replay_identical': bool(same), 'alt_dedup_convention_same_selection': bool(alt_same), 'pass': bool(same)}
    log(f'stage D done: replay_identical={same} alt_dedup_same={alt_same}')
    sets = {}
    for key in SET_KEYS:
        split, domain = key.split('|')
        ids = torch.tensor([r['token_ids'] for r in selection[key]], dtype=torch.long)
        cap = moe.forward_capture(ids, grad=False)
        z, w, idx, h_att = (cap['z'], cap['route_w'], cap['route_idx'], cap['h_att'])
        with torch.no_grad():
            m_mine = moe.moe(z, w, idx)
            m_nat = cap['m_native']
            rel_m = float((m_mine - m_nat).norm() / max(float(m_nat.norm()), 1e-30))
            lg_mine = moe.head(h_att + m_mine)
            rel_lg = float((lg_mine - cap['lm_logits'].reshape(-1, VOCAB)).norm() / max(float(cap['lm_logits'].reshape(-1, VOCAB).norm()), 1e-30))
            rl_mine = z @ moe.gate_w.t()
            rel_rl = float((rl_mine - cap['router_logits']).norm() / max(float(cap['router_logits'].norm()), 1e-30))
        sets[key] = {'ids': ids, 'z': z, 'route_w': w, 'route_idx': idx, 'h_att': h_att, 'rs': _rows_slots(idx, N_EXPERTS), 'm_native': m_nat, 'router_logits': cap['router_logits'], 'lm_logits': cap['lm_logits'].reshape(-1, VOCAB), 'recon': {'moe_output_rel_error': rel_m, 'lm_logits_rel_error': rel_lg, 'router_logits_rel_error': rel_rl, 'pass': bool(rel_m < thr['gate_native_vs_instrumented_rel'])}}
        np.savez(cache_dir / f"cache_{key.replace('|', '_')}.npz", z=z.numpy(), h_att=h_att.numpy(), route_w=w.numpy(), route_idx=idx.numpy(), ids=ids.numpy())
        log(f'  cached {key}: recon_rel={rel_m:.3e} logits_rel={rel_lg:.3e}')
    raw['stages']['E_cache'] = {'per_set': {k: v['recon'] for k, v in sets.items()}, 'pass': all((v['recon']['pass'] for v in sets.values()))}
    moe.set_grad_mode()
    P = 3 * D_FF * D_MODEL
    direct = {}
    for key in SET_KEYS:
        s = sets[key]
        for p in moe.all_params():
            p.grad = None
        m = moe.moe(s['z'], s['route_w'], s['route_idx'])
        m.retain_grad()
        lg = moe.head(s['h_att'] + m)
        L = manual_next_token_ce(lg.view(16, 128, VOCAB), s['ids'])
        L.backward()
        d = m.grad.detach().clone()
        s['d'] = d
        s['loss'] = float(L.detach())
        G = np.zeros((N_EXPERTS, P), dtype=np.float32)
        for j in range(N_EXPERTS):
            G[j] = _flat_blocks(moe, j).numpy()
            for p in moe.param_tensors(j):
                p.grad = None
        direct[key] = G
        np.save(grad_dir / f"direct_{key.replace('|', '_')}.npy", G)
        np.save(cache_dir / f"d_{key.replace('|', '_')}.npy", d.numpy())
        log(f'  direct grads {key}: loss={float(L.detach()):.6f} |d|={float(d.norm()):.4e}')
    del m, lg, L
    T_direct = np.zeros((8, 8), dtype=np.float64)
    for j in range(N_EXPERTS):
        M = np.stack([direct[k][j] for k in SET_KEYS]).astype(np.float64)
        T_direct += M @ M.T
        del M
    G_norm = {}
    for k in SET_KEYS:
        acc = 0.0
        for j in range(N_EXPERTS):
            v = direct[k][j].astype(np.float64)
            acc += float(v @ v)
        G_norm[k] = float(np.sqrt(acc))
    Uc = np.zeros((8, P), dtype=np.float64)
    T_cached = np.zeros((8, 8), dtype=np.float64)
    for j in range(N_EXPERTS):
        for si, key in enumerate(SET_KEYS):
            s = sets[key]
            rows, slot = s['rs'][j]
            Uc[si] = _expert_vector(moe, j, s['z'], s['d'], s['route_w'], rows, slot).numpy().astype(np.float64)
        T_cached += Uc @ Uc.T
    rel_cached = float(np.linalg.norm(T_cached - T_direct) / max(np.linalg.norm(T_direct), 1e-30))
    raw['stages']['F_gradients'] = {'loss_per_set': {k: sets[k]['loss'] for k in SET_KEYS}, 'G_s': {k: G_norm[k] for k in SET_KEYS}, 'G_s_min': float(min(G_norm.values())), 'T_direct': T_direct.tolist(), 'T_cached': T_cached.tolist(), 'cached_vs_direct_rel': rel_cached, 'threshold_cached_vs_direct': thr['gate_cached_vs_direct_rel'], 'pass': bool(rel_cached < thr['gate_cached_vs_direct_rel'] and min(G_norm.values()) >= cfg['response']['G_s_invalid_threshold'])}
    log(f'stage F done: cached-vs-direct rel={rel_cached:.3e}')
    null = common_expert_null(moe, cfg)
    raw['stages']['G_common_expert_null'] = null
    log(f"stage G done: null ratio={null['ratio']:.3e} pass={null['pass']}")
    T_pi = np.zeros((perms.shape[0], 8, 8), dtype=np.float64)
    U = np.zeros((8, P), dtype=np.float64)
    t0 = time.time()
    for b in range(perms.shape[0]):
        inv = inv_perms[b]
        for j in range(N_EXPERTS):
            c = int(inv[j])
            for si, key in enumerate(SET_KEYS):
                s = sets[key]
                rows, slot = s['rs'][c]
                U[si] = _expert_vector(moe, j, s['z'], s['d'], s['route_w'], rows, slot).numpy().astype(np.float64)
            T_pi[b] += U @ U.T
        log(f'  perm {b + 1}/{perms.shape[0]} ({time.time() - t0:.0f}s)')
    T_route_hat = T_pi.mean(axis=0)
    raw['stages']['H_estimator'] = {'seconds': time.time() - t0, 'T_route_hat': T_route_hat.tolist(), 'all_finite': bool(np.isfinite(T_route_hat).all()), 'pass': bool(np.isfinite(T_route_hat).all())}
    np.savez(PATHS['raw'] / 'u4_permutation_T.npz', T_pi=T_pi, permutations=perms, set_keys=np.array(SET_KEYS))
    log(f'stage H done in {time.time() - t0:.0f}s')
    Theta = 0.0
    for j in range(N_EXPERTS):
        for p in moe.param_tensors(j):
            Theta += float((p.detach().numpy().astype(np.float64) ** 2).sum())
    Theta = float(np.sqrt(Theta))
    radii = cfg['radii']['set']
    dirs = {}
    dir_t = {}
    for sk in VAL_KEYS:
        g = direct[sk].astype(np.float64)
        dvec = (-g / G_norm[sk]).astype(np.float32).reshape(N_EXPERTS, 3, -1)
        dirs[sk] = dvec
        dir_t[sk] = [[torch.from_numpy(dvec[j, k].reshape((D_FF, D_MODEL) if k < 2 else (D_MODEL, D_FF))) for k in range(3)] for j in range(N_EXPERTS)]
    probe = {r: np.zeros((4, 4), dtype=np.float64) for r in radii}
    probe_rows = []
    t0 = time.time()
    for r in radii:
        eps = r * Theta
        for sj, sk in enumerate(VAL_KEYS):
            losses = {}
            for sign in (+1.0, -1.0):
                deltas = [[eps * sign * dir_t[sk][j][k] for k in range(3)] for j in range(N_EXPERTS)]
                with torch.no_grad():
                    for ti, tk in enumerate(TEST_KEYS):
                        s = sets[tk]
                        m = moe.moe(s['z'], s['route_w'], s['route_idx'], deltas)
                        lg = moe.head(s['h_att'] + m)
                        losses[sign, ti] = float(manual_next_token_ce(lg.view(16, 128, VOCAB), s['ids']))
            for ti in range(4):
                probe[r][ti, sj] = (losses[+1.0, ti] - losses[-1.0, ti]) / (2.0 * eps)
            log(f'  rho={r:g} source={sk}: D={np.round(probe[r][:, sj], 6).tolist()}')
            for ti in range(4):
                probe_rows.append([r, TEST_KEYS[ti], VAL_KEYS[sj], losses[+1.0, ti], losses[-1.0, ti], probe[r][ti, sj]])
    raw['stages']['I_probe'] = {'Theta': Theta, 'seconds': time.time() - t0, 'eps': {str(r): r * Theta for r in radii}, 'D_probe': {str(r): probe[r].tolist() for r in radii}, 'all_finite': bool(all((np.isfinite(probe[r]).all() for r in radii))), 'pass': bool(all((np.isfinite(probe[r]).all() for r in radii)))}
    write_csv(probe_rows, PATHS['raw'] / 'tables' / 'u4_probe_raw.csv', ['rho', 'target_set', 'source_set', 'L_plus', 'L_minus', 'D_probe'])
    log(f'stage I done in {time.time() - t0:.0f}s')
    audit = routing_stability_audit(moe, sets, dirs, Theta, cfg)
    raw['stages']['J_router_audit'] = audit
    log(f"stage J done: mismatches={audit['total_route_index_mismatches']} max_logit_delta={audit['max_router_logit_abs_delta']:.3e} max_weight_delta={audit['max_router_weight_abs_delta']:.3e} pass={audit['pass']}")
    T_full_4 = T_direct[0:4, 4:8]
    T_route_4 = T_route_hat[0:4, 4:8]
    D_prim = probe[cfg['radii']['primary']]
    G_src = np.array([G_norm[k] for k in VAL_KEYS], dtype=np.float64)
    boot = permutation_bootstrap(T_pi[:, 0:4, 4:8], T_full_4, D_prim, G_src, cfg['bootstrap']['n_replicates'], cfg['bootstrap']['seed'])
    np.savez(PATHS['raw'] / 'u4_bootstrap_arrays.npz', I_MAE=boot['I_MAE'], S_B=boot['S_B'], P_route_replicates=boot['P_route_replicates'])
    metrics = compute_all(T_full_4, T_route_4, D_prim, probe, boot, G_src, cfg)
    decision = decide(metrics, raw, cfg)
    metrics['node_decision'] = decision
    json_dump(metrics, PATHS['raw'] / 'u4_metrics.json')
    json_dump({'node_decision': decision, 'gates': metrics['gates'], 'generated_utc': now_utc()}, PATHS['raw'] / 'u4_decision.json')
    _write_tables(T_direct, T_route_hat, T_direct - T_route_hat, SET_KEYS)
    diag = routing_diagnostics(sets)
    write_csv(diag['rows'], PATHS['raw'] / 'tables' / 'u4_routing_diagnostics.csv', diag['header'])
    raw['routing_diagnostics'] = diag['summary']
    replay = one_source_replay(moe, sets, direct, dirs, Theta, cfg)
    raw['stages']['K_replay'] = replay
    raw['total_seconds'] = time.time() - t_start
    json_dump(raw, PATHS['raw'] / 'u4_run_raw.json')
    log(json.dumps({k: v.get('pass', None) for k, v in raw['stages'].items()}, indent=1))
    log(f'NODE DECISION: {decision}  (total {time.time() - t_start:.0f}s)')

def _alt_dedup_selection(cfg, tok):
    """Secondary declared diagnostic: 'keep one copy' instead of 'drop both'."""
    import hashlib
    import pyarrow.parquet as pq
    from u4_data import DOMAINS, PARQUET, SPLITS, k_row
    data = {}
    for split in SPLITS:
        tbl = pq.read_table(str(PATHS['data_dir'] / PARQUET[split]), columns=['text', 'meta', '__index_level_0__'])
        texts = tbl.column('text').to_pylist()
        sets_ = [m['redpajama_set_name'] for m in tbl.column('meta').to_pylist()]
        idxs = tbl.column('__index_level_0__').to_pylist()
        data[split] = (texts, sets_, idxs)
    hval = {s: [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in data[s][0]] for s in SPLITS}
    seen = set()
    out = {}
    for split in SPLITS:
        texts, sets_, idxs = data[split]
        for domain in DOMAINS:
            cand = sorted(((k_row(split, domain, idxs[i], texts[i]), i) for i in range(len(texts)) if sets_[i] == domain), key=lambda t: t[0])
            elig = []
            for k, i in cand:
                h = hval[split][i]
                if h in seen:
                    continue
                seen.add(h)
                ids = tok(texts[i], add_special_tokens=False)['input_ids']
                if len(ids) >= 128:
                    elig.append(ids[:128])
                if len(elig) >= 16:
                    break
            out[f'{split}|{domain}'] = [{'token_ids': e} for e in elig]
    return out

def common_expert_null(moe, cfg):
    """Sec. 18: identical experts => B = 0 exactly."""
    import copy
    orig = [p.detach().clone() for p in moe.all_params()]
    digest_before = moe.params_digest()
    ids = torch.tensor([moe.tok(SMOKE_TEXT, add_special_tokens=False)['input_ids'][:32]], dtype=torch.long)
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
        L = manual_next_token_ce(lg.view(1, -1, VOCAB), ids)
        L.backward()
        d = m.grad.detach().clone()
        rs = _rows_slots(idx, N_EXPERTS)
        P = 3 * D_FF * D_MODEL
        T_full = 0.0
        for j in range(N_EXPERTS):
            g = _flat_blocks(moe, j).numpy().astype(np.float64)
            T_full += float(g @ g)
            for p in moe.param_tensors(j):
                p.grad = None
        perms, inv_perms = load_permutations(PATHS['reference_docs'] / 'u4_reynolds_permutations.json')
        vals = []
        for b in range(perms.shape[0]):
            inv = inv_perms[b]
            tot = 0.0
            for j in range(N_EXPERTS):
                c = int(inv[j])
                rows, slot = rs[c]
                if rows.numel() == 0:
                    continue
                g = _expert_vector(moe, j, z, d, w, rows, slot).numpy().astype(np.float64)
                tot += float(g @ g)
            vals.append(tot)
        T_route = float(np.mean(vals))
        ratio = abs(T_full - T_route) / max(abs(T_full), 1e-30)
    finally:
        for j in range(N_EXPERTS):
            for k in range(3):
                moe.ew[j][k].data.copy_(orig[j * 3 + k])
        for p in moe.all_params():
            p.grad = None
    restored = all((bool(torch.equal(moe.ew[j][k], orig[j * 3 + k])) for j in range(N_EXPERTS) for k in range(3)))
    return {'T_full_null': T_full, 'T_route_null': T_route, 'ratio': ratio, 'threshold': cfg['thresholds']['gate_common_expert_null_rel'], 'per_permutation_T_route_null': vals, 'permutation_spread': float(np.max(vals) - np.min(vals)), 'params_restored_exactly': bool(restored), 'digest_before': digest_before, 'digest_after': moe.params_digest(), 'pass': bool(ratio < cfg['thresholds']['gate_common_expert_null_rel'] and restored)}

def routing_stability_audit(moe, sets, dirs, Theta, cfg):
    """Sec. 13: end-to-end full-model audit at rho = 2e-5 on every selected sequence."""
    eps = cfg['radii']['max'] * Theta
    mismatches = 0
    max_logit = 0.0
    max_weight = 0.0
    per_state = []
    for sk in VAL_KEYS:
        dvec = dirs[sk]
        for sign in (+1.0, -1.0):
            deltas = [[] for _ in range(N_EXPERTS)]
            for j in range(N_EXPERTS):
                for k in range(3):
                    deltas[j].append(torch.from_numpy(eps * sign * dvec[j, k].reshape((D_FF, D_MODEL) if k < 2 else (D_MODEL, D_FF))))
            n_mm = 0
            ml = 0.0
            mw = 0.0
            for key in SET_KEYS:
                s = sets[key]
                cap = moe.full_forward_with_expert_delta(s['ids'], deltas)
                n_mm += int((cap['route_idx'] != s['route_idx']).sum())
                ml = max(ml, float((cap['router_logits'] - s['router_logits']).abs().max()))
                mw = max(mw, float((cap['route_w'] - s['route_w']).abs().max()))
                assert cap['restored_exactly'], 'parameter restore failed'
            mismatches += n_mm
            max_logit = max(max_logit, ml)
            max_weight = max(max_weight, mw)
            per_state.append({'source': sk, 'sign': sign, 'route_index_mismatches': int(n_mm), 'max_router_logit_abs_delta': ml, 'max_router_weight_abs_delta': mw, 'params_restored_exactly': True})
    return {'rho': cfg['radii']['max'], 'eps': eps, 'states': per_state, 'total_route_index_mismatches': int(mismatches), 'max_router_logit_abs_delta': max_logit, 'max_router_weight_abs_delta': max_weight, 'thresholds': [cfg['thresholds']['gate_router_logit_abs'], cfg['thresholds']['gate_router_weight_abs']], 'pass': bool(mismatches == 0 and max_logit < cfg['thresholds']['gate_router_logit_abs'] and (max_weight < cfg['thresholds']['gate_router_weight_abs']))}

def routing_diagnostics(sets):
    rows = []
    summary = {}
    for key in SET_KEYS:
        idx = sets[key]['route_idx'].numpy()
        w = sets[key]['route_w'].numpy()
        n = idx.shape[0]
        freq = np.bincount(idx.ravel(), minlength=N_EXPERTS) / n
        wsum = np.bincount(idx.ravel(), weights=w.ravel(), minlength=N_EXPERTS)
        cnt = np.bincount(idx.ravel(), minlength=N_EXPERTS).astype(np.float64)
        mean_w = np.divide(wsum, cnt, out=np.zeros_like(wsum), where=cnt > 0)
        wn = w / w.sum(axis=1, keepdims=True)
        ent = float(-(wn * np.log(wn + 1e-30)).sum(axis=1).mean())
        distinct = int(np.unique(idx).size)
        summary[key] = {'top8_route_entropy': ent, 'distinct_experts': distinct, 'weight_sum_mean': float(w.sum(axis=1).mean()), 'mean_selected_weight_overall': float(w.mean()), 'selection_freq_min': float(freq.min()), 'selection_freq_max': float(freq.max())}
        for e in range(N_EXPERTS):
            rows.append([key, e, freq[e], mean_w[e], cnt[e]])
    for a in range(len(SET_KEYS)):
        for b in range(a + 1, len(SET_KEYS)):
            A = sets[SET_KEYS[a]]['route_idx'].numpy()
            B = sets[SET_KEYS[b]]['route_idx'].numpy()
            n = min(A.shape[0], B.shape[0])
            ov = np.mean([len(set(A[i].tolist()) & set(B[i].tolist())) for i in range(n)])
            summary[f'overlap|{SET_KEYS[a]}|{SET_KEYS[b]}'] = float(ov)
    return {'rows': rows, 'header': ['set', 'expert', 'selection_frequency', 'mean_selected_weight', 'n_selections'], 'summary': summary}

def one_source_replay(moe, sets, direct, dirs, Theta, cfg):
    """Sec. 21.16: deterministic replay of one source full gradient and one probe cell."""
    key = VAL_KEYS[0]
    s = sets[key]
    for p in moe.all_params():
        p.grad = None
    m = moe.moe(s['z'], s['route_w'], s['route_idx'])
    m.retain_grad()
    lg = moe.head(s['h_att'] + m)
    L = manual_next_token_ce(lg.view(16, 128, VOCAB), s['ids'])
    L.backward()
    d2 = m.grad.detach().clone()
    G2 = np.zeros((N_EXPERTS, 3 * D_FF * D_MODEL), dtype=np.float32)
    for j in range(N_EXPERTS):
        G2[j] = _flat_blocks(moe, j).numpy()
        for p in moe.param_tensors(j):
            p.grad = None
    rel_grad = float(np.linalg.norm(G2.astype(np.float64) - direct[key].astype(np.float64)) / max(np.linalg.norm(direct[key].astype(np.float64)), 1e-30))
    rel_d = float((d2 - s['d']).norm() / max(float(s['d'].norm()), 1e-30))
    eps = cfg['radii']['primary'] * Theta
    tgt = sets[TEST_KEYS[0]]
    vals = []
    dvec = dirs[key]
    with torch.no_grad():
        for sign in (+1.0, -1.0):
            deltas = [[torch.from_numpy(eps * sign * dvec[j, k].reshape((D_FF, D_MODEL) if k < 2 else (D_MODEL, D_FF))) for k in range(3)] for j in range(N_EXPERTS)]
            mm = moe.moe(tgt['z'], tgt['route_w'], tgt['route_idx'], deltas)
            lgt = moe.head(tgt['h_att'] + mm)
            vals.append(float(manual_next_token_ce(lgt.view(16, 128, VOCAB), tgt['ids'])))
    return {'replayed_set': key, 'rel_grad_error': rel_grad, 'rel_adjoint_error': rel_d, 'probe_cell_losses': vals, 'probe_cell_D': (vals[0] - vals[1]) / (2 * eps), 'threshold': cfg['thresholds']['gate_replay_rel'], 'pass': bool(rel_grad <= cfg['thresholds']['gate_replay_rel'] and rel_d <= cfg['thresholds']['gate_replay_rel'])}

def compute_all(T_full_4, T_route_4, D_prim, probe, boot, G_src, cfg):
    thr = cfg['thresholds']
    Pfull = -T_full_4 / G_src[None, :]
    Proute = -T_route_4 / G_src[None, :]
    PB = Pfull - Proute
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    d_off = np.array([D_prim[i, j] for i, j in OFFDIAG])
    ef = np.array([D_prim[i, j] - Pfull[i, j] for i, j in OFFDIAG])
    er = np.array([D_prim[i, j] - Proute[i, j] for i, j in OFFDIAG])
    E_full, MAE_full, MAE_route = (rms(ef), float(np.mean(np.abs(ef))), float(np.mean(np.abs(er))))
    I_MAE = 1.0 - MAE_full / MAE_route
    S_B = rms(np.array([PB[i, j] for i, j in OFFDIAG]))
    sd = boot['P_route_replicates'].std(axis=0, ddof=1)
    E_MC = rms(sd)
    ci_I = percentile_ci(boot['I_MAE'])
    ci_SB = percentile_ci(boot['S_B'])
    D_RMS = rms(d_off)
    ds = np.array([(max((probe[r][i, j] for r in cfg['radii']['set'])) - min((probe[r][i, j] for r in cfg['radii']['set']))) / max(abs(D_prim[i, j]), D_RMS, 1e-12) for i, j in OFFDIAG])
    locality = bool(np.median(ds) <= thr['locality_median'] and ds.max() <= thr['locality_max'])
    R_full = E_full / max(D_RMS, 1e-12)
    Q_B = S_B / max(E_full, E_MC, 1e-12)
    gates = {'C1': {'R_full': R_full, 'E_full': E_full, 'threshold': thr['C1_R_full_max'], 'pass': bool(R_full <= thr['C1_R_full_max'])}, 'C2': {'I_MAE': I_MAE, 'MAE_full': MAE_full, 'MAE_route': MAE_route, 'CI95': list(ci_I), 'CI_low': ci_I[0], 'threshold': thr['C2_I_MAE_min'], 'CI_low_threshold': thr['C2_CI_low_min'], 'pass': bool(I_MAE >= thr['C2_I_MAE_min'] and ci_I[0] > thr['C2_CI_low_min'])}, 'C3': {'Q_B': Q_B, 'S_B': S_B, 'E_full': E_full, 'E_MC': E_MC, 'S_B_CI95': list(ci_SB), 'S_B_CI_low': ci_SB[0], 'threshold': thr['C3_Q_B_min'], 'pass': bool(Q_B >= thr['C3_Q_B_min'] and ci_SB[0] > 0.0)}, 'C4': {'E_MC_over_S_B': E_MC / S_B, 'E_MC': E_MC, 'S_B': S_B, 'threshold': thr['C4_E_MC_over_S_B_max'], 'pass': bool(E_MC / S_B <= thr['C4_E_MC_over_S_B_max'])}, 'C5': {'median_delta_scale': float(np.median(ds)), 'max_delta_scale': float(ds.max()), 'thresholds': [thr['locality_median'], thr['locality_max']], 'pass': locality}}
    return {'rho_primary': cfg['radii']['primary'], 'radii': cfg['radii']['set'], 'T_full_4x4': T_full_4.tolist(), 'T_route_4x4': T_route_4.tolist(), 'B_4x4': (T_full_4 - T_route_4).tolist(), 'P_full': Pfull.tolist(), 'P_route': Proute.tolist(), 'P_B': PB.tolist(), 'D_probe_primary': D_prim.tolist(), 'G_source': G_src.tolist(), 'RMS_D_probe': D_RMS, 'E_full': E_full, 'R_full': R_full, 'MAE_full': MAE_full, 'MAE_route': MAE_route, 'I_MAE': I_MAE, 'I_MAE_CI95': list(ci_I), 'S_B': S_B, 'S_B_CI95': list(ci_SB), 'E_MC': E_MC, 'E_MC_over_S_B': E_MC / S_B, 'Q_B': Q_B, 'median_delta_scale': float(np.median(ds)), 'max_delta_scale': float(ds.max()), 'delta_scale_rows': [[int(i), int(j), float(v)] for (i, j), v in zip(OFFDIAG, ds)], 'bootstrap': {'n_replicates': boot['n_replicates'], 'seed': boot['seed'], 'I_MAE_mean': float(boot['I_MAE'].mean()), 'S_B_mean': float(boot['S_B'].mean())}, 'gates': gates}

def decide(metrics, raw, cfg):
    stages = raw['stages']
    valid = all((bool(v.get('pass', False)) for v in stages.values()))
    gates = metrics['gates']
    if not valid:
        return 'INVALID'
    if all((g['pass'] for g in gates.values())):
        return 'CONFIRMED'
    if not gates['C4']['pass'] and all((gates[k]['pass'] for k in ('C1', 'C2', 'C3', 'C5'))):
        return 'INCONCLUSIVE_MC'
    return 'NOT_CONFIRMED'

def _write_tables(T_direct, T_route, B, keys):
    lab = [k.replace('|', '/') for k in keys]
    write_csv([[lab[i]] + [f'{v:.10e}' for v in T_direct[i]] for i in range(8)], PATHS['raw'] / 'tables' / 'u4_full_T.csv', ['set'] + lab)
    write_csv([[lab[i]] + [f'{T_route[i, j]:.10e}' for j in range(4, 8)] for i in range(4)], PATHS['raw'] / 'tables' / 'u4_route_T.csv', ['target'] + lab[4:])
    write_csv([[lab[i]] + [f'{B[i, j]:.10e}' for j in range(4, 8)] for i in range(4)], PATHS['raw'] / 'tables' / 'u4_B.csv', ['target'] + lab[4:])
    write_csv([[lab[i]] + [f'{v:.10e}' for v in T_direct[i, 4:8]] for i in range(4)], PATHS['raw'] / 'tables' / 'u4_full_T_4x4.csv', ['target'] + lab[4:])
if __name__ == '__main__':
    main()
