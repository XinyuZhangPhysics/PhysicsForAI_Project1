# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Pre-science gates (preregistration Sections 4, 21, 22).

Run BEFORE any scientific source/target metric is looked at.  Writes
raw/preflight.json and logs/preflight.log.
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
from u4_common import PATHS, ensure_dirs, json_dump, log, load_config, now_utc, sha256_file, torch_setup
from u4_data import DOMAINS, PARQUET, SPLITS
from u4_model import D_FF, D_MODEL, N_EXPERTS, N_TOK, FinalLayerMoE, direct_next_token_ce, load_model_and_tokenizer, manual_next_token_ce
SMOKE_TEXT = 'The quick brown fox jumps over the lazy dog. Attention is all you need; experts route tokens to specialists. ' * 8

def main() -> None:
    ensure_dirs()
    cfg = load_config()
    res = {'generated_utc': now_utc(), 'checks': {}, 'blocked': None}
    md = PATHS['model_dir']
    asset = {}
    for name, want in cfg['model']['safetensors_sha256'].items():
        got = sha256_file(md / name)
        asset[name] = {'sha256': got, 'match': got == want, 'bytes': (md / name).stat().st_size}
    got = sha256_file(md / 'config.json')
    asset['config.json'] = {'sha256': got, 'match': got == cfg['model']['config_sha256'], 'bytes': (md / 'config.json').stat().st_size}
    res['checks']['asset_hashes'] = asset
    if not all((v['match'] for v in asset.values())):
        res['blocked'] = 'ASSET_BLOCKED'
    data = {}
    for split in SPLITS:
        p = PATHS['data_dir'] / PARQUET[split]
        d = sha256_file(p)
        want = cfg['data'][f'{split}_parquet_sha256']
        data[split] = {'path': str(p), 'sha256': d, 'match': d == want, 'bytes': p.stat().st_size}
    res['checks']['data_hashes'] = data
    if not all((v['match'] for v in data.values())):
        res['blocked'] = 'DATA_BLOCKED'
    if res['blocked']:
        json_dump(res, PATHS['raw'] / 'preflight.json')
        log(f"BLOCKED: {res['blocked']}")
        return
    torch = torch_setup(threads=cfg.get('threads', 18), seed=202609280001)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(cfg)
    load_s = time.time() - t0
    types = {p.dtype for p in model.parameters()}
    res['checks']['load'] = {'seconds': load_s, 'param_dtypes': sorted((str(t) for t in types)), 'local_files_only': True, 'offline': True}
    moe = FinalLayerMoE(model, tok)
    arch = {'n_layers': len(model.model.layers), 'n_experts': len(moe.mlp.experts), 'top_k': int(model.config.num_experts_per_tok), 'selected_layer': 15, 'router_param_groups': 1, 'n_expert_param_tensors': len(moe.all_params()), 'expert_tensor_shapes': [list(p.shape) for p in moe.ew[0]], 'norm_topk_prob': bool(model.config.norm_topk_prob), 'num_experts_cfg': int(model.config.num_experts), 'hidden_size': int(model.config.hidden_size), 'moe_intermediate_size': int(getattr(model.config, 'intermediate_size')), 'has_shared_expert': hasattr(moe.mlp, 'shared_expert')}
    arch['arch_ok'] = bool(arch['n_layers'] == 16 and arch['n_experts'] == 64 and (arch['top_k'] == 8) and (arch['n_expert_param_tensors'] == 192) and (arch['norm_topk_prob'] is False) and (arch['has_shared_expert'] is False))
    res['checks']['architecture'] = arch
    ids32 = tok(SMOKE_TEXT, add_special_tokens=False)['input_ids'][:32]
    ids = torch.tensor([ids32], dtype=torch.long)
    moe.set_grad_mode()
    t0 = time.time()
    cap = moe.forward_capture(ids, grad=True)
    fwd_s = time.time() - t0
    ce_man = manual_next_token_ce(cap['lm_logits'], ids)
    ce_dir = direct_next_token_ce(cap['lm_logits'], ids)
    ce_man.backward()
    bwd_s = time.time() - t0 - fwd_s
    grads = {n: p.grad is not None for n, p in model.named_parameters()}
    out_of_scope = [n for n, g in grads.items() if g and (not n.startswith('model.layers.15.mlp.experts.'))]
    n_with_grad = sum((1 for v in grads.values() if v))
    gnorms = [float(p.grad.norm()) for j in range(N_EXPERTS) for p in moe.param_tensors(j) if p.grad is not None]
    moe.set_grad_mode()
    for p in moe.all_params():
        p.grad = None
    torch.manual_seed(202609280001)
    cap2 = moe.forward_capture(ids, grad=True)
    ce_man2 = manual_next_token_ce(cap2['lm_logits'], ids)
    ce_man2.backward()
    gnorms2 = [float(p.grad.norm()) for j in range(N_EXPERTS) for p in moe.param_tensors(j) if p.grad is not None]
    rel_loss = abs(float(ce_man.detach()) - float(ce_man2.detach())) / max(abs(float(ce_man)), 1e-30)
    rel_grad = max((abs(a - b) / max(abs(a), 1e-30) for a, b in zip(gnorms, gnorms2)))
    rl = cap['router_logits']
    res['checks']['smoke'] = {'seq_tokens': len(ids32), 'forward_seconds': fwd_s, 'backward_seconds': bwd_s, 'manual_ce': float(ce_man.detach()), 'direct_ce': float(ce_dir), 'manual_vs_direct_abs': abs(float(ce_man.detach()) - float(ce_dir.detach())), 'loss_finite': bool(np.isfinite(float(ce_man.detach()))), 'router_logits_finite': bool(np.isfinite(rl.numpy()).all()), 'router_logits_shape': list(rl.shape), 'expert_grad_finite': bool(np.isfinite(gnorms).all()), 'n_params_with_grad': int(n_with_grad), 'expected_n_params_with_grad': 192, 'grads_outside_layer15_experts': out_of_scope, 'repeat_rel_loss': rel_loss, 'repeat_rel_max_grad_norm': rel_grad, 'repeat_rtol_target': 1e-07}
    res['checks']['smoke']['smoke_ok'] = bool(res['checks']['smoke']['loss_finite'] and res['checks']['smoke']['router_logits_finite'] and res['checks']['smoke']['expert_grad_finite'] and (n_with_grad == 192) and (not out_of_scope) and (rel_loss <= 1e-07) and (rel_grad <= 1e-07) and (abs(float(ce_man.detach()) - float(ce_dir.detach())) <= 1e-05))
    per_tok = fwd_s / len(ids32)
    res['checks']['timing'] = {'forward_seconds_per_token': per_tok, 'projected_full_forward_seconds_128seq_128tok': per_tok * 128 * 128, 'note': 'smoke uses a 32-token sequence, so attention cost is not representative'}
    z, w, idx = (cap['z'], cap['route_w'], cap['route_idx'])
    m_my = moe.moe(z, w, idx)
    m_nat = cap['m_native']
    rel = float((m_my - m_nat).norm() / max(float(m_nat.norm()), 1e-30))
    h_out_my = cap['h_att'] + m_my
    lg_my = moe.head(h_out_my)
    rel_lg = float((lg_my - cap['lm_logits']).norm() / max(float(cap['lm_logits'].norm()), 1e-30))
    rl_my = z @ moe.gate_w.t()
    rel_rl = float((rl_my - rl).norm() / max(float(rl.norm()), 1e-30))
    res['checks']['instrumented_reconstruction'] = {'moe_output_rel_error': rel, 'lm_logits_rel_error': rel_lg, 'router_logits_rel_error': rel_rl, 'pass': bool(rel < cfg['thresholds']['gate_native_vs_instrumented_rel'])}
    res['all_gates_pass'] = bool(arch['arch_ok'] and res['checks']['smoke']['smoke_ok'] and res['checks']['instrumented_reconstruction']['pass'])
    json_dump(res, PATHS['raw'] / 'preflight.json')
    log(json.dumps({k: v for k, v in res.items() if k != 'checks'}, indent=1))
    log(f"arch_ok={arch['arch_ok']} smoke_ok={res['checks']['smoke']['smoke_ok']} recon_rel={rel:.3e} all_gates_pass={res['all_gates_pass']}")
if __name__ == '__main__':
    main()
