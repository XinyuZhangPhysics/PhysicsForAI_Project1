# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Execute frozen U4 source/target response on CPU float32."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import gc
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from transformers import OlmoeForCausalLM
from u4_core import actual_routes, copy_grad_flat, dot_blockwise, expert_output, flatten_params, manual_ce, param_layout, set_experts_flat
BASE = Path(__file__).parent
OUT = BASE / 'u4_run'
OUT.mkdir(exist_ok=True)
MODEL = _p1_path('path_f2699b70cfbf')
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']
GROUPS = [(split, domain) for split in ('validation', 'test') for domain in DOMAINS]
assert os.getenv('HF_HUB_OFFLINE') == '1' and os.getenv('TRANSFORMERS_OFFLINE') == '1'
assert json.loads((BASE / 'u4_preflight_evidence.json').read_text())['status'] == 'PASS'
torch.set_num_threads(8)
start = time.monotonic()
selected = json.loads((BASE / 'u4_selected_sequences.json').read_text())
assert len(selected) == 128
by_group = {(s, d): [r for r in selected if (r['split'], r['domain']) == (s, d)] for s, d in GROUPS}
assert all((len(v) == 16 for v in by_group.values()))
perms = json.loads((BASE / 'u4_codex_pkg/task/u4_reynolds_permutations.json').read_text())['permutations']
print('Loading pinned CPU float32 OLMoE', flush=True)
model = OlmoeForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager')
model.eval()
mlp = model.model.layers[15].mlp
assert len(model.model.layers) == 16 and len(mlp.experts) == 64 and (mlp.top_k == 8) and (not mlp.norm_topk_prob)
for p in model.parameters():
    p.requires_grad_(False)
for p in mlp.experts.parameters():
    p.requires_grad_(True)
params, offsets, total = param_layout(mlp)
baseline = flatten_params(params, offsets, total)
theta = math.sqrt(dot_blockwise(baseline, baseline, offsets))

def forbidden_hash():
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not name.startswith('model.layers.15.mlp.experts.'):
            h.update(name.encode())
            h.update(memoryview(p.detach().numpy()).cast('B'))
    return h.hexdigest()
forbidden_before = forbidden_hash()
stash = {}

def norm_pre(_module, inp):
    stash['residual'] = inp[0].detach().clone()

def mlp_hook(_module, inp, out):
    stash['h'] = inp[0].detach().clone()
    stash['m'] = out[0]
    stash['native_m'] = out[0].detach().clone()
    stash['router_logits'] = out[1].detach().clone()
    if out[0].requires_grad:
        out[0].retain_grad()
handle1 = model.model.layers[15].post_attention_layernorm.register_forward_pre_hook(norm_pre)
handle2 = mlp.register_forward_hook(mlp_hook)

def direct_sequence(ids):
    stash.clear()
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, use_cache=False)
    loss = manual_ce(out.logits, ids)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    h = stash['h'].detach().clone().squeeze(0)
    residual = stash['residual'].detach().clone().squeeze(0)
    d = stash['m'].grad.detach().clone().squeeze(0)
    native = stash['native_m'].detach().clone()
    router = stash['router_logits'].detach().clone()
    with torch.no_grad():
        _, labels, weights = actual_routes(mlp, h)
        recon = expert_output(mlp, h, labels, weights)
        err = float(torch.linalg.vector_norm(recon - native.squeeze(0)) / max(float(torch.linalg.vector_norm(native)), 1e-30))
    cache = {'h': h, 'residual': residual, 'd': d, 'labels': labels, 'weights': weights, 'router_logits': router, 'token_ids': ids.squeeze(0).detach().clone(), 'native_error': err, 'loss': float(loss.detach())}
    del out, loss
    return cache

def aggregate_cache(seq_list):
    return {'h': torch.stack([x['h'] for x in seq_list]), 'residual': torch.stack([x['residual'] for x in seq_list]), 'd': torch.stack([x['d'] for x in seq_list]), 'labels': torch.cat([x['labels'] for x in seq_list], dim=0), 'weights': torch.cat([x['weights'] for x in seq_list], dim=0), 'token_ids': torch.stack([x['token_ids'] for x in seq_list])}

def pseudo_gradient(group_cache, permutation=None):
    model.zero_grad(set_to_none=True)
    m = expert_output(mlp, group_cache['h'], group_cache['labels'], group_cache['weights'], permutation)
    scalar = (m * group_cache['d']).sum() / 16.0
    scalar.backward()
    result = torch.zeros(total, dtype=torch.float32)
    copy_grad_flat(params, offsets, result)
    return result

def transfer_matrix(gradients):
    return np.array([[dot_blockwise(gradients[4 + t], gradients[s], offsets) for s in range(4)] for t in range(4)], dtype=np.float64)

def cached_loss(group_cache):
    with torch.no_grad():
        m = expert_output(mlp, group_cache['h'], group_cache['labels'], group_cache['weights'])
        logits = model.lm_head(model.model.norm(group_cache['residual'] + m))
        return float(manual_ce(logits, group_cache['token_ids']))
direct_grad = []
caches = {}
router_logits_by_group = {}
native_errors = []
native_losses = []
for gi, group in enumerate(GROUPS):
    grad = torch.zeros(total, dtype=torch.float32)
    seq = []
    for row in by_group[group]:
        ids = torch.tensor(row['token_ids'], dtype=torch.long).unsqueeze(0)
        c = direct_sequence(ids)
        if c['native_error'] >= 1e-06:
            raise RuntimeError('INVALID: native reconstruction')
        native_errors.append(c['native_error'])
        copy_grad_flat(params, offsets, grad, 1 / 16)
        seq.append(c)
    direct_grad.append(grad)
    caches[group] = aggregate_cache(seq)
    router_logits_by_group[group] = torch.cat([x['router_logits'] for x in seq], dim=0)
    native_losses.append(float(np.mean([x['loss'] for x in seq])))
    print(json.dumps({'phase': 'direct', 'group': group, 'loss': native_losses[-1], 'gradient_norm': math.sqrt(dot_blockwise(grad, grad, offsets)), 'elapsed_seconds': time.monotonic() - start}), flush=True)
    del seq
handle1.remove()
handle2.remove()
G = np.array([math.sqrt(dot_blockwise(direct_grad[s], direct_grad[s], offsets)) for s in range(4)])
assert np.all(np.isfinite(G)) and np.min(G) >= 1e-12
T_full = transfer_matrix(direct_grad)
cached_grad = [pseudo_gradient(caches[g]) for g in GROUPS]
T_cached = transfer_matrix(cached_grad)
cached_T_error = float(np.linalg.norm(T_cached - T_full) / max(np.linalg.norm(T_full), 1e-30))
cached_g_errors = [float(torch.linalg.vector_norm(a - b) / max(float(torch.linalg.vector_norm(a)), 1e-30)) for a, b in zip(direct_grad, cached_grad)]
assert cached_T_error < 1e-05 and max(cached_g_errors) < 1e-05
del cached_grad
gc.collect()
baseline_loss_errors = []
for i, g in enumerate(GROUPS):
    baseline_loss_errors.append(abs(cached_loss(caches[g]) - native_losses[i]) / max(abs(native_losses[i]), 1e-30))
assert max(baseline_loss_errors) < 1e-06
print(json.dumps({'phase': 'identity_audit', 'T_error': cached_T_error, 'max_g_error': max(cached_g_errors), 'max_baseline_loss_error': max(baseline_loss_errors), 'elapsed_seconds': time.monotonic() - start}), flush=True)
T_perm = np.empty((16, 4, 4), dtype=np.float64)
for b, perm in enumerate(perms):
    p = torch.tensor(perm, dtype=torch.long)
    perm_grad = [pseudo_gradient(caches[g], p) for g in GROUPS]
    T_perm[b] = transfer_matrix(perm_grad)
    del perm_grad
    gc.collect()
    print(json.dumps({'phase': 'permutation', 'index': b + 1, 'elapsed_seconds': time.monotonic() - start}), flush=True)
T_route = T_perm.mean(axis=0)
B = T_full - T_route
np.savez_compressed(OUT / 'u4_permutation_T.npz', T_permutation=T_perm)
probe = np.empty((3, 4, 4), dtype=np.float64)
probe_rows = []
router_errors = {'indices_mismatches': 0, 'max_logit_abs': 0.0, 'max_weight_abs': 0.0}
radii = [5e-06, 1e-05, 2e-05]
for si in range(4):
    direction = -direct_grad[si] / G[si]
    for ri, rho in enumerate(radii):
        epsilon = rho * theta
        losses = {}
        for sign in (1, -1):
            set_experts_flat(params, offsets, baseline, direction, sign * epsilon)
            if ri == 2:
                for group in GROUPS:
                    c = caches[group]
                    with torch.no_grad():
                        logits, labels, weights = actual_routes(mlp, c['h'])
                    old_labels = c['labels']
                    old_weights = c['weights']
                    router_errors['indices_mismatches'] += int((labels != old_labels).sum())
                    router_errors['max_weight_abs'] = max(router_errors['max_weight_abs'], float((weights - old_weights).abs().max()))
                    old_logits = router_logits_by_group[group]
                    router_errors['max_logit_abs'] = max(router_errors['max_logit_abs'], float((logits - old_logits).abs().max()))
            losses[sign] = [cached_loss(caches['test', d]) for d in DOMAINS]
            set_experts_flat(params, offsets, baseline)
        for ti in range(4):
            derivative = (losses[1][ti] - losses[-1][ti]) / (2 * epsilon)
            probe[ri, ti, si] = derivative
            probe_rows.append({'source': DOMAINS[si], 'target': DOMAINS[ti], 'rho': rho, 'epsilon': epsilon, 'loss_plus': losses[1][ti], 'loss_minus': losses[-1][ti], 'D_probe': derivative})
        print(json.dumps({'phase': 'probe', 'source': DOMAINS[si], 'rho': rho, 'elapsed_seconds': time.monotonic() - start}), flush=True)
assert router_errors['indices_mismatches'] == 0
assert router_errors['max_logit_abs'] < 1e-07 and router_errors['max_weight_abs'] < 1e-07
assert all((torch.equal(p.detach().reshape(-1), baseline[a:b]) for p, (a, b) in zip(params, offsets)))
assert forbidden_hash() == forbidden_before
replay_grad = torch.zeros(total, dtype=torch.float32)
stash.clear()
handle1 = model.model.layers[15].post_attention_layernorm.register_forward_pre_hook(norm_pre)
handle2 = mlp.register_forward_hook(mlp_hook)
for row in by_group['validation', DOMAINS[0]]:
    c = direct_sequence(torch.tensor(row['token_ids'], dtype=torch.long).unsqueeze(0))
    copy_grad_flat(params, offsets, replay_grad, 1 / 16)
handle1.remove()
handle2.remove()
replay_gradient_error = float(torch.linalg.vector_norm(replay_grad - direct_grad[0]) / max(float(torch.linalg.vector_norm(direct_grad[0])), 1e-30))
assert replay_gradient_error < 1e-06
direction = -direct_grad[0] / G[0]
epsilon = 1e-05 * theta
replay_losses = {}
for sign in (1, -1):
    set_experts_flat(params, offsets, baseline, direction, sign * epsilon)
    replay_losses[sign] = cached_loss(caches['test', DOMAINS[1]])
    set_experts_flat(params, offsets, baseline)
replay_probe = (replay_losses[1] - replay_losses[-1]) / (2 * epsilon)
replay_probe_error = abs(replay_probe - probe[1, 1, 0]) / max(abs(probe[1, 1, 0]), 1e-30)
assert replay_probe_error < 1e-06
np.savez_compressed(OUT / 'u4_scientific_arrays.npz', T_full=T_full, T_route=T_route, B=B, T_permutation=T_perm, probe=probe, G=G, theta=theta, radii=np.array(radii), native_losses=np.array(native_losses))
with (OUT / 'u4_probe_raw.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(probe_rows[0]))
    w.writeheader()
    w.writerows(probe_rows)
audit = {'status': 'PASS', 'max_native_reconstruction_error': max(native_errors), 'cached_T_relative_error': cached_T_error, 'max_cached_gradient_relative_error': max(cached_g_errors), 'max_cached_baseline_loss_relative_error': max(baseline_loss_errors), 'router_stability': router_errors, 'forbidden_parameter_hash_before_after': forbidden_before, 'expert_restoration_exact': True, 'replay_gradient_relative_error': replay_gradient_error, 'replay_probe_relative_error': replay_probe_error, 'source_gradient_norms': G.tolist(), 'theta': theta, 'elapsed_seconds': time.monotonic() - start, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(OUT / 'u4_execution_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps({'phase': 'complete', **audit}, indent=2), flush=True)
