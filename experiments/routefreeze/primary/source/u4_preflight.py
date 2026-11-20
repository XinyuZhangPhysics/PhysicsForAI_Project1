# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Non-scientific U4 gates; run before source/target metrics."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, OlmoeForCausalLM
from u4_core import actual_routes, copy_grad_flat, dot_blockwise, expert_output, flatten_params, manual_ce, param_layout, set_experts_flat
BASE = Path(__file__).parent
MODEL = _p1_path('path_f2699b70cfbf')
assert os.getenv('HF_HUB_OFFLINE') == '1' and os.getenv('TRANSFORMERS_OFFLINE') == '1'
torch.set_num_threads(8)
start = time.monotonic()
frozen = json.loads((BASE / 'u4_codex_pkg/task/u4_reynolds_permutations.json').read_text())
perms = frozen['permutations']
digest = hashlib.sha256(json.dumps(perms, separators=(',', ':')).encode()).hexdigest()
assert digest == '238a09d483374cfcf45863c0b4ffa93cf70baa8b1d1c14ae2f47650ca3c840fb'
assert len(perms) == 16 and len({tuple(p) for p in perms}) == 16
assert all((sorted(p) == list(range(64)) for p in perms))
rng = np.random.default_rng(2619)
H = 4
v = rng.normal(size=(H, 7))
a = rng.random(H)
b = rng.random(H)
analytic = float(a @ b * np.mean(np.sum(v * v, axis=1)))
enumerated = np.mean([sum((a[i] * b[i] * float(v[p[i]] @ v[p[i]]) for i in range(H))) for p in itertools.permutations(range(H))])
synthetic_error = abs(enumerated - analytic) / max(abs(analytic), 1e-30)
assert synthetic_error < 1e-10
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = OlmoeForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager')
model.eval()
mlp = model.model.layers[15].mlp
assert len(model.model.layers) == 16 and len(mlp.experts) == 64 and (mlp.top_k == 8) and (not mlp.norm_topk_prob)
assert not any(('shared_expert' in n for n, _ in model.named_parameters()))
for p in model.parameters():
    p.requires_grad_(False)
for p in mlp.experts.parameters():
    p.requires_grad_(True)
params, offsets, total = param_layout(mlp)
assert len(params) == 192
baseline = flatten_params(params, offsets, total)

def forbidden_hash():
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not name.startswith('model.layers.15.mlp.experts.'):
            h.update(name.encode())
            h.update(memoryview(p.detach().numpy()).cast('B'))
    return h.hexdigest()
forbidden_before = forbidden_hash()
ids = tok('The quick brown fox jumps over the lazy dog.', add_special_tokens=False, return_tensors='pt')['input_ids']
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

def direct_capture():
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, use_cache=False)
    loss = manual_ce(out.logits, ids)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    grad = torch.zeros(total)
    copy_grad_flat(params, offsets, grad)
    d = stash['m'].grad.detach().clone()
    h = stash['h'].clone()
    native = stash['native_m'].clone()
    router = stash['router_logits'].clone()
    return (float(loss.detach()), grad, h, d, native, router)
loss, g, h, d, native, router = direct_capture()
with torch.no_grad():
    output = model(input_ids=ids, labels=ids, use_cache=False)
    direct_ce = float(output.loss)
assert abs(loss - direct_ce) / max(abs(loss), 1e-30) < 1e-07
_, labels, weights = actual_routes(mlp, h)
with torch.no_grad():
    reconstructed = expert_output(mlp, h, labels, weights)
native_error = float(torch.linalg.vector_norm(reconstructed - native) / max(float(torch.linalg.vector_norm(native)), 1e-30))
assert native_error < 1e-06
model.zero_grad(set_to_none=True)
scalar = (expert_output(mlp, h, labels, weights) * d).sum()
scalar.backward()
cached_g = torch.zeros(total)
copy_grad_flat(params, offsets, cached_g)
gradient_error = float(torch.linalg.vector_norm(cached_g - g) / max(float(torch.linalg.vector_norm(g)), 1e-30))
assert gradient_error < 1e-05
smoke_ids = ids
ids = tok('Hello world', add_special_tokens=False, return_tensors='pt')['input_ids']
assert ids.shape[1] == 2
with torch.no_grad():
    reference = [p.detach().clone() for p in mlp.experts[0].parameters()]
    for expert in mlp.experts:
        for p, ref in zip(expert.parameters(), reference):
            p.copy_(ref)
null_loss, null_g, null_h, null_d, null_native, _ = direct_capture()
_, null_labels, null_weights = actual_routes(mlp, null_h)
null_full = dot_blockwise(null_g, null_g, offsets)
null_values = []
for perm in perms:
    model.zero_grad(set_to_none=True)
    (expert_output(mlp, null_h, null_labels, null_weights, torch.tensor(perm)) * null_d).sum().backward()
    pg = torch.zeros(total)
    copy_grad_flat(params, offsets, pg)
    null_values.append(dot_blockwise(pg, pg, offsets))
null_route = float(np.mean(null_values))
null_ratio = abs(null_full - null_route) / max(abs(null_full), 1e-30)
print(json.dumps({'null_full': null_full, 'null_route': null_route, 'null_ratio': null_ratio, 'null_min': min(null_values), 'null_max': max(null_values)}), flush=True)
assert null_ratio < 1e-05
set_experts_flat(params, offsets, baseline)
assert all((torch.equal(p.detach().reshape(-1), baseline[a:b]) for p, (a, b) in zip(params, offsets)))
ids = smoke_ids
G = float(torch.linalg.vector_norm(g))
theta = float(torch.linalg.vector_norm(baseline))
assert G > 1e-12
direction = -g / G
max_epsilon = 2e-05 * theta
baseline_router = router
baseline_labels = labels
baseline_weights = weights
router_logit_error = 0.0
router_weight_error = 0.0
for sign in (1, -1):
    set_experts_flat(params, offsets, baseline, direction, sign * max_epsilon)
    with torch.no_grad():
        out = model(input_ids=ids, output_router_logits=True, use_cache=False)
        new_router = out.router_logits[15]
        _, new_labels, new_weights = actual_routes(mlp, stash['h'])
    assert torch.equal(new_labels, baseline_labels)
    router_logit_error = max(router_logit_error, float((new_router - baseline_router).abs().max()))
    router_weight_error = max(router_weight_error, float((new_weights - baseline_weights).abs().max()))
set_experts_flat(params, offsets, baseline)
assert router_logit_error < 1e-07 and router_weight_error < 1e-07
assert forbidden_hash() == forbidden_before
assert all((torch.equal(p.detach().reshape(-1), baseline[a:b]) for p, (a, b) in zip(params, offsets)))
handle1.remove()
handle2.remove()
result = {'status': 'PASS', 'model_revision': Path(MODEL).name, 'precision': 'CPU float32', 'architecture': {'layers': 16, 'routed_experts': 64, 'top_k': 8, 'final_layer': 15, 'shared_experts': 0}, 'permutation_digest': digest, 'synthetic_relative_error': float(synthetic_error), 'manual_ce_relative_error': abs(loss - direct_ce) / max(abs(loss), 1e-30), 'native_reconstruction_relative_error': native_error, 'cached_gradient_relative_error': gradient_error, 'common_expert_null_ratio': null_ratio, 'common_expert_null_full': null_full, 'common_expert_null_route': null_route, 'router_max_logit_error': router_logit_error, 'router_max_weight_error': router_weight_error, 'forbidden_parameter_hash': forbidden_before, 'expert_restoration_exact': True, 'calibration_tokens': int(ids.shape[1]), 'elapsed_seconds': time.monotonic() - start}
(BASE / 'u4_preflight_evidence.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
