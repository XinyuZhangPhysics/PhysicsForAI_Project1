# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import huggingface_hub

def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()

def mem_snapshot(device):
    out = {}
    if device.type == 'mps':
        for name in ('current_allocated_memory', 'driver_allocated_memory', 'recommended_max_memory'):
            fn = getattr(torch.mps, name, None)
            if fn is not None:
                try:
                    out[name] = int(fn())
                except Exception as exc:
                    out[name] = f'ERROR:{exc}'
    return out

def tensor_norm(x):
    return float(torch.linalg.vector_norm(x.float()).cpu())

def resolve_dtype(name):
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[name]

def timed(fn, device, repeats):
    vals = []
    result = None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        result = fn()
        sync(device)
        vals.append(time.perf_counter() - t0)
    return (result, {'times_s': vals, 'median_s': statistics.median(vals), 'min_s': min(vals), 'max_s': max(vals)})

def benchmark_timed(fn, device, warmups=3, repeats=10):
    for _ in range(warmups):
        fn()
        sync(device)
    values = []
    result = None
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        values.append(time.perf_counter() - start)
    median = statistics.median(values)
    return (result, {'warmups': warmups, 'repeats': repeats, 'times_s': values, 'median_s': median, 'mad_s': statistics.median((abs(t - median) for t in values)), 'min_s': min(values), 'max_s': max(values)})

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)

def get_target(model):
    base = getattr(model, 'model', None)
    if base is None or not hasattr(base, 'layers'):
        raise RuntimeError('Cannot locate model.layers')
    layer = base.layers[-1]
    if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'experts') or (not hasattr(layer.mlp, 'gate')):
        raise RuntimeError('Last layer is not recognized as routed OLMoE')
    experts = layer.mlp.experts
    for attr in ('gate_up_proj', 'down_proj'):
        if not hasattr(experts, attr):
            raise RuntimeError(f'Missing expert tensor: {attr}')
    return (layer, layer.mlp, experts)

def tiny_inputs(tokenizer, device, seq_len):
    texts = ['Statistical mechanics relates microscopic fluctuations to macroscopic response. Coarse graining asks which collective variables preserve predictive information. ' * 64, 'A sparse mixture of experts routes each token through a subset of parameter blocks. We measure how local expert responses combine into cross task transfer and interference. ' * 64]
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'use_cache': False}

def env_smoke(device):
    dtypes = [torch.float32]
    if device.type == 'mps':
        dtypes = [torch.bfloat16, torch.float16, torch.float32]
    checks = {}
    for dt in dtypes:
        key = str(dt).replace('torch.', '')
        try:
            x = torch.randn(64, 32, device=device, dtype=dt, requires_grad=True)
            w = torch.randn(48, 32, device=device, dtype=dt, requires_grad=True)
            y = F.linear(x, w)
            p = torch.softmax(y.float(), dim=-1)
            v, idx = torch.topk(p, 4, dim=-1)
            acc = torch.zeros_like(y.float())
            acc.index_add_(0, torch.arange(y.shape[0], device=device), y.float())
            loss = v.square().mean() + acc.square().mean()
            loss.backward()
            sync(device)
            checks[key] = {'pass': True}
        except Exception as exc:
            checks[key] = {'pass': False, 'error': repr(exc)}
    return checks

def ordinary_oracle(model, experts, inputs, device, repeats):
    freeze_all(model)
    experts.gate_up_proj.requires_grad_(True)
    experts.down_proj.requires_grad_(True)

    def run():
        model.zero_grad(set_to_none=True)
        out = model(**inputs)
        out.loss.backward()
        gu = experts.gate_up_proj.grad.detach().clone()
        dp = experts.down_proj.grad.detach().clone()
        return (float(out.loss.detach().float().cpu()), gu, dp)
    (loss, gu, dp), timing = timed(run, device, repeats)
    other_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and p is not experts.gate_up_proj and (p is not experts.down_proj):
            other_grads.append(name)
    freeze_all(model)
    model.zero_grad(set_to_none=True)
    return {'loss': loss, 'gate_up_grad': gu, 'down_grad': dp, 'timing': timing, 'other_parameters_with_grad': other_grads}

def probe_capture(model, mlp, inputs, device, repeats):
    freeze_all(model)
    stash = {}

    def hook(module, args, output):
        stash['h'] = args[0].detach()
        leaf = output.detach().requires_grad_(True)
        stash['leaf'] = leaf
        return leaf

    def run():
        stash.clear()
        model.zero_grad(set_to_none=True)
        handle = mlp.register_forward_hook(hook)
        try:
            out = model(**inputs)
            out.loss.backward()
            h = stash['h'].detach().clone()
            r = stash['leaf'].grad.detach().clone()
            loss = float(out.loss.detach().float().cpu())
        finally:
            handle.remove()
        with torch.no_grad():
            flat = h.reshape(-1, h.shape[-1])
            _, route_w, route_i = mlp.gate(flat)
        return (loss, h, r, route_w.detach(), route_i.detach())
    (loss, h, r, rw, ri), timing = timed(run, device, repeats)
    return {'loss': loss, 'h': h, 'r': r, 'route_w': rw, 'route_i': ri, 'timing': timing}

def benchmark_probe_capture(model, mlp, inputs, device):
    freeze_all(model)
    stash = {}

    def hook(module, args, output):
        stash['h'] = args[0].detach()
        leaf = output.detach().requires_grad_(True)
        stash['leaf'] = leaf
        return leaf

    def run():
        stash.clear()
        model.zero_grad(set_to_none=True)
        handle = mlp.register_forward_hook(hook)
        try:
            out = model(**inputs)
            out.loss.backward()
            h = stash['h'].detach().clone()
            r = stash['leaf'].grad.detach().clone()
            loss = float(out.loss.detach().float().cpu())
        finally:
            handle.remove()
        with torch.no_grad():
            flat = h.reshape(-1, h.shape[-1])
            _, route_w, route_i = mlp.gate(flat)
        return (loss, h, r, route_w.detach(), route_i.detach())
    (loss, h, r, rw, ri), timing = benchmark_timed(run, device)
    return {'loss': loss, 'h': h, 'r': r, 'route_w': rw, 'route_i': ri, 'timing': timing}

def local_vjp_one(experts, expert_idx, h, r, route_w, route_i):
    hflat = h.reshape(-1, h.shape[-1])
    rflat = r.reshape(-1, r.shape[-1])
    hits = torch.nonzero(route_i == expert_idx, as_tuple=False)
    if hits.numel() == 0:
        return (None, None, 0)
    token_idx = hits[:, 0]
    topk_pos = hits[:, 1]
    current = hflat[token_idx]
    probe = rflat[token_idx]
    weights = route_w[token_idx, topk_pos]
    wgu = experts.gate_up_proj[expert_idx].detach().clone().requires_grad_(True)
    wdn = experts.down_proj[expert_idx].detach().clone().requires_grad_(True)
    gate_up = F.linear(current, wgu)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = experts.act_fn(gate) * up
    y = F.linear(hidden, wdn) * weights[:, None]
    scalar = (y * probe).sum()
    ggu, gdn = torch.autograd.grad(scalar, (wgu, wdn))
    return (ggu.detach(), gdn.detach(), int(hits.shape[0]))

def equivalence(experts, oracle, probe, device):
    num_experts = int(experts.gate_up_proj.shape[0])
    eps = 1e-30
    rows = []
    loads = []
    for i in range(num_experts):
        ggu, gdn, n = local_vjp_one(experts, i, probe['h'], probe['r'], probe['route_w'], probe['route_i'])
        loads.append(n)
        if n == 0:
            rows.append({'expert': i, 'assignments': 0})
            continue
        ogu = oracle['gate_up_grad'][i]
        odn = oracle['down_grad'][i]
        agu = tensor_norm(ggu - ogu)
        adn = tensor_norm(gdn - odn)
        ngu = tensor_norm(ogu)
        ndn = tensor_norm(odn)
        rows.append({'expert': i, 'assignments': n, 'gate_up_abs_error': agu, 'gate_up_rel_error': agu / max(ngu, eps), 'down_abs_error': adn, 'down_rel_error': adn / max(ndn, eps), 'gate_up_oracle_norm': ngu, 'down_oracle_norm': ndn})
    return (rows, loads)

def stratified_order(loads):
    ids = list(range(len(loads)))
    ids.sort(key=lambda i: (loads[i], i))
    if not ids:
        return ids
    order = []
    lo, hi = (0, len(ids) - 1)
    take_hi = True
    while lo <= hi:
        if take_hi:
            order.append(ids[hi])
            hi -= 1
        else:
            order.append(ids[lo])
            lo += 1
        take_hi = not take_hi
    return order

def cost_curve(experts, probe, loads, device, repeats):
    order = stratified_order(loads)
    sizes = [1, 2, 4, 8, 16, 32, 64]
    sizes = [m for m in sizes if m <= len(order)]
    out = []
    for m in sizes:
        subset = order[:m]

        def run():
            total_assign = 0
            norms = 0.0
            for i in subset:
                ggu, gdn, n = local_vjp_one(experts, i, probe['h'], probe['r'], probe['route_w'], probe['route_i'])
                total_assign += n
                if ggu is not None:
                    norms += tensor_norm(ggu) + tensor_norm(gdn)
            return (total_assign, norms)
        (assignments, norms), timing = timed(run, device, repeats)
        out.append({'m': m, 'subset': subset, 'assignments': assignments, 'gradient_norm_checksum': norms, 'timing': timing, 'memory': mem_snapshot(device)})
    return out

def benchmark_curve(experts, probe, loads, device, family):
    if family == 'all_experts':
        order = stratified_order(loads)
        sizes = [1, 2, 4, 8, 16, 32, 64]
    elif family == 'active_only':
        active = [i for i, load in enumerate(loads) if load > 0]
        active_loads = [loads[i] for i in active]
        order = [active[j] for j in stratified_order(active_loads)]
        sizes = list(dict.fromkeys([m for m in (1, 2, 4, 8, 16, 32, len(active)) if m <= len(active)]))
    else:
        raise ValueError(f'unknown family: {family}')
    rows = []
    for m in sizes:
        subset = order[:m]

        def run():
            assignment_count = 0
            norm_checksum = 0.0
            for i in subset:
                ggu, gdn, n = local_vjp_one(experts, i, probe['h'], probe['r'], probe['route_w'], probe['route_i'])
                assignment_count += n
                if ggu is not None:
                    norm_checksum += tensor_norm(ggu) + tensor_norm(gdn)
            return (assignment_count, norm_checksum)
        (assignments, checksum), timing = benchmark_timed(run, device)
        assert assignments == sum((loads[i] for i in subset))
        rows.append({'m': m, 'subset': subset, 'assignments': assignments, 'active_experts_in_subset': sum((loads[i] > 0 for i in subset)), 'gradient_norm_checksum': checksum, 'timing': timing, 'local_seconds_per_assignment': timing['median_s'] / assignments if assignments else None, 'memory': mem_snapshot(device)})
    return rows

def quantile(values, probability):
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])

def cost_contrast(curve, probe_timing, seed=20260928, bootstrap_repeats=10000):
    by_m = {row['m']: row for row in curve}
    maximum = max(by_m)
    if 16 not in by_m or maximum <= 16:
        raise RuntimeError('insufficient active experts for m=16 cost contrast')
    small = by_m[16]['timing']
    large = by_m[maximum]['timing']
    probe_median = probe_timing['median_s']
    cost16 = probe_median + small['median_s']
    costmax = probe_median + large['median_s']
    reduction = 1.0 - cost16 / costmax
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_repeats):

        def resampled_median(times):
            return statistics.median(rng.choices(times, k=len(times)))
        p = resampled_median(probe_timing['times_s'])
        t16 = resampled_median(small['times_s'])
        tmax = resampled_median(large['times_s'])
        bootstrap.append(1.0 - (p + t16) / (p + tmax))
    return {'m_small': 16, 'm_large': maximum, 'total_cost_16_s': cost16, 'total_cost_max_s': costmax, 'reduction': reduction, 'bootstrap_seed': seed, 'bootstrap_repeats': bootstrap_repeats, 'reduction_95pct_interval': [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)]}

def assignment_fit(curve):
    xs = [float(row['assignments']) for row in curve]
    ys = [row['timing']['median_s'] for row in curve]
    mean_x, mean_y = (statistics.mean(xs), statistics.mean(ys))
    denominator = sum(((x - mean_x) ** 2 for x in xs))
    slope = sum(((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))) / denominator if denominator else 0.0
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    median_residual = statistics.median(residuals)
    total_variation = sum(((y - mean_y) ** 2 for y in ys))
    return {'model': 'C_localVJP_seconds = intercept_seconds + slope_seconds_per_assignment * N_assign', 'intercept_seconds': intercept, 'slope_seconds_per_assignment': slope, 'r_squared': 1.0 - sum((r * r for r in residuals)) / total_variation if total_variation else None, 'residuals_s_by_m': {str(row['m']): residual for row, residual in zip(curve, residuals)}, 'residual_summary_s': {'median': median_residual, 'mad': statistics.median((abs(r - median_residual) for r in residuals)), 'min': min(residuals), 'max': max(residuals)}}

def monotonicity(curve):
    ordered = sorted(curve, key=lambda row: (row['assignments'], row['m']))
    violations = []
    for previous, current in zip(ordered, ordered[1:]):
        if current['assignments'] == previous['assignments']:
            continue
        earlier = previous['timing']
        later = current['timing']
        drop = earlier['median_s'] - later['median_s']
        noise_allowance = 2.0 * (earlier['mad_s'] + later['mad_s'])
        if drop > noise_allowance:
            violations.append({'from_m': previous['m'], 'to_m': current['m'], 'from_assignments': previous['assignments'], 'to_assignments': current['assignments'], 'median_drop_s': drop, 'two_mad_allowance_s': noise_allowance})
    return {'criterion': 'adjacent assignment-ordered median decreases exceeding twice the sum of MADs', 'broadly_monotone': not violations, 'violations': violations}

def route_state(mlp, captured_h):
    with torch.no_grad():
        flat = captured_h.reshape(-1, captured_h.shape[-1])
        logits, weights, indices = mlp.gate(flat)
    return (logits.detach().clone(), weights.detach().clone(), indices.detach().clone())

def route_invariance(model, mlp, experts, inputs, probe, oracle, device, rel_step):
    loads = torch.bincount(probe['route_i'].reshape(-1), minlength=experts.gate_up_proj.shape[0])
    expert_idx = int(torch.argmax(loads).item())
    gu0 = experts.gate_up_proj[expert_idx].detach().clone()
    dn0 = experts.down_proj[expert_idx].detach().clone()
    base_logits, base_w, base_i = route_state(mlp, probe['h'])
    gu_grad = oracle['gate_up_grad'][expert_idx].to(experts.gate_up_proj.dtype)
    dn_grad = oracle['down_grad'][expert_idx].to(experts.down_proj.dtype)
    pnorm = torch.sqrt(experts.gate_up_proj[expert_idx].float().square().sum() + experts.down_proj[expert_idx].float().square().sum())
    gnorm = torch.sqrt(gu_grad.float().square().sum() + dn_grad.float().square().sum())
    eta = float(rel_step * pnorm / max(float(gnorm), 1e-30))
    with torch.no_grad():
        experts.gate_up_proj[expert_idx].add_(-eta * gu_grad)
        experts.down_proj[expert_idx].add_(-eta * dn_grad)
        actual_step = torch.sqrt((experts.gate_up_proj[expert_idx].float() - gu0.float()).square().sum() + (experts.down_proj[expert_idx].float() - dn0.float()).square().sum())
        actual_rel_step = float(actual_step / max(float(pnorm), 1e-30))
    after = probe_capture(model, mlp, inputs, device, 1)
    aft_logits, aft_w, aft_i = route_state(mlp, after['h'])
    with torch.no_grad():
        experts.gate_up_proj[expert_idx].copy_(gu0)
        experts.down_proj[expert_idx].copy_(dn0)
    restored = probe_capture(model, mlp, inputs, device, 1)
    return {'expert': expert_idx, 'eta': eta, 'relative_step_target': rel_step, 'actual_relative_step': actual_rel_step, 'input_max_abs_change': float((after['h'].float() - probe['h'].float()).abs().max().cpu()), 'router_logits_max_abs_change': float((aft_logits.float() - base_logits.float()).abs().max().cpu()), 'router_weights_max_abs_change': float((aft_w.float() - base_w.float()).abs().max().cpu()), 'router_indices_equal': bool(torch.equal(aft_i, base_i)), 'base_loss': probe['loss'], 'updated_loss': after['loss'], 'restored_loss': restored['loss'], 'restore_abs_loss_error': abs(restored['loss'] - probe['loss'])}

def summarize_equivalence(rows):
    gu = [r['gate_up_rel_error'] for r in rows if 'gate_up_rel_error' in r]
    dn = [r['down_rel_error'] for r in rows if 'down_rel_error' in r]

    def stats(xs):
        return {'count': len(xs), 'median': statistics.median(xs) if xs else None, 'max': max(xs) if xs else None}
    return {'gate_up': stats(gu), 'down': stats(dn)}

def recommend_branch(eq_summary, curve, probe_t, oracle_t, fallback, loads):
    active = sum((int(x > 0) for x in loads))
    total = len(loads)
    if total == 0 or active / total < 0.9:
        return ('BLOCKED', f'insufficient router coverage for cost curve: {active}/{total} experts active')
    rels = []
    for k in ('gate_up', 'down'):
        s = eq_summary[k]
        if s['median'] is not None:
            rels.append((s['median'], s['max']))
    eq_ok = bool(rels) and all((m <= 0.001 and x <= 0.01 for m, x in rels))
    if not eq_ok:
        return ('BLOCKED', 'local VJP does not meet BF16/FP16 provisional equivalence tolerance')
    by_m = {row['m']: row['timing']['median_s'] for row in curve}
    if 16 in by_m and 64 in by_m:
        total16 = probe_t + by_m[16]
        total64 = probe_t + by_m[64]
        reduction = 1.0 - total16 / total64 if total64 > 0 else 0.0
        if reduction >= 0.25 and fallback not in ('1', 'true', 'TRUE'):
            return ('EXPERT', f'm=16 reduces probe+localVJP wall time by {reduction:.3f} relative to m=64')
    return ('DATA', 'expert-count did not demonstrate the preregistered material cost reduction')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='allenai/OLMoE-1B-7B-0924')
    ap.add_argument('--revision', default='6d84c48581ece794365f2b8e9cfb043c68ade9c5')
    ap.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    ap.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    ap.add_argument('--seq-len', type=int, default=128)
    ap.add_argument('--output', default='WP1_H3_Preflight_CostBenchmark_v1p1.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == 'mps' and (not torch.backends.mps.is_available()):
        raise RuntimeError('MPS requested but unavailable')
    sha = HfApi().model_info(args.model, revision=args.revision).sha
    if sha != '6d84c48581ece794365f2b8e9cfb043c68ade9c5':
        raise RuntimeError(f'Resolved unexpected model SHA: {sha}')
    dtype = resolve_dtype(args.dtype)
    first_path = Path('WP1_H3_Preflight_Result_v1.json')
    first_bytes = first_path.read_bytes()
    first = json.loads(first_bytes)
    if first['resolved_sha'] != sha:
        raise RuntimeError('First-run model SHA does not match pinned revision')
    result = {'status': 'RUNNING', 'model': args.model, 'requested_revision': args.revision, 'resolved_sha': sha, 'device': str(device), 'dtype': args.dtype, 'environment': {'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__, 'mps_built': bool(torch.backends.mps.is_built()), 'mps_available': bool(torch.backends.mps.is_available()), 'PYTORCH_ENABLE_MPS_FALLBACK': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}, 'env_smoke': env_smoke(device), 'protocol': {'amendment': 'WP1_H3_Preflight_Amendment_v1.1.md', 'source_script': 'preflight_olmoe.py', 'warmups': 3, 'recorded_repeats': 10, 'bootstrap_seed': 20260928, 'first_run_json_sha256': hashlib.sha256(first_bytes).hexdigest()}, 'first_run_inherited': {'status': first['status'], 'equivalence': first['local_vjp_equivalence']['summary'], 'other_parameters_with_grad': first['oracle']['other_parameters_with_grad'], 'route_invariance': first['route_invariance'], 'fallback': first['environment']['PYTORCH_ENABLE_MPS_FALLBACK']}}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=sha)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(revision=sha, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    layer, mlp, experts = get_target(model)
    freeze_all(model)
    result['config'] = {'num_hidden_layers': int(model.config.num_hidden_layers), 'hidden_size': int(model.config.hidden_size), 'intermediate_size': int(model.config.intermediate_size), 'num_experts': int(model.config.num_experts), 'num_experts_per_tok': int(model.config.num_experts_per_tok), 'target_layer_index': int(model.config.num_hidden_layers - 1), 'gate_up_proj_shape': list(experts.gate_up_proj.shape), 'down_proj_shape': list(experts.down_proj.shape), 'expert_params_target_layer': int(experts.gate_up_proj.numel() + experts.down_proj.numel()), 'model_params': int(sum((p.numel() for p in model.parameters())))}
    result['memory_after_load'] = mem_snapshot(device)
    inputs = tiny_inputs(tokenizer, device, args.seq_len)
    probe = benchmark_probe_capture(model, mlp, inputs, device)
    result['probe'] = {'loss': probe['loss'], 'timing': probe['timing'], 'h_shape': list(probe['h'].shape), 'r_shape': list(probe['r'].shape), 'memory': mem_snapshot(device)}
    loads = torch.bincount(probe['route_i'].reshape(-1), minlength=int(experts.gate_up_proj.shape[0])).cpu().tolist()
    result['assignment_counts_by_expert'] = loads
    result['active_expert_count'] = sum((load > 0 for load in loads))
    result['total_assignment_count'] = sum(loads)
    result['families'] = {}
    for family in ('all_experts', 'active_only'):
        curve = benchmark_curve(experts, probe, loads, device, family)
        for row in curve:
            row['total_cost_median_s'] = probe['timing']['median_s'] + row['timing']['median_s']
            row['total_seconds_per_assignment'] = row['total_cost_median_s'] / row['assignments'] if row['assignments'] else None
        result['families'][family] = {'curve': curve, 'contrast': cost_contrast(curve, probe['timing']), 'cost_vs_assignments_fit': assignment_fit(curve), 'monotonicity': monotonicity(curve)}
    eq_summary = first['local_vjp_equivalence']['summary']
    equivalence_ok = all((eq_summary[name]['count'] > 0 and eq_summary[name]['median'] <= 0.001 and (eq_summary[name]['max'] <= 0.01) for name in ('gate_up', 'down'))) and (not first['oracle']['other_parameters_with_grad'])
    route = first['route_invariance']
    route_ok = route['router_indices_equal'] and route['restore_abs_loss_error'] == 0.0
    no_fallback = first['environment']['PYTORCH_ENABLE_MPS_FALLBACK'] == '0' and result['environment']['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    smoke_ok = result['env_smoke'][str(dtype).replace('torch.', '')]['pass']
    all_family = result['families']['all_experts']
    active_family = result['families']['active_only']
    material_cost = all((family['contrast']['reduction'] >= 0.25 and family['contrast']['reduction_95pct_interval'][0] > 0.2 for family in (all_family, active_family)))
    broadly_monotone = all((family['monotonicity']['broadly_monotone'] for family in (all_family, active_family)))
    if not (equivalence_ok and route_ok and no_fallback and smoke_ok):
        branch = 'BLOCKED'
        reason = 'first-run correctness, route invariance, BF16/MPS smoke, or no-fallback requirement failed'
    elif not broadly_monotone:
        branch = 'BLOCKED'
        reason = 'cost timing has assignment-ordered decreases beyond repeat noise'
    elif material_cost:
        branch = 'EXPERT'
        reason = 'both subset families meet the 25% median reduction and >20% lower 95% bootstrap bound'
    else:
        branch = 'DATA'
        reason = 'numerical equivalence holds, but reproducible material expert-count cost scaling was not established'
    result['decision_checks'] = {'equivalence_ok': equivalence_ok, 'route_invariance_ok': route_ok, 'no_cpu_fallback': no_fallback, 'bf16_mps_smoke_ok': smoke_ok, 'both_families_material_cost': material_cost, 'both_families_broadly_monotone': broadly_monotone}
    result['budget_recommendation'] = {'branch': branch, 'reason': reason}
    result['status'] = 'PASS_PREFLIGHT' if branch in ('EXPERT', 'DATA') else 'BLOCKED'
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'], 'resolved_sha': sha, 'active_expert_count': result['active_expert_count'], 'all_experts_contrast': all_family['contrast'], 'active_only_contrast': active_family['contrast'], 'budget_recommendation': result['budget_recommendation'], 'output': str(out.resolve())}, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
