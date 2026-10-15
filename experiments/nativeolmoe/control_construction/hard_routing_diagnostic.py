# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""v151 additive diagnostic maps. Existing v150 response methods stay untouched."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import itertools
import math
import numpy as np
import torch
from . import DOMAINS
from .routing import Routes
COMMIT = 'e5642d2d7b37c138775ddf59f685f91788e173b7'
AMPLITUDES = (0.04, 0.02, 0.01, 0.005, 0.0025)
SHADOW_AMPLITUDES = (0.001, 0.0005)
DIRECTIONS = ('candidate_0', 'gain_only_0', 'router_only_0', 'random_0', 'expert_permutation_0')
COMPONENTS = ('R_tangent', 'R_shadow', 'R_native_fixed_support_bf16', 'R_native_full_bf16', 'R_smoothNL', 'R_quantization', 'R_support_switch')

def select_diagnostic_documents(documents):
    """Domain declaration order, half 0 then 1; preserve first stored row per pair."""
    result = []
    for domain in DOMAINS:
        for half in (0, 1):
            row = next((d for d in documents if d['domain'] == domain and d['half'] == half), None)
            if row is None:
                raise ValueError('Missing frozen diagnostic domain/half.')
            result.append(dict(row))
    return result

def validate_amplitudes(native, shadow):
    if tuple(native) != AMPLITUDES or tuple(shadow) != SHADOW_AMPLITUDES:
        raise ValueError('Diagnostic amplitude ladder must remain frozen.')

def _blocks(local, control):
    control = torch.as_tensor(control, dtype=torch.float32, device=local.hidden.device)
    if control.shape != (2 * local.n_experts,):
        raise ValueError('Control block shape mismatch.')
    return (control[:local.n_experts], control[local.n_experts:])

def fixed_support_reweighted_routes(local, biases):
    logits = local.logits.float() + biases
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    indices = local.routes.indices.clone()
    weights = probabilities.gather(1, indices)
    if local.normalized:
        weights = weights / weights.sum(-1, keepdim=True)
    return Routes(indices, weights, probabilities, logits, local.routes.margin.clone())

def continuous_fixed_support_fp32(local, control):
    """Real cached BF16 experts cast to FP32; fully continuous weights and gains."""
    gains, biases = _blocks(local, control)
    routes = fixed_support_reweighted_routes(local, biases)
    out = torch.zeros_like(local.hidden, dtype=torch.float32)
    for item in local.contributions:
        weight = routes.weights[item.tokens, item.slots]
        scaled = item.outputs.float() * (1 + gains[item.expert]) * weight[:, None]
        out.index_add_(0, item.tokens, scaled)
    if local.residual is not None:
        out = local.residual.float() + out
    return (out, routes)

def native_bf16_fixed_support_reweighted(local, control):
    """Freeze indices only. Mirror v150 native gain/weight/accumulation and bypass."""
    gains, biases = _blocks(local, control)
    routes = fixed_support_reweighted_routes(local, biases)
    out = torch.zeros_like(local.hidden)
    for item in local.contributions:
        weight = routes.weights[item.tokens, item.slots].to(local.hidden.dtype)
        scaled = item.outputs * (1 + gains[item.expert]).to(local.hidden.dtype) * weight[:, None]
        out.index_add_(0, item.tokens, scaled.to(local.hidden.dtype))
    if local.residual is not None:
        out = local.residual + out
    return (out, routes)

def _norm(x):
    return float(x.detach().cpu().double().norm().item())

def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0 if numerator == 0 else None

def shadow_jvp_certification(local, direction):
    vector = torch.as_tensor(direction, dtype=torch.float32, device=local.hidden.device)
    tangent = local.differential(vector)
    derivatives = []
    for h in SHADOW_AMPLITUDES:
        plus = continuous_fixed_support_fp32(local, h * vector)[0]
        minus = continuous_fixed_support_fp32(local, -h * vector)[0]
        derivatives.append((plus - minus) / (2 * h))
    full, half = derivatives
    errors = [_ratio(_norm(fd - tangent), _norm(tangent)) for fd in derivatives]
    change = _ratio(_norm(half - full), _norm(half))
    return dict(analytic_vs_shadow_fd_relative_error=errors[0], analytic_vs_shadow_half_fd_relative_error=errors[1], shadow_half_step_discrepancy=change, analytic_norm=_norm(tangent), shadow_fd_norm=_norm(full), shadow_half_fd_norm=_norm(half), analytic_minus_shadow_fd_norm=_norm(full - tangent), half_minus_full_fd_norm=_norm(half - full), gate_pass=errors[0] is not None and change is not None and (errors[0] < 0.01) and (change < 0.01))

def _distribution(values):
    values = values.detach().float().cpu().numpy()
    if not len(values):
        return dict(count=0, mean=None, min=None, max=None, quantiles=None)
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()), max=float(values.max()), quantiles={str(q): float(np.quantile(values, q)) for q in (0, 0.25, 0.5, 0.75, 1)})

def support_switch_statistics(local, routes):
    base, current = (local.routes.indices, routes.indices)
    slot_changed = base != current
    changed = slot_changed.any(-1)
    set_changed = (base.sort(-1).values != current.sort(-1).values).any(-1)
    leaving, entering = ({}, {})
    b, c = (base.cpu().tolist(), current.cpu().tolist())
    for old, new in zip(b, c):
        for expert in set(old) - set(new):
            leaving[str(expert)] = leaving.get(str(expert), 0) + 1
        for expert in set(new) - set(old):
            entering[str(expert)] = entering.get(str(expert), 0) + 1
    return dict(any_topk_index_change_fraction=changed.float().mean().item(), support_set_replacement_fraction=set_changed.float().mean().item(), changed_slots=int(slot_changed.sum().item()), baseline_margin_mean=local.routes.margin.float().mean().item(), baseline_small_margin_masses={str(delta): (local.routes.margin < delta).float().mean().item() for delta in sorted(AMPLITUDES)}, switched_token_baseline_margin_distribution=_distribution(local.routes.margin[changed]), set_replaced_token_baseline_margin_distribution=_distribution(local.routes.margin[set_changed]), leaving_expert_counts=leaving, entering_expert_counts=entering)

def bf16_gain_quantization_report(gain_direction, h):
    gain = torch.as_tensor(gain_direction, dtype=torch.float32).detach().cpu()
    plus = (1 + h * gain).to(torch.bfloat16).float()
    minus = (1 - h * gain).to(torch.bfloat16).float()
    q = (plus - minus) / (2 * h)
    gnorm, qnorm = (_norm(gain), _norm(q))
    cosine = float((gain.double().cpu() * q.double().cpu()).sum().item()) / (gnorm * qnorm) if gnorm * qnorm else None
    return dict(h=h, coordinate_count=gain.numel(), changing_either_sign=int(((plus != 1) | (minus != 1)).sum().item()), relative_direction_error=_ratio(_norm(q - gain), gnorm), effective_direction_norm_ratio=_ratio(qnorm, gnorm), cosine=cosine, gain_direction_norm=gnorm, effective_direction_norm=qnorm, zero_gain_block=gnorm == 0)

def routing_weight_quantization_report(plus_routes, minus_routes, baseline, h):
    plus, minus = (plus_routes.weights, minus_routes.weights)
    p, m, b = (plus.to(torch.bfloat16).float(), minus.to(torch.bfloat16).float(), baseline.weights.to(torch.bfloat16).float())
    q, ref = ((p - m) / (2 * h), (plus - minus) / (2 * h))
    return dict(selected_slot_count=plus.numel(), changing_either_sign=int(((p != b) | (m != b)).sum().item()), unquantized_centered_weight_direction_norm=_norm(ref), quantized_centered_weight_direction_norm=_norm(q), relative_direction_error=_ratio(_norm(q - ref), _norm(ref)))

def reduce_decomposition(vectors):
    """FP64 CPU reductions only; hidden/expert evaluations remain on original device."""
    flat = {name: x.detach().cpu().double().reshape(-1) for name, x in vectors.items()}
    norms = {name: _norm(v) for name, v in flat.items()}
    cosine = {}
    for a, b in itertools.combinations(COMPONENTS, 2):
        denom = norms[a] * norms[b]
        cosine[a + '__' + b] = float(torch.dot(flat[a], flat[b]).item()) / denom if denom else None
    closure = flat['R_native_full_bf16'] - sum((flat[name] for name in ('R_tangent', 'R_smoothNL', 'R_quantization', 'R_support_switch')))
    error = _norm(closure)
    scale = sum((norms[name] for name in ('R_tangent', 'R_smoothNL', 'R_quantization', 'R_support_switch')))
    budget = 16 * np.finfo(np.float32).eps * max(scale, 1e-30)
    return dict(norms=norms, rms={name: n / math.sqrt(flat[name].numel()) for name, n in norms.items()}, pairwise_cosines=cosine, closure_norm=error, closure_roundoff_budget=budget, closure_relative_to_full=_ratio(error, norms['R_native_full_bf16']), closure_pass=error <= budget)

def response_decomposition(local, direction, h):
    gains, biases = _blocks(local, direction)
    vector = torch.cat((gains, biases))
    shadow_plus, sp = continuous_fixed_support_fp32(local, h * vector)
    shadow_minus, sm = continuous_fixed_support_fp32(local, -h * vector)
    fixed_plus, fp = native_bf16_fixed_support_reweighted(local, h * vector)
    fixed_minus, fm = native_bf16_fixed_support_reweighted(local, -h * vector)
    full_plus, rp = local.native_residual_output(h * gains, h * biases, frozen=False)
    full_minus, rm = local.native_residual_output(-h * gains, -h * biases, frozen=False)
    tangent = h * local.differential(vector)
    shadow = (shadow_plus - shadow_minus) / 2
    fixed = (fixed_plus.float() - fixed_minus.float()) / 2
    full = (full_plus.float() - full_minus.float()) / 2
    vectors = dict(R_tangent=tangent, R_shadow=shadow, R_native_fixed_support_bf16=fixed, R_native_full_bf16=full, R_smoothNL=shadow - tangent, R_quantization=fixed - shadow, R_support_switch=full - fixed)
    reduced = reduce_decomposition(vectors)
    reduced.update(h=h, token_count=local.hidden.shape[0], hidden_size=local.hidden.shape[1], plus_support=support_switch_statistics(local, rp), minus_support=support_switch_statistics(local, rm), union_index_change_fraction=((rp.indices != local.routes.indices).any(-1) | (rm.indices != local.routes.indices).any(-1)).float().mean().item(), routing_weight_quantization=routing_weight_quantization_report(fp, fm, local.routes, h), fixed_indices_unchanged=torch.equal(fp.indices, local.routes.indices) and torch.equal(fm.indices, local.routes.indices))
    return reduced

def hard_routing_scaling_report(rows):
    """Equal domains/documents, per-document TH-normalized power. Adjacent slopes only."""
    reports = []
    for layer in (7, 15):
        for direction in ('candidate_0', 'router_only_0'):
            ladder = []
            for h in AMPLITUDES:
                group = [r for r in rows if r['layer'] == layer and r['direction'] == direction and (r['h'] == h)]
                if len(group) != 14:
                    raise ValueError('Incomplete diagnostic grid for scaling.')

                def mean(field):
                    return float(np.mean([np.mean([field(r) for r in group if r['domain'] == domain]) for domain in DOMAINS]))
                ladder.append(dict(h=h, S_union=mean(lambda r: r['union_index_change_fraction']), S_plus=mean(lambda r: r['plus_support']['any_topk_index_change_fraction']), S_minus=mean(lambda r: r['minus_support']['any_topk_index_change_fraction']), S_set_plus=mean(lambda r: r['plus_support']['support_set_replacement_fraction']), S_set_minus=mean(lambda r: r['minus_support']['support_set_replacement_fraction']), D_switch=math.sqrt(mean(lambda r: r['rms']['R_support_switch'] ** 2)), D_full=math.sqrt(mean(lambda r: r['rms']['R_native_full_bf16'] ** 2))))
            slopes = []
            for a, b in zip(ladder, ladder[1:]):
                values = {key: math.log(b[key] / a[key]) / math.log(b['h'] / a['h']) if a[key] > 0 and b[key] > 0 else None for key in ('S_union', 'S_plus', 'S_minus', 'D_switch', 'D_full')}
                slopes.append(dict(h_from=a['h'], h_to=b['h'], local_log_slopes=values))
            reports.append(dict(layer=layer, direction=direction, ladder=ladder, adjacent_slopes=slopes))
    return dict(weighting='equal domains, two equal documents/domain, norm squared divided by T*H', switch_definition='S_union: any ordered Top-K index change in either sign; S_plus/minus and set replacements separate', universal_exponent_claim=None, measurements=reports)
