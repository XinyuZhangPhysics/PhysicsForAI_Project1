# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""v152 additive clean shadow tangent and independent numerical oracles."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import itertools
import torch
from .hard_routing_diagnostic import SHADOW_AMPLITUDES, continuous_fixed_support_fp32, fixed_support_reweighted_routes, native_bf16_fixed_support_reweighted, reduce_decomposition, support_switch_statistics, routing_weight_quantization_report

def norm(x):
    return x.detach().cpu().double().norm().item()

def relative(a, reference):
    numerator, denominator = (norm(a), norm(reference))
    return numerator / denominator if denominator else 0.0 if numerator == 0 else None

def cosine(a, b):
    a, b = (a.detach().cpu().double().flatten(), b.detach().cpu().double().flatten())
    denominator = a.norm().item() * b.norm().item()
    return torch.dot(a, b).item() / denominator if denominator else None

def direction_tensor(local, control):
    v = torch.as_tensor(control, dtype=torch.float32, device=local.hidden.device)
    if v.shape != (2 * local.n_experts,):
        raise ValueError('Exact gain/router control dimension required.')
    return v

def continuous_fixed_support_jvp_fp32(local, control):
    """Exact tangent of v151 FP32 shadow; the gain weight is never BF16-cast."""
    v = direction_tensor(local, control)
    gains, biases = (v[:local.n_experts], v[local.n_experts:])
    routes = fixed_support_reweighted_routes(local, torch.zeros_like(biases))
    if local.normalized:
        mean = (routes.weights * biases[routes.indices]).sum(-1)
    else:
        mean = (routes.probabilities * biases).sum(-1)
    out = torch.zeros_like(local.hidden, dtype=torch.float32)
    for item in local.contributions:
        weight = routes.weights[item.tokens, item.slots]
        coefficient = weight * (gains[item.expert] + biases[item.expert] - mean[item.tokens])
        out.index_add_(0, item.tokens, item.outputs.float() * coefficient[:, None])
    return out

class FixedSupportADOracle:
    """Independent differentiable forward graph; constants cloned out of inference mode."""

    def __init__(self, local, device=None):
        self.device = torch.device(device or local.hidden.device)
        self.logits = local.logits.detach().to(device=self.device, dtype=torch.float32).clone()
        self.indices = local.routes.indices.detach().to(self.device).clone()
        self.shape = tuple(local.hidden.shape)
        self.n_experts = local.n_experts
        self.normalized = local.normalized
        self.experts = [(item.expert, item.tokens.detach().to(self.device).clone(), item.slots.detach().to(self.device).clone(), item.outputs.detach().to(device=self.device, dtype=torch.float32).clone()) for item in local.contributions]

    def forward(self, control):
        gains, biases = (control[:self.n_experts], control[self.n_experts:])
        probabilities = torch.softmax(self.logits + biases, dim=-1, dtype=torch.float32)
        weights = probabilities.gather(1, self.indices)
        if self.normalized:
            weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        for expert, tokens, slots, outputs in self.experts:
            weighted = outputs * (1 + gains[expert]) * weights[tokens, slots, None]
            out = out.index_add(0, tokens, weighted)
        return out

    def jvp(self, direction):
        zero = torch.zeros(2 * self.n_experts, dtype=torch.float32, device=self.device)
        vector = torch.as_tensor(direction, dtype=torch.float32, device=self.device)
        _, jvp = torch.autograd.functional.jvp(self.forward, zero, vector, create_graph=False, strict=True)
        return jvp.detach()

def independent_ad_jvp(local, direction, oracle=None):
    oracle = oracle or FixedSupportADOracle(local)
    try:
        return (oracle.jvp(direction), dict(device=str(oracle.device), cached_tensors_moved_to_cpu=False, fallback_reason=None))
    except (NotImplementedError, RuntimeError) as exc:
        message = str(exc).lower()
        unsupported = 'not implemented' in message or 'not supported' in message or 'unsupported' in message
        if oracle.device.type != 'mps' or not unsupported:
            raise
        cpu = FixedSupportADOracle(local, device='cpu')
        return (cpu.jvp(direction), dict(device='cpu', cached_tensors_moved_to_cpu=True, fallback_reason=str(exc), model_reevaluated_on_cpu=False))

def baseline_free_centered_fd_fp32(local, control, h):
    """Coefficient-first A_i+B_i; never forms or subtracts complete residual outputs."""
    if h not in SHADOW_AMPLITUDES:
        raise ValueError('Only frozen repair shadow amplitudes allowed.')
    vector = direction_tensor(local, control)
    gains, biases = (vector[:local.n_experts], vector[local.n_experts:])
    plus = fixed_support_reweighted_routes(local, h * biases)
    minus = fixed_support_reweighted_routes(local, -h * biases)
    out = torch.zeros_like(local.hidden, dtype=torch.float32)
    for item in local.contributions:
        wp = plus.weights[item.tokens, item.slots]
        wm = minus.weights[item.tokens, item.slots]
        coefficient = (wp - wm) / (2 * h) + gains[item.expert] * (wp + wm) / 2
        out.index_add_(0, item.tokens, item.outputs.float() * coefficient[:, None])
    return out

def legacy_full_output_fd_fp32(local, control, h):
    vector = direction_tensor(local, control)
    plus = continuous_fixed_support_fp32(local, h * vector)[0]
    minus = continuous_fixed_support_fp32(local, -h * vector)[0]
    return (plus - minus) / (2 * h)

def zero_control_shadow_reproducibility(local, repeats=4):
    if repeats < 4:
        raise ValueError('At least four zero-control repetitions are required.')
    zero = torch.zeros(2 * local.n_experts, dtype=torch.float32, device=local.hidden.device)
    outputs = [continuous_fixed_support_fp32(local, zero)[0] for _ in range(repeats)]
    comparisons = []
    for i, j in itertools.combinations(range(repeats), 2):
        comparisons.append(dict(first=i, second=j, relative_difference=relative(outputs[i] - outputs[j], outputs[j]), absolute_difference_norm=norm(outputs[i] - outputs[j])))
    return dict(repetitions=repeats, pairwise_comparisons=comparisons, max_pairwise_relative_difference=max((c['relative_difference'] for c in comparisons)), baseline_norms=[norm(v) for v in outputs])

def shadow_oracle_cell_v152(local, control, oracle=None):
    vector = direction_tensor(local, control)
    soft = continuous_fixed_support_jvp_fp32(local, vector)
    ad, device = independent_ad_jvp(local, control, oracle)
    ad_error = relative(soft.detach().cpu() - ad.cpu(), ad.cpu())
    fd = [baseline_free_centered_fd_fp32(local, vector, h) for h in SHADOW_AMPLITUDES]
    legacy = [legacy_full_output_fd_fp32(local, vector, h) for h in SHADOW_AMPLITUDES]
    hybrid = local.differential(vector)
    fd_errors = [relative(v - soft, soft) for v in fd]
    half_error = relative(fd[1] - fd[0], fd[1])
    legacy_errors = [relative(v - soft, soft) for v in legacy]
    result = dict(E_AD=ad_error, baseline_free_fd_errors=fd_errors, baseline_free_half_step_discrepancy=half_error, analytic_norm=norm(soft), AD_norm=norm(ad), baseline_free_fd_norms=[norm(v) for v in fd], legacy_full_output_fd_norms=[norm(v) for v in legacy], legacy_full_output_fd_errors=legacy_errors, legacy_full_output_half_step_discrepancy=relative(legacy[1] - legacy[0], legacy[1]), ad_oracle=device, ad_gate_pass=ad_error is not None and ad_error < 0.001, baseline_free_gate_pass=fd_errors[0] is not None and half_error is not None and (fd_errors[0] < 0.01) and (half_error < 0.01))
    cancellation = dict(legacy_full_output_fd_minus_baseline_free_norms=[norm(a - b) for a, b in zip(legacy, fd)], legacy_full_output_fd_minus_baseline_free_relative_to_soft=[relative(a - b, soft) for a, b in zip(legacy, fd)], legacy_full_output_fd_errors=legacy_errors, legacy_full_output_half_step_discrepancy=result['legacy_full_output_half_step_discrepancy'], gate_usage='descriptive only; never used for repaired gate')
    hybrid_report = dict(legacy_hybrid_tangent_norm=norm(hybrid), clean_soft_tangent_norm=norm(soft), difference_norm=norm(hybrid - soft), difference_relative_to_soft=relative(hybrid - soft, soft), legacy_to_soft_norm_ratio=norm(hybrid) / norm(soft) if norm(soft) else None, cosine=cosine(hybrid, soft), legacy_function_modified=False)
    return (result, cancellation, hybrid_report, soft)

def cpu_fp64_sentinel(local, control, soft):
    """Promote captured constants only; no model/expert reevaluation."""
    logits = local.logits.detach().cpu().double()
    indices = local.routes.indices.detach().cpu()
    vector = torch.as_tensor(control, dtype=torch.float32, device='cpu').double()
    gains, biases = (vector[:local.n_experts], vector[local.n_experts:])
    pieces = [(item.expert, item.tokens.cpu(), item.slots.cpu(), item.outputs.detach().cpu().double()) for item in local.contributions]
    residual = local.residual.detach().cpu().double() if local.residual is not None else None

    def weights(sign):
        probs = torch.softmax(logits + sign * biases, dim=-1)
        selected = probs.gather(1, indices)
        if local.normalized:
            selected = selected / selected.sum(-1, keepdim=True)
        return (probs, selected)

    def forward(sign):
        _, w = weights(sign)
        out = torch.zeros(local.hidden.shape, dtype=torch.float64, device='cpu')
        for expert, tokens, slots, e in pieces:
            out.index_add_(0, tokens, e * (1 + sign * gains[expert]) * w[tokens, slots, None])
        return out if residual is None else residual + out
    p, w = weights(0.0)
    mean = (w * biases[indices]).sum(-1) if local.normalized else (p * biases).sum(-1)
    analytic = torch.zeros(local.hidden.shape, dtype=torch.float64, device='cpu')
    for expert, tokens, slots, e in pieces:
        analytic.index_add_(0, tokens, e * (w[tokens, slots] * (gains[expert] + biases[expert] - mean[tokens]))[:, None])
    finite = [(forward(h) - forward(-h)) / (2 * h) for h in SHADOW_AMPLITUDES]
    reference = soft.detach().cpu().double()
    errors = [relative(v - reference, reference) for v in finite]
    exact_error = relative(analytic - reference, reference)
    return dict(device='cpu', dtype='float64', constants_promoted_only=True, model_reevaluated=False, analytic_fp64_vs_clean_soft_relative_error=exact_error, fp64_centered_fd_vs_clean_soft_relative_errors=errors, fp64_centered_half_step_discrepancy=relative(finite[1] - finite[0], finite[1]), analytic_fp64_norm=norm(analytic), fp64_fd_norms=[norm(v) for v in finite], gate_pass=exact_error is not None and exact_error < 0.01 and all((e is not None and e < 0.01 for e in errors)))

def response_decomposition_clean_soft_v152(local, control, h):
    """v151 Sections 7–10 maps/ladder; only tangent object repaired under a new name."""
    vector = direction_tensor(local, control)
    gains, biases = (vector[:local.n_experts], vector[local.n_experts:])
    shadow_plus, _ = continuous_fixed_support_fp32(local, h * vector)
    shadow_minus, _ = continuous_fixed_support_fp32(local, -h * vector)
    fixed_plus, fp = native_bf16_fixed_support_reweighted(local, h * vector)
    fixed_minus, fm = native_bf16_fixed_support_reweighted(local, -h * vector)
    full_plus, rp = local.native_residual_output(h * gains, h * biases, frozen=False)
    full_minus, rm = local.native_residual_output(-h * gains, -h * biases, frozen=False)
    tangent = h * continuous_fixed_support_jvp_fp32(local, vector)
    shadow = (shadow_plus - shadow_minus) / 2
    fixed = (fixed_plus.float() - fixed_minus.float()) / 2
    full = (full_plus.float() - full_minus.float()) / 2
    reduced = reduce_decomposition(dict(R_tangent=tangent, R_shadow=shadow, R_native_fixed_support_bf16=fixed, R_native_full_bf16=full, R_smoothNL=shadow - tangent, R_quantization=fixed - shadow, R_support_switch=full - fixed))
    reduced.update(h=h, token_count=local.hidden.shape[0], hidden_size=local.hidden.shape[1], plus_support=support_switch_statistics(local, rp), minus_support=support_switch_statistics(local, rm), union_index_change_fraction=((rp.indices != local.routes.indices).any(-1) | (rm.indices != local.routes.indices).any(-1)).float().mean().item(), routing_weight_quantization=routing_weight_quantization_report(fp, fm, local.routes, h), fixed_indices_unchanged=torch.equal(fp.indices, local.routes.indices) and torch.equal(fm.indices, local.routes.indices), tangent_definition='h * continuous_fixed_support_jvp_fp32 (v152 clean-soft)', shadow_forward_definition='unchanged v151 continuous_fixed_support_fp32; centered finite response, not divided by h')
    return reduced
