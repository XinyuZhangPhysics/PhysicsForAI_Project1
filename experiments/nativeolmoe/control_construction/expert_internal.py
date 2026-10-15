# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Sparse per-document expert contributions; never allocate [tokens,64,hidden]."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from dataclasses import dataclass
import torch
from . import StopPilot
from .routing import route

@dataclass
class Contribution:
    expert: int
    tokens: torch.Tensor
    slots: torch.Tensor
    outputs: torch.Tensor

class LocalMoE:

    def __init__(self, module, hidden, residual=None):
        self.module = module
        self.shape = hidden.shape
        self.hidden = hidden.reshape(-1, hidden.shape[-1])
        self.residual = residual.reshape_as(self.hidden) if residual is not None else None
        self.n_experts = module.num_experts
        self.top_k = module.top_k
        self.normalized = module.norm_topk_prob
        with torch.inference_mode():
            self.logits = module.gate(self.hidden)
            self.routes = route(self.logits, self.top_k, self.normalized)
            self.contributions = []
            for expert in range(self.n_experts):
                slots, tokens = torch.where(self.routes.indices.T == expert)
                if len(tokens):
                    outputs = module.experts[expert](self.hidden[tokens]).detach()
                    self.contributions.append(Contribution(expert, tokens, slots, outputs))

    def native_output(self, gains=None, biases=None, frozen=True):
        if gains is None:
            gains = torch.zeros(self.n_experts, device=self.hidden.device)
        if biases is None:
            biases = torch.zeros(self.n_experts, device=self.hidden.device)
        routes = self.routes if frozen else route(self.logits.float() + biases, self.top_k, self.normalized)
        same_support = torch.equal(routes.indices, self.routes.indices)
        contributions = self.contributions
        if not same_support:
            contributions = []
            with torch.inference_mode():
                for expert in range(self.n_experts):
                    slots, tokens = torch.where(routes.indices.T == expert)
                    if len(tokens):
                        contributions.append(Contribution(expert, tokens, slots, self.module.experts[expert](self.hidden[tokens]).detach()))
        out = torch.zeros_like(self.hidden)
        for item in contributions:
            weight = routes.weights[item.tokens, item.slots].to(self.hidden.dtype)
            scaled = item.outputs * (1 + gains[item.expert]).to(self.hidden.dtype) * weight[:, None]
            out.index_add_(0, item.tokens, scaled.to(self.hidden.dtype))
        return (out, routes)

    def native_residual_output(self, gains=None, biases=None, frozen=True):
        out, routes = self.native_output(gains, biases, frozen)
        return (out if self.residual is None else self.residual + out, routes)

    def validate_baseline(self, native):
        reconstructed, _ = self.native_output()
        expected = native.reshape_as(reconstructed)
        mismatch = int((expected != reconstructed).sum().item())
        relative = (reconstructed.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
        if mismatch:
            raise StopPilot(f'Native BF16 sparse-output reconstruction not exact: {mismatch} entries; relative={relative.item()}')
        return dict(native_baseline_mismatch=mismatch, relative_error=relative.item())

    def amplitude(self):
        out = torch.zeros(self.n_experts, device=self.hidden.device)
        for item in self.contributions:
            w = self.routes.weights[item.tokens, item.slots].to(self.hidden.dtype)
            contribution = item.outputs * w[:, None]
            out[item.expert] = contribution.float().norm(dim=-1).sum() / len(self.hidden)
        return out.cpu().numpy()

    def differential(self, control):
        """Within-support local JVP of gain/bias controls with explicit FP32 differentiation.

        Native BF16 finite-difference checks certify it before it can pass a pilot gate.
        Bias softmax derivatives use the native pre-cast FP32 probabilities. Gain
        derivatives use the native cast routing weights. The residual bypass is constant.
        """
        gains, biases = (control[:self.n_experts], control[self.n_experts:])
        biases = biases - biases.mean()
        weights_for_mean = self.routes.probabilities
        if self.normalized:
            weights_for_mean = torch.zeros_like(weights_for_mean)
            weights_for_mean.scatter_(1, self.routes.indices, self.routes.weights)
        mean_bias = (weights_for_mean * biases).sum(-1)
        out = torch.zeros_like(self.hidden, dtype=torch.float32)
        for item in self.contributions:
            w_float = self.routes.weights[item.tokens, item.slots]
            w_gain = w_float.to(self.hidden.dtype).float()
            factor = gains[item.expert] * w_gain + w_float * (biases[item.expert] - mean_bias[item.tokens])
            out.index_add_(0, item.tokens, item.outputs.float() * factor[:, None])
        return out

    def route_freeze_check(self):
        from .routing import route_comparison
        before_idx, before_w = (self.routes.indices.clone(), self.routes.weights.clone())
        gains = torch.linspace(-0.02, 0.02, self.n_experts, device=self.hidden.device)
        baseline, first = self.native_output()
        perturbed, second = self.native_output(gains=gains, frozen=True)
        comparison = route_comparison(first, second)
        comparison['cache_mutations'] = int((before_idx != second.indices).sum().item() + (before_w != second.weights).sum().item())
        if any(comparison.values()):
            raise StopPilot('Exact route-freeze correctness failed.')
        common_gain = torch.cat((torch.ones(self.n_experts, device=self.hidden.device), torch.zeros(self.n_experts, device=self.hidden.device)))
        common_bias = torch.cat((torch.zeros(self.n_experts, device=self.hidden.device), torch.ones(self.n_experts, device=self.hidden.device)))
        private = common_gain.clone()
        private[:self.n_experts] -= private[:self.n_experts].mean()
        common_gain_error = (self.differential(common_gain) - baseline.float()).norm() / baseline.float().norm().clamp_min(1e-30)
        comparison.update(private_common_gain_null_norm=self.differential(private).norm().item(), common_router_bias_null_norm=self.differential(common_bias).norm().item(), common_gain_scale_identity_relative_error=common_gain_error.item(), common_gain_raw_response_norm=self.differential(common_gain).norm().item(), expert_response_norm=(perturbed.float() - baseline.float()).norm().item())
        return comparison
