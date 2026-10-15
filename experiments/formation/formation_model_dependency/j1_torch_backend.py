# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""PyTorch backend for the pinned J1 model. Dependency unavailable in this run.

This module has no automatic execution, no audit-bank loader, and no training loop.
Its preflight use still requires an independent real-model test before freeze.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import torch
from torch import nn
from torch.nn import functional as F
PARENTS = 4
HIDDEN = 576
INTERMEDIATE = 1536
LAYER_INDEX = 19

def parent_vectors(seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(PARENTS, HIDDEN, generator=gen, dtype=torch.float32) * 0.02).to(dtype)

class Expert(nn.Module):

    def __init__(self, dense: nn.Module, sl: slice, g: int):
        super().__init__()
        m = INTERMEDIATE // g
        kwargs = {'bias': False, 'dtype': dense.gate_proj.weight.dtype, 'device': dense.gate_proj.weight.device}
        self.gate_proj = nn.Linear(HIDDEN, m, **kwargs)
        self.up_proj = nn.Linear(HIDDEN, m, **kwargs)
        self.down_proj = nn.Linear(m, HIDDEN, **kwargs)
        with torch.no_grad():
            self.gate_proj.weight.copy_(dense.gate_proj.weight[sl])
            self.up_proj.weight.copy_(dense.up_proj.weight[sl])
            self.down_proj.weight.copy_(dense.down_proj.weight[:, sl] * g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = F.silu(F.linear(x, self.gate_proj.weight)) * F.linear(x, self.up_proj.weight)
        return F.linear(phi, self.down_proj.weight)

class CoTrainMoE(nn.Module):
    """All 4g experts are independently addressable after S2-copy initialization."""

    def __init__(self, dense: nn.Module, g: int, seed: int):
        super().__init__()
        if g not in (1, 2, 4, 8) or dense.gate_proj.bias is not None:
            raise ValueError('unsupported S2 upcycle')
        self.g = g
        m = INTERMEDIATE // g
        self.experts = nn.ModuleList((Expert(dense, slice(q * m, (q + 1) * m), g) for _ in range(PARENTS) for q in range(g)))
        self.router = nn.Linear(HIDDEN, PARENTS * g, bias=False, dtype=dense.gate_proj.weight.dtype, device=dense.gate_proj.weight.device)
        with torch.no_grad():
            self.router.weight.copy_(parent_vectors(seed, self.router.weight.dtype).to(self.router.weight.device).repeat_interleave(g, dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, HIDDEN)
        p = torch.softmax(self.router(flat), dim=-1)
        result = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            result = result + p[:, i:i + 1] * expert(flat)
        return result.reshape_as(x)

def upcycle(model: nn.Module, g: int, seed: int) -> CoTrainMoE:
    model.requires_grad_(False)
    layer = model.model.layers[LAYER_INDEX]
    moe = CoTrainMoE(layer.mlp, g, seed)
    layer.mlp = moe
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if len(names) != 1 + 3 * 4 * g or not all((f'layers.{LAYER_INDEX}.mlp.' in n for n in names)):
        raise AssertionError('trainable block mismatch')
    return moe

def initial_source_gradients(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None):
    """One LM backward for four inherited 256-token sequences, no optimizer state."""
    if input_ids.shape != (4, 256):
        raise ValueError('source batch must be four complete 256-token sequences')
    moe = model.model.layers[LAYER_INDEX].mlp
    params = [moe.router.weight]
    for expert in moe.experts:
        params.extend((expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight))
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    gradients = torch.autograd.grad(output.loss, params, allow_unused=False)
    directions = [(-grad).detach().clone() for grad in gradients]
    if not all((torch.isfinite(d).all() for d in directions)):
        raise FloatingPointError('nonfinite source gradient')
    return (float(output.loss.detach()), directions)

def float64_q_jvp(router: torch.Tensor, direction: torch.Tensor, hs: torch.Tensor, ht: torch.Tensor):
    """Explicit bank-normalized router path, independent torch.func JVP."""
    router, direction, hs, ht = (x.detach().to(device='cpu', dtype=torch.float64) for x in (router, direction, hs, ht))

    def q(w):
        raw = torch.softmax(hs @ w.T, dim=-1) * torch.softmax(ht @ w.T, dim=-1)
        center = raw - raw.mean(dim=-1, keepdim=True)
        power = (center.square().sum(dim=-1) / (w.shape[0] - 1)).mean()
        return center / power.sqrt()
    return torch.func.jvp(q, (router,), (direction,))

def float64_kappa_jvp(gate: torch.Tensor, up: torch.Tensor, dgate: torch.Tensor, dup: torch.Tensor, hs: torch.Tensor, ht: torch.Tensor):
    """Primary S2 geometry; down projection has zero direct tangent here."""
    gate, up, dgate, dup, hs, ht = (x.detach().to(device='cpu', dtype=torch.float64) for x in (gate, up, dgate, dup, hs, ht))

    def k(wg, wu):
        ps = F.silu(torch.einsum('pd,hmd->phm', hs, wg)) * torch.einsum('pd,hmd->phm', hs, wu)
        pt = F.silu(torch.einsum('pd,hmd->phm', ht, wg)) * torch.einsum('pd,hmd->phm', ht, wu)
        raw = (ps * pt).mean(dim=-1)
        return raw - raw.mean(dim=-1, keepdim=True)
    return torch.func.jvp(k, (gate, up), (dgate, dup))

def float64_joint_response_jvp(q: torch.Tensor, k: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, group: torch.Tensor, g: int):
    """Independent torch.func JVP of the joint signed response composition."""
    q, k, dq, dk = (x.detach().to(device='cpu', dtype=torch.float64) for x in (q, k, dq, dk))
    group = group.detach().to(device='cpu', dtype=torch.long)

    def signed_response(a, b):
        return torch.stack([torch.sum(a[group == mu] * b[group == mu], dim=-1).mean() for mu in range(12)])
    return torch.func.jvp(signed_response, (q, k), (dq, dk))
