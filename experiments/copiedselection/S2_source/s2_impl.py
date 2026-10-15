# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""EXP-S2 preflight-only SwiGLU adaptation of the frozen EXP-S1 construction.

The four parent copies and child marginalization deliberately follow the frozen
S1 implementation.  There is no scientific-audit entry point in this module.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'
MODEL_REVISION = '28e66ca6931668447a3bac213f23d990ad3b0e2b'
DATASET_ID = 'DKYoon/SlimPajama-6B'
DATASET_REVISION = 'b5f90f419b7489cdba26fdbc8c022fcb5562f968'
LAYER_1INDEXED = 20
LAYER_INDEX = LAYER_1INDEXED - 1
HIDDEN = 576
INTERMEDIATE = 1536
PARENTS = 4
ARMS = {'S1': 1, 'S2': 2, 'S4': 4, 'S8': 8}

def parent_vectors(seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(PARENTS, HIDDEN, generator=gen, dtype=torch.float32) * 0.02).to(dtype)

class FrozenChild(nn.Module):

    def __init__(self, dense: nn.Module, sl: slice, g: int):
        super().__init__()
        m = INTERMEDIATE // g
        w = dense.gate_proj.weight
        self.gate_proj = nn.Linear(HIDDEN, m, bias=False, dtype=w.dtype, device=w.device)
        self.up_proj = nn.Linear(HIDDEN, m, bias=False, dtype=w.dtype, device=w.device)
        self.down_proj = nn.Linear(m, HIDDEN, bias=False, dtype=w.dtype, device=w.device)
        with torch.no_grad():
            self.gate_proj.weight.copy_(dense.gate_proj.weight[sl])
            self.up_proj.weight.copy_(dense.up_proj.weight[sl])
            self.down_proj.weight.copy_(dense.down_proj.weight[:, sl] * g)
        self.requires_grad_(False)

class SoftRouterSwiGLU(nn.Module):

    def __init__(self, dense: nn.Module, g: int, parent: torch.Tensor):
        super().__init__()
        assert g in (1, 2, 4, 8)
        assert tuple(parent.shape) == (PARENTS, HIDDEN)
        assert dense.gate_proj.bias is None and dense.up_proj.bias is None
        assert dense.down_proj.bias is None
        m = INTERMEDIATE // g
        self.g = g
        self.experts = nn.ModuleList((FrozenChild(dense, slice(q * m, (q + 1) * m), g) for _ in range(PARENTS) for q in range(g)))
        self.router = nn.Linear(HIDDEN, PARENTS * g, bias=False, dtype=dense.gate_proj.weight.dtype, device=dense.gate_proj.weight.device)
        with torch.no_grad():
            self.router.weight.copy_(parent.repeat_interleave(g, dim=0))
        self.act_fn = dense.act_fn

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.router(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, HIDDEN)
        child_prob = self.probabilities(flat).reshape(-1, PARENTS, self.g).sum(dim=1)
        result = torch.zeros_like(flat)
        for q in range(self.g):
            e = self.experts[q]
            hidden = self.act_fn(F.linear(flat, e.gate_proj.weight)) * F.linear(flat, e.up_proj.weight)
            result = result + child_prob[:, q:q + 1] * F.linear(hidden, e.down_proj.weight)
        return result.reshape(x.shape)

def upcycle(model: nn.Module, g: int, seed: int) -> SoftRouterSwiGLU:
    model.requires_grad_(False)
    layer = model.model.layers[LAYER_INDEX]
    moe = SoftRouterSwiGLU(layer.mlp, g, parent_vectors(seed, layer.mlp.gate_proj.weight.dtype))
    layer.mlp = moe
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == [f'model.layers.{LAYER_INDEX}.mlp.router.weight'], trainable
    return moe

def projectors(g: int, dtype: torch.dtype=torch.float64) -> dict[str, torch.Tensor]:
    j4 = torch.full((PARENTS, PARENTS), 1 / PARENTS, dtype=dtype)
    p4 = torch.eye(PARENTS, dtype=dtype) - j4
    jg = torch.full((g, g), 1 / g, dtype=dtype)
    pg = torch.eye(g, dtype=dtype) - jg
    return {'A': torch.kron(p4, jg), 'C': torch.kron(j4, pg), 'M': torch.kron(p4, pg)}

def initial_kappa(dense: nn.Module, x: torch.Tensor, y: torch.Tensor, g: int) -> torch.Tensor:
    """S1's per-child expert-standard feature Gram, with SwiGLU features."""
    gate_x = dense.act_fn(F.linear(x, dense.gate_proj.weight))
    gate_y = dense.act_fn(F.linear(y, dense.gate_proj.weight))
    up_x = F.linear(x, dense.up_proj.weight)
    up_y = F.linear(y, dense.up_proj.weight)
    phi_x = (gate_x * up_x).reshape(-1, g, INTERMEDIATE // g)
    phi_y = (gate_y * up_y).reshape(-1, g, INTERMEDIATE // g)
    child = (phi_x * phi_y).mean(dim=-1)
    all_experts = child[:, None, :].expand(-1, PARENTS, -1).reshape(-1, PARENTS * g)
    return all_experts - all_experts.mean(dim=-1, keepdim=True)

def q_standard(moe: SoftRouterSwiGLU, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    raw = moe.probabilities(x) * moe.probabilities(y)
    centered = raw - raw.mean(dim=-1, keepdim=True)
    power = (centered.square().sum(dim=-1) / (PARENTS * moe.g - 1)).mean()
    assert bool(power > 0)
    return centered / power.sqrt()

def tensor_digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def frozen_digests(model: nn.Module) -> dict[str, str]:
    return {n: tensor_digest(p) for n, p in model.named_parameters() if not p.requires_grad}

def buffer_digests(model: nn.Module) -> dict[str, str]:
    return {n: tensor_digest(b) for n, b in model.named_buffers()}

def assert_router_optimizer_only(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(trainable) == 1
    groups = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert groups == trainable
    assert {id(p) for p in optimizer.state}.issubset(trainable)
