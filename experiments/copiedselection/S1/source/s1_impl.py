# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import copy
import torch
from torch import nn
from torch.nn import functional as F
ARMS = {'S1': 1, 'S2': 2, 'S4': 4, 'S8': 8}

class FrozenChild(nn.Module):

    def __init__(self, dense: nn.Module, sl: slice, g: int):
        super().__init__()
        up, down = (dense.dense_h_to_4h, dense.dense_4h_to_h)
        self.up = nn.Linear(512, 2048 // g, dtype=up.weight.dtype, device=up.weight.device)
        self.down = nn.Linear(2048 // g, 512, dtype=up.weight.dtype, device=up.weight.device)
        with torch.no_grad():
            self.up.weight.copy_(up.weight[sl])
            self.up.bias.copy_(up.bias[sl])
            self.down.weight.copy_(down.weight[:, sl] * g)
            self.down.bias.copy_(down.bias)
        for p in self.parameters():
            p.requires_grad_(False)

class SoftRouterMLP(nn.Module):

    def __init__(self, dense: nn.Module, g: int, parent: torch.Tensor):
        super().__init__()
        assert g in (1, 2, 4, 8) and tuple(parent.shape) == (4, 512)
        self.g = g
        m = 2048 // g
        self.experts = nn.ModuleList((FrozenChild(dense, slice(q * m, (q + 1) * m), g) for _ in range(4) for q in range(g)))
        self.router = nn.Linear(512, 4 * g, bias=False, dtype=dense.dense_h_to_4h.weight.dtype, device=dense.dense_h_to_4h.weight.device)
        with torch.no_grad():
            self.router.weight.copy_(parent.repeat_interleave(g, dim=0))
        self.act = copy.deepcopy(dense.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, 512)
        probs = self.router(flat).softmax(-1).reshape(-1, 4, self.g).sum(dim=1)
        ys = []
        for q in range(self.g):
            e = self.experts[q]
            y = F.linear(self.act(F.linear(flat, e.up.weight, e.up.bias)), e.down.weight, None)
            ys.append(y * probs[:, q:q + 1])
        result = torch.stack(ys, dim=0).sum(dim=0) + self.experts[0].down.bias
        return result.reshape(x.shape)

def parent_vectors(seed: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device='cpu').manual_seed(seed)
    return (torch.randn(4, 512, generator=gen, dtype=torch.float32) * 0.02).to(dtype)

def upcycle(model: nn.Module, g: int, seed: int) -> SoftRouterMLP:
    for p in model.parameters():
        p.requires_grad_(False)
    layer = model.gpt_neox.layers[3]
    moe = SoftRouterMLP(layer.mlp, g, parent_vectors(seed, layer.mlp.dense_h_to_4h.weight.dtype))
    layer.mlp = moe
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ['gpt_neox.layers.3.mlp.router.weight']
    return moe

def three_functions(dense: nn.Module, moe: SoftRouterMLP, x: torch.Tensor):
    flat = x.reshape(-1, 512)
    F_dense = dense(flat)
    blocks = []
    for q in range(moe.g):
        e = moe.experts[q]
        blocks.append(F.linear(moe.act(F.linear(flat, e.up.weight, e.up.bias)), e.down.weight, e.down.bias) / moe.g)
    F_block = torch.stack(blocks).sum(dim=0)
    F_soft_moe = moe(flat)
    return (F_dense, F_block, F_soft_moe)
