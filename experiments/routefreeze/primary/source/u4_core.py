# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen U4 model conventions and exact expert-gradient utilities."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math
import torch
import torch.nn.functional as F

def actual_routes(mlp, hidden):
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = mlp.gate(flat)
    probs = F.softmax(logits, dim=1, dtype=torch.float32)
    weights, labels = torch.topk(probs, mlp.top_k, dim=-1)
    assert not mlp.norm_topk_prob
    return (logits, labels, weights.to(flat.dtype))

def expert_output(mlp, hidden, labels, weights, permutation=None):
    """Native OLMoE operation with optional frozen permutation of route labels."""
    shape = hidden.shape
    flat = hidden.reshape(-1, shape[-1])
    if permutation is not None:
        labels = permutation[labels]
    mask = F.one_hot(labels, num_classes=mlp.num_experts).permute(2, 1, 0)
    result = torch.zeros_like(flat)
    for j, expert in enumerate(mlp.experts):
        rank, token = torch.where(mask[j])
        x = flat[None, token].reshape(-1, shape[-1])
        y = expert(x) * weights[token, rank, None]
        result.index_add_(0, token, y.to(flat.dtype))
    return result.reshape(shape)

def param_layout(mlp):
    params = list(mlp.experts.parameters())
    offsets = []
    cursor = 0
    for p in params:
        offsets.append((cursor, cursor + p.numel()))
        cursor += p.numel()
    return (params, offsets, cursor)

def copy_grad_flat(params, offsets, target, factor=1.0):
    for p, (a, b) in zip(params, offsets):
        if p.grad is not None:
            target[a:b].add_(p.grad.reshape(-1), alpha=factor)

def set_experts_flat(params, offsets, baseline, direction=None, scale=0.0):
    with torch.no_grad():
        for p, (a, b) in zip(params, offsets):
            p.copy_(baseline[a:b].view_as(p))
            if direction is not None and scale:
                p.add_(direction[a:b].view_as(p), alpha=scale)

def flatten_params(params, offsets, total):
    flat = torch.empty(total, dtype=torch.float32)
    for p, (a, b) in zip(params, offsets):
        flat[a:b].copy_(p.detach().reshape(-1))
    return flat

def dot_blockwise(a, b, offsets):
    """Sum parameter-tensor dots in float64 to suppress reduction-order drift."""
    return math.fsum((float(torch.dot(a[x:y], b[x:y])) for x, y in offsets))

def manual_ce(logits, token_ids):
    return F.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]), token_ids[..., 1:].contiguous().view(-1))
