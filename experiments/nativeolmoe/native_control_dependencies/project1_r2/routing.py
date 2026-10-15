# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Routes:
    indices: torch.Tensor
    weights: torch.Tensor
    probabilities: torch.Tensor
    logits: torch.Tensor
    margin: torch.Tensor

def route(logits, top_k, norm_topk_prob=False):
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, indices = torch.topk(probabilities, top_k, dim=-1)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    ordered = torch.topk(logits.float(), top_k + 1, dim=-1).values
    return Routes(indices.detach().clone(), weights.detach().clone(), probabilities.detach(), logits.detach(), (ordered[:, top_k - 1] - ordered[:, top_k]).detach())

def sufficient_statistics(routes, n_experts, margin_delta):
    support = torch.zeros_like(routes.probabilities)
    weighted = torch.zeros_like(routes.probabilities)
    support.scatter_(1, routes.indices, 1.0)
    weighted.scatter_(1, routes.indices, routes.weights)
    return dict(logit_mean=routes.logits.float().mean(0).cpu().numpy(), probability_mean=routes.probabilities.mean(0).cpu().numpy(), support_mean=support.mean(0).cpu().numpy(), selected_weight_mean=weighted.mean(0).cpu().numpy(), margin_mean=routes.margin.mean().item(), small_margin_mass=(routes.margin < margin_delta).float().mean().item())

def route_comparison(first, second):
    return dict(route_index_mismatch=int((first.indices != second.indices).sum().item()), cached_weight_changes=int((first.weights != second.weights).sum().item()))
