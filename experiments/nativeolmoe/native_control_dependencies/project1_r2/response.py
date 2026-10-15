# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import numpy as np
import torch
from . import StopPilot
CONTROL_NAMES = tuple([f'expert_gain_{i}' for i in range(64)] + [f'router_bias_{i}' for i in range(64)])

def private_control(control):
    result = np.asarray(control, dtype=float).copy()
    if result.shape != (128,):
        raise ValueError('Exactly 64 gains then 64 router biases are required.')
    result[:64] -= result[:64].mean()
    result[64:] -= result[64:].mean()
    return result

def symmetric_difference(function, direction, step):
    if step <= 0:
        raise ValueError('Finite difference step must be positive.')
    return (function(step * direction) - function(-step * direction)) / (2 * step)

def half_step_check(function, direction, step):
    full = symmetric_difference(function, direction, step)
    half = symmetric_difference(function, direction, step / 2)
    discrepancy = (half - full).norm() / half.norm().clamp_min(1e-30)
    return (half, float(discrepancy.item()))

def check_native_response(local, direction, step):
    direction = torch.as_tensor(direction, dtype=torch.float32, device=local.hidden.device)

    def evaluate(control):
        out, _ = local.native_residual_output(control[:local.n_experts], control[local.n_experts:], frozen=False)
        return out.float()
    finite, change = half_step_check(evaluate, direction, step)
    analytic = local.differential(direction)
    agreement = (finite - analytic).norm() / analytic.norm().clamp_min(1e-30)
    plus = local.native_output(biases=step * direction[local.n_experts:], frozen=False)[1]
    minus = local.native_output(biases=-step * direction[local.n_experts:], frozen=False)[1]
    switches = float(((plus.indices != local.routes.indices).any(-1) | (minus.indices != local.routes.indices).any(-1)).float().mean().item())
    return dict(half_step_relative_change=change, analytic_fd_relative_error=agreement.item(), support_switch_fraction=switches, response_norm=analytic.norm().item(), linearity_ok=change < 0.05 and agreement.item() < 0.05)

def candidate_controls(router, expert, rank, seed):
    """Block normalization for engineering only; no bank-A joint whitening claim."""
    router_norm, expert_norm = (np.linalg.norm(router), np.linalg.norm(expert))
    if router_norm == 0 or expert_norm == 0:
        raise StopPilot('Internal block amplitude is zero; candidate undefined.')
    y = np.vstack([expert / expert_norm, router / router_norm])
    u, singular, _ = np.linalg.svd(y, full_matrices=False)
    if singular[rank - 1] <= 1e-12:
        raise StopPilot('Frozen engineering candidate rank exceeds measured internal rank.')
    controls = {}
    for j in range(rank):
        v = private_control(u[:, j])
        v /= np.linalg.norm(v)
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
        controls[f'candidate_{j}'] = v
        controls[f'gain_only_{j}'] = np.r_[v[:64], np.zeros(64)]
        controls[f'router_only_{j}'] = np.r_[np.zeros(64), v[64:]]
        rng = np.random.default_rng(seed + j)
        null = private_control(rng.normal(size=128))
        null[:64] *= np.linalg.norm(v[:64]) / np.linalg.norm(null[:64])
        null[64:] *= np.linalg.norm(v[64:]) / np.linalg.norm(null[64:])
        controls[f'random_{j}'] = null
        permutation = rng.permutation(64)
        controls[f'expert_permutation_{j}'] = np.r_[v[:64][permutation], v[64:][permutation]]
    return controls
