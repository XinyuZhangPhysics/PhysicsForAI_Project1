# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Preflight-only CPU/MPS semantic and frozen-parameter smoke checks."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import copy
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from s2_impl import LAYER_INDEX, SoftRouterSwiGLU, assert_router_optimizer_only, buffer_digests, frozen_digests, initial_kappa, parent_vectors, projectors, q_standard, tensor_digest, upcycle

def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().cpu().double()
    b = b.detach().cpu().double()
    delta = (a - b).abs()
    return {'max_abs': float(delta.max()), 'relative_to_reference_scale': float(delta.max() / a.abs().max().clamp_min(1e-30))}

def run(model_dir: Path) -> dict:
    assert torch.backends.mps.is_available(), 'MPS_UNAVAILABLE'
    torch.set_num_threads(8)
    seed = 2026110201
    gen = torch.Generator().manual_seed(917)
    x64 = torch.randn(11, 576, generator=gen, dtype=torch.float64) * 1.1
    y64 = torch.randn(11, 576, generator=gen, dtype=torch.float64) * 0.8
    cpu_model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float64, local_files_only=True).eval()
    cpu_dense = cpu_model.model.layers[LAYER_INDEX].mlp
    cpu_moe = SoftRouterSwiGLU(cpu_dense, 8, parent_vectors(seed, torch.float64))
    with torch.no_grad():
        cpu_original = cpu_dense(x64)
        cpu_upcycled = cpu_moe(x64)
        cpu_probs = cpu_moe.probabilities(x64)
        cpu_q = q_standard(cpu_moe, x64, y64)
        cpu_k = initial_kappa(cpu_dense, x64, y64, 8)
    cpu_loss = cpu_moe(x64).square().mean()
    cpu_loss.backward()
    cpu_grad = cpu_moe.router.weight.grad.detach().clone()
    del cpu_model
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32, local_files_only=True).eval()
    dense = copy.deepcopy(model.model.layers[LAYER_INDEX].mlp)
    moe = upcycle(model, 8, seed)
    model.to('mps')
    dense.to('mps')
    x = x64.float().to('mps')
    y = y64.float().to('mps')
    with torch.no_grad():
        mps_original = dense(x)
        mps_upcycled = moe(x)
        mps_probs = moe.probabilities(x)
        mps_q = q_standard(moe, x, y)
        mps_k = initial_kappa(dense, x, y, 8)
        c8 = projectors(8, torch.float32)['C'].to('mps')
        mps_private_contraction = (mps_q * (mps_k @ c8)).sum(-1)
    torch.mps.synchronize()
    assert all((t.device.type == 'mps' for t in (mps_original, mps_upcycled, mps_probs, mps_q, mps_k, mps_private_contraction)))
    assert all((bool(torch.isfinite(t).all()) for t in (mps_original, mps_upcycled, mps_probs, mps_q, mps_k, mps_private_contraction)))
    result = {'device': 'mps', 'dtype': 'torch.float32', 'reference_dtype': 'torch.float64', 'original_forward_cpu_mps': metrics(cpu_original, mps_original), 'upcycled_forward_cpu_mps': metrics(cpu_upcycled, mps_upcycled), 'mps_original_upcycled': metrics(mps_original, mps_upcycled), 'router_probabilities_cpu_mps': metrics(cpu_probs, mps_probs), 'q_standard_cpu_mps': metrics(cpu_q, mps_q), 'kappa_cpu_mps': metrics(cpu_k, mps_k), 'initial_private_contraction_max_abs_mps': float(mps_private_contraction.abs().max().cpu()), 'mps_prob_sum_max_error': float((mps_probs.sum(-1) - 1).abs().max().cpu()), 'all_mps_finite': True}
    px = mps_probs.detach().cpu().double()
    py = moe.probabilities(y).detach().cpu().double()
    raw = px * py
    centered = raw - raw.mean(dim=-1, keepdim=True)
    pq = (centered.square().sum(dim=-1) / 31).mean()
    q64 = centered / pq.sqrt()
    c64 = projectors(8)['C']
    q_child64 = q64 @ c64
    k_child64 = mps_k.detach().cpu().double() @ c64
    result['s1_order_cpu64_projected_q_child_max_abs'] = float(q_child64.abs().max())
    result['s1_order_cpu64_initial_private_contraction_max_abs'] = float((q_child64 * k_child64).sum(-1).abs().max())
    p1 = projectors(1, torch.float32)
    result['g1_C_M_exact_zero'] = bool(torch.count_nonzero(p1['C']) == 0 and torch.count_nonzero(p1['M']) == 0)
    result['g1_structural_B_power'] = 0.0
    before_frozen = frozen_digests(model)
    before_buffers = buffer_digests(model)
    before_router = tensor_digest(moe.router.weight)
    opt = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    assert_router_optimizer_only(model, opt)
    grads = []
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = moe(x).square().mean()
        loss.backward()
        grads.append(moe.router.weight.grad.detach().clone())
        assert all((p.grad is None for p in model.parameters() if not p.requires_grad))
        torch.nn.utils.clip_grad_norm_([moe.router.weight], 1.0)
        opt.step()
    torch.mps.synchronize()
    assert_router_optimizer_only(model, opt)
    after_frozen = frozen_digests(model)
    after_buffers = buffer_digests(model)
    after_router = tensor_digest(moe.router.weight)
    result['router_grad_cpu_mps'] = metrics(cpu_grad, grads[0])
    result['immutability'] = {'frozen_parameter_count': len(before_frozen), 'frozen_parameters_hash_identical': before_frozen == after_frozen, 'buffer_count': len(before_buffers), 'buffers_hash_identical': before_buffers == after_buffers, 'router_changed': before_router != after_router, 'frozen_gradients_absent': all((p.grad is None for p in model.parameters() if not p.requires_grad)), 'optimizer_router_only': True}
    assert all(result['immutability'].values())
    assert result['g1_C_M_exact_zero']
    assert result['mps_original_upcycled']['relative_to_reference_scale'] < 1e-05
    assert result['mps_prob_sum_max_error'] < 1e-06
    result['peak_mps_driver_bytes'] = int(torch.mps.driver_allocated_memory())
    return result
if __name__ == '__main__':
    d = run(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(d, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'mps_original_upcycled': d['mps_original_upcycled'], 'immutability': d['immutability']}))
