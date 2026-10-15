# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Preflight-only CPU float64 identity and sector checks."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from s2_impl import ARMS, HIDDEN, INTERMEDIATE, LAYER_INDEX, MODEL_REVISION, SoftRouterSwiGLU, initial_kappa, parent_vectors, projectors, q_standard

def check(model_dir: Path) -> dict:
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float64, local_files_only=True).eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    cfg = model.config
    assert cfg.num_hidden_layers == len(model.model.layers) == 30
    assert cfg.hidden_size == HIDDEN and cfg.intermediate_size == INTERMEDIATE
    assert cfg.hidden_act == 'silu' and cfg.num_attention_heads == 9
    assert cfg.num_key_value_heads == 3 and cfg.num_attention_heads > cfg.num_key_value_heads
    assert type(model.model.layers[LAYER_INDEX].mlp).__name__ == 'LlamaMLP'
    assert tuple(model.model.layers[LAYER_INDEX].mlp.gate_proj.weight.shape) == (1536, 576)
    assert tok.vocab_size == cfg.vocab_size == 49152
    dense = model.model.layers[LAYER_INDEX].mlp
    result = {'model_revision': MODEL_REVISION, 'layer_1indexed': LAYER_INDEX + 1, 'layer_index_zero_based': LAYER_INDEX, 'architecture_pass': True, 'tokenizer_class': type(tok).__name__, 'arms': {}}
    seed = 2026110201
    for name, g in ARMS.items():
        moe = SoftRouterSwiGLU(dense, g, parent_vectors(seed, torch.float64)).eval()
        moe.requires_grad_(False)
        m = INTERMEDIATE // g
        expert_copy_error = 0.0
        parent_replication_error = 0.0
        for parent_index in range(4):
            for q in range(g):
                child = moe.experts[parent_index * g + q]
                sl = slice(q * m, (q + 1) * m)
                expert_copy_error = max(expert_copy_error, float((child.gate_proj.weight - dense.gate_proj.weight[sl]).abs().max()), float((child.up_proj.weight - dense.up_proj.weight[sl]).abs().max()), float((child.down_proj.weight - dense.down_proj.weight[:, sl] * g).abs().max()))
                parent_replication_error = max(parent_replication_error, float((moe.router.weight[parent_index * g + q] - parent_vectors(seed, torch.float64)[parent_index]).abs().max()))
        abs_errors, rel_errors, residual_errors, child_prob_errors = ([], [], [], [])
        gen = torch.Generator().manual_seed(1900 + g)
        for batch, sequence, scale in ((1, 1, 0.1), (2, 3, 1.0), (1, 7, 5.0), (4, 2, 0.3)):
            x = torch.randn(batch, sequence, HIDDEN, generator=gen, dtype=torch.float64) * scale
            with torch.no_grad():
                ref = dense(x)
                got = moe(x)
                delta = (got - ref).abs()
                abs_errors.append(float(delta.max()))
                rel_errors.append(float(delta.max() / ref.abs().max().clamp_min(1e-30)))
                norm = model.model.layers[LAYER_INDEX].post_attention_layernorm(x)
                residual_errors.append(float((dense(norm) + x - (moe(norm) + x)).abs().max()))
                p = moe.probabilities(x.reshape(-1, HIDDEN)).reshape(-1, 4, g)
                parent = (x.reshape(-1, HIDDEN) @ parent_vectors(seed, torch.float64).T).softmax(-1)
                child_prob_errors.append(float((p.sum(dim=2) - parent).abs().max()))
                assert torch.allclose(p.sum(dim=(1, 2)), torch.ones(p.shape[0], dtype=p.dtype), atol=1e-14)
        ps = projectors(g)
        eye_errors = {k: float((p @ p - p).abs().max()) for k, p in ps.items()}
        orth_errors = {f'{a}{b}': float((ps[a] @ ps[b]).abs().max()) for a in ps for b in ps if a != b}
        x = torch.randn(13, HIDDEN, generator=gen, dtype=torch.float64) * 1.2
        y = torch.randn(13, HIDDEN, generator=gen, dtype=torch.float64) * 0.7
        q = q_standard(moe, x, y)
        k = initial_kappa(dense, x, y, g)
        q_parent_error = float((q - q @ ps['A']).abs().max())
        k_child_error = float((k - k @ ps['C']).abs().max()) if g > 1 else 0.0
        k_private = k @ ps['C']
        contraction = float((q * k_private).sum(dim=-1).abs().max())
        null_c = float(ps['C'].abs().max()) if g == 1 else None
        null_m = float(ps['M'].abs().max()) if g == 1 else None
        null_router_grad = None
        if g == 1:
            moe.router.weight.requires_grad_(True)
            moe(x).square().mean().backward()
            null_router_grad = float(moe.router.weight.grad.abs().max())
            moe.router.weight.requires_grad_(False)
        entry = {'g': g, 'H': 4 * g, 'm': INTERMEDIATE // g, 'expert_copy_max_abs_error': expert_copy_error, 'parent_router_replication_max_abs_error': parent_replication_error, 'max_abs_error': max(abs_errors), 'max_relative_to_output_scale': max(rel_errors), 'max_residual_mlp_error': max(residual_errors), 'max_parent_probability_error': max(child_prob_errors), 'projector_idempotence_errors': eye_errors, 'projector_orthogonality_errors': orth_errors, 'q_parent_sector_error': q_parent_error, 'k_child_sector_error': k_child_error, 'initial_private_contraction_abs': contraction, 'g1_C_max_abs': null_c, 'g1_M_max_abs': null_m, 'g1_router_gradient_max_abs': null_router_grad}
        assert entry['max_abs_error'] < 1e-10
        assert expert_copy_error == 0 and parent_replication_error == 0
        assert entry['max_residual_mlp_error'] < 1e-10
        assert entry['max_parent_probability_error'] < 1e-14
        assert max(eye_errors.values()) < 1e-14 and max(orth_errors.values()) < 1e-14
        assert q_parent_error < 1e-12 and k_child_error < 1e-12
        assert contraction < 1e-10
        if g == 1:
            assert null_c == 0 and null_m == 0
            assert null_router_grad ** 2 < 1e-20
        result['arms'][name] = entry
    return result
if __name__ == '__main__':
    result = check(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'arms': {k: v['max_abs_error'] for k, v in result['arms'].items()}}))
