# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""OLMoE final-layer instrumentation, functional MoE, expert tangent kernels.

Everything here is reimplemented from the mathematical specification:

  m_{l r}   = sum_{i in A(r)} a_{l r i} e_{l i}(z_{l r}; theta_{l i})
  g_{i,d}   = (1/N_d) sum_r a_{i}(r) J_i(z_r)^T d_r
  K_i(z,z') = J_i(z) M_i J_i(z')^T ,   M_i = I   (Euclidean parameter metric)

with, for OLMoE,  e_i(z) = W^d_i [ silu(W^g_i z) * (W^u_i z) ]  (SwiGLU MLP, bias=False)
and router weights = softmax over all 64 logits, restricted to the top-8 entries with
**no** renormalisation (`norm_topk_prob = false`), per preregistration Sections 3 and 8.

NOTE ON INDEPENDENCE: the *native* `transformers.Olmoe*` modules are used as the model
(they are the object under study, i.e. public model code, not Codex replication code).
The response algebra, estimator, probe and estimator bootstrap are reimplemented here.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from u4_common import PATHS, log, sha256_bytes
LAYER = 15
N_EXPERTS = 64
TOP_K = 8
D_MODEL = 2048
D_FF = 1024
N_TOK = 128
N_PARAMS_PER_EXPERT = 3 * D_FF * D_MODEL

def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    md = str(PATHS['model_dir'])
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(md, dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    return (model, tok)

def manual_next_token_ce(logits, ids):
    """L = mean_r CE(f_r, x_{r+1}), r = 0..126   (preregistration Section 7)."""
    import torch
    lg = logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    logp = torch.log_softmax(lg, dim=-1)
    return -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()

def direct_next_token_ce(logits, ids):
    """Independent equivalent CE (used only as a self-consistency control)."""
    import torch
    import torch.nn.functional as F
    lg = logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))

class FinalLayerMoE:
    """Adapter around `model.model.layers[15]`."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer
        self.layer = model.model.layers[15]
        self.mlp = self.layer.mlp
        self.gate_w = self.mlp.gate.weight.detach()
        self.ew = [[e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight] for e in self.mlp.experts]
        for e in self.mlp.experts:
            assert not hasattr(e, 'bias') or e.gate_proj.bias is None
        assert len(self.ew) == N_EXPERTS
        assert self.gate_w.shape == (N_EXPERTS, D_MODEL)
        assert self.ew[0][0].shape == (D_FF, D_MODEL)
        assert self.ew[0][2].shape == (D_MODEL, D_FF)

    def param_tensors(self, j: int) -> List:
        return self.ew[j]

    def all_params(self) -> List:
        return [p for j in range(N_EXPERTS) for p in self.ew[j]]

    def set_grad_mode(self, expert_only: bool=True) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.all_params():
            p.requires_grad_(True)

    def params_digest(self) -> str:
        import torch
        h = []
        for j in range(N_EXPERTS):
            for p in self.ew[j]:
                h.append(sha256_bytes(p.detach().contiguous().numpy().tobytes()))
        return sha256_bytes(''.join(h).encode())

    def route(self, z):
        """Actual OLMoE routing: softmax over 64 logits -> top-8, no renorm."""
        import torch
        logits = z @ self.gate_w.t()
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, TOP_K, dim=-1)
        return (logits, probs, w, idx)

    @staticmethod
    def expert_out(z, Wg, Wu, Wd):
        import torch
        import torch.nn.functional as F
        return F.silu(z @ Wg.t()) * (z @ Wu.t()) @ Wd.t()

    def moe(self, z, w, idx, deltas: Optional[Sequence]=None):
        """m_r = sum_{i in A(r)} a_{ri} e_i(z_r ; theta_i + delta_i)."""
        import torch
        out = torch.zeros_like(z)
        for j in range(N_EXPERTS):
            rows, slot = (idx == j).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            Wg, Wu, Wd = self.ew[j]
            if deltas is not None:
                Wg, Wu, Wd = (Wg + deltas[j][0], Wu + deltas[j][1], Wd + deltas[j][2])
            oj = self.expert_out(z[rows], Wg, Wu, Wd) * w[rows, slot].unsqueeze(-1)
            out.index_add_(0, rows, oj)
        return out

    def head(self, h_out):
        """Final layernorm + LM head (all downstream dense propagation)."""
        return self.model.lm_head(self.model.model.norm(h_out))

    def expert_grad_scaled(self, j: int, z, dadj):
        """grad_{theta_j} sum_r <a_r d_r, e_j(z_r)>, flattened in param order.

        z, dadj: (n, 2048) with dadj already multiplied by the scalar routing coefficient.
        """
        import torch
        ps = self.ew[j]
        zz = z.detach()
        dd = dadj.detach()
        with torch.enable_grad():
            out = self.expert_out(zz, ps[0], ps[1], ps[2])
            s = (out * dd).sum()
            gs = torch.autograd.grad(s, ps)
        return torch.cat([g.reshape(-1) for g in gs]).detach()

    def forward_capture(self, ids, grad: bool=False) -> Dict:
        """One forward of the full model with layer-15 hooks.

        Returns h_in (layer-15 block input), h_att (post-attention residual),
        z (MoE input), m_native, native lm logits and the native layer-15 routing.
        """
        import torch
        cap: Dict[str, object] = {}
        handles = []

        def _flat(t):
            return t.detach().reshape(-1, D_MODEL).contiguous()

        def mk_input(name):

            def hook(_m, inp, _out):
                cap[name] = _flat(inp[0])
            return hook

        def mk_output(name, flat=True):

            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cap[name] = _flat(t) if flat else t.detach()
            return hook
        handles.append(self.layer.input_layernorm.register_forward_hook(mk_input('h_in')))
        handles.append(self.layer.post_attention_layernorm.register_forward_hook(mk_input('h_att')))
        handles.append(self.layer.post_attention_layernorm.register_forward_hook(mk_output('z')))
        handles.append(self.mlp.gate.register_forward_hook(mk_output('router_logits', flat=False)))
        handles.append(self.mlp.register_forward_hook(mk_output('m_native')))
        try:
            if grad:
                out = self.model(ids)
            else:
                with torch.no_grad():
                    out = self.model(ids)
        finally:
            for h in handles:
                h.remove()
        cap['lm_logits'] = out.logits
        cap['ids'] = ids
        cap['loss'] = manual_next_token_ce(out.logits, ids)
        rl = cap['router_logits'].reshape(-1, N_EXPERTS)
        probs = torch.softmax(rl.float(), dim=-1)
        w, idx = torch.topk(probs, TOP_K, dim=-1)
        cap['router_logits'] = rl
        cap['router_probs'] = probs
        cap['route_w'], cap['route_idx'] = (w, idx)
        return cap

    def full_forward_with_expert_delta(self, ids, deltas) -> Dict:
        """Mutate layer-15 experts in place, run the **whole model**, restore exactly.

        Used only for the routing-freeze audit (preregistration Section 13): it is a real
        end-to-end forward on the perturbed model, not a shortcut.
        """
        import torch
        backups = [p.detach().clone() for p in self.all_params()]
        try:
            for j in range(N_EXPERTS):
                for k in range(3):
                    self.ew[j][k].data.add_(deltas[j][k])
            cap = self.forward_capture(ids, grad=False)
        finally:
            for j in range(N_EXPERTS):
                for k in range(3):
                    self.ew[j][k].data.copy_(backups[j * 3 + k])
        restored = all((bool(torch.equal(self.ew[j][k], backups[j * 3 + k])) for j in range(N_EXPERTS) for k in range(3)))
        cap['restored_exactly'] = restored
        cap['params_digest_after'] = self.params_digest()
        return cap
