# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen Q3 adapter: two FP32 scalars added to native pre-Top-k router logits."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gc, types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import *

def seed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(6)

def load_model():
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS_UNAVAILABLE')
    model = AutoModelForCausalLM.from_pretrained(str(SNAPSHOT), local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='eager', trust_remote_code=False).to('mps').eval()
    model.requires_grad_(False)
    if any((p.dtype != torch.bfloat16 or p.device.type != 'mps' for p in model.parameters())):
        raise RuntimeError('BASE_DTYPE_DEVICE_MISMATCH')
    return model

def directions(arm):
    out = {}
    for layer in (7, 15):
        with np.load(BUNDLE / 'controls' / f'frozen_null_controls_layer_{layer}_v164.npz', allow_pickle=False) as z:
            if arm == 'BASELINE':
                vector = np.zeros(64, dtype=np.float64)
            elif arm == 'CANDIDATE':
                vector = z['candidate'][64:]
            else:
                vector = z['null_controls'][int(arm[-3:]), 64:]
        out[layer] = torch.tensor(vector, dtype=torch.float32, device='mps')
    return out

class Adapter:

    def __init__(self, model, arm):
        self.alpha = {l: torch.nn.Parameter(torch.zeros((), device='mps', dtype=torch.float32)) for l in (7, 15)}
        self.bias = directions(arm)
        self.handles = []
        self.semantics = []
        for l in (7, 15):

            def hook(module, args, output, layer=l):
                modified = output.float() + self.alpha[layer] * self.bias[layer]
                return modified
            self.handles.append(model.model.layers[l].mlp.gate.register_forward_hook(hook))

    def optimizer(self):
        return torch.optim.Adam(list(self.alpha.values()), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def project(self):
        with torch.no_grad():
            for alpha in self.alpha.values():
                alpha.clamp_(-0.02, 0.02)

    def values(self):
        return {f'alpha{l}': float(p.detach().cpu()) for l, p in self.alpha.items()}

    def close(self):
        for h in self.handles:
            h.remove()

def nll(model, ids):
    x = torch.tensor(ids, dtype=torch.long, device='mps').unsqueeze(0)
    output = model(input_ids=x, use_cache=False, output_router_logits=False)
    loss = F.cross_entropy(output.logits[:, :-1, :].float().reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1), reduction='mean')
    if not bool(torch.isfinite(loss)):
        raise RuntimeError('NONFINITE_NLL')
    return loss

def ordered(rows):
    bydomain = {d: sorted([r for r in rows if r['domain'] == d], key=lambda r: r['token_array_sha256']) for d in DOMAINS}
    return [bydomain[d][i] for i in range(len(bydomain[DOMAINS[0]])) for d in DOMAINS]

def base_audit(model, before):
    return dict(base_parameter_updates=sum((p._version != before[name] for name, p in model.named_parameters())), base_parameters_with_grad=sum((p.grad is not None for p in model.parameters())), base_trainable_parameters=sum((p.requires_grad for p in model.parameters())))

def unload(model):
    del model
    gc.collect()
    torch.mps.empty_cache()
