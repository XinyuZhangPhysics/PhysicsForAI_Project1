# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gc
import os
import torch
from . import MODEL_REPO, LAYERS, StopPilot
INVARIANTS = dict(num_hidden_layers=16, num_experts=64, num_experts_per_tok=8, hidden_size=2048)

def config_guard(config):
    values = config if isinstance(config, dict) else vars(config)
    differences = {k: (values.get(k), v) for k, v in INVARIANTS.items() if values.get(k) != v}
    if differences:
        raise StopPilot(f'Frozen architecture mismatch: {differences}')
    if values.get('model_type') != 'olmoe':
        raise StopPilot('Expected OLMoE model_type.')

def device_guard(device):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise StopPilot('PYTORCH_ENABLE_MPS_FALLBACK=1 is forbidden.')
    if device != 'mps':
        raise StopPilot('Production pilot requires MPS. CPU is for explicit synthetic/metadata checks only.')
    if not torch.backends.mps.is_available():
        raise StopPilot('MPS unavailable; CPU model fallback is forbidden.')
    x = torch.ones((2, 2), dtype=torch.bfloat16, device=device)
    if (x @ x).device.type != 'mps':
        raise StopPilot('MPS operation did not remain on MPS.')

def load_checkpoint(entry, cache_dir, device='mps'):
    from transformers import AutoConfig, AutoModelForCausalLM
    device_guard(device)
    config = AutoConfig.from_pretrained(MODEL_REPO, revision=entry['revision_commit_sha'], cache_dir=cache_dir, trust_remote_code=False)
    config_guard(config)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, revision=entry['revision_commit_sha'], config=config, torch_dtype=torch.bfloat16, cache_dir=cache_dir, attn_implementation='eager', trust_remote_code=False).to(device).eval()
    model.requires_grad_(False)
    if any((p.device.type != device or p.dtype != torch.bfloat16 for p in model.parameters())):
        raise StopPilot('Weights are not uniformly BF16 on MPS.')
    for idx in LAYERS:
        mlp = model.model.layers[idx].mlp
        if not hasattr(mlp, 'experts') or len(mlp.experts) != 64:
            raise StopPilot('Unsupported expert layout; pinned Transformers adapter requires ModuleList.')
    return model

def capture_layers(model, ids):
    captured, handles = ({i: {} for i in LAYERS}, [])

    def hook_for(index):

        def hook(module, args, output):
            captured[index].update(mlp_input=args[0].detach(), mlp_output=output[0].detach())
        return hook

    def residual_hook(index):

        def hook(module, args):
            captured[index]['residual'] = args[0].detach()
        return hook

    def layer_hook(index):

        def hook(module, args, output):
            captured[index]['layer_output'] = output[0].detach()
        return hook
    try:
        for index in LAYERS:
            layer = model.model.layers[index]
            handles.append(layer.mlp.register_forward_hook(hook_for(index)))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(residual_hook(index)))
            handles.append(layer.register_forward_hook(layer_hook(index)))
        with torch.inference_mode():
            result = model.model(input_ids=ids, use_cache=False, output_router_logits=False)
        del result
    finally:
        for handle in handles:
            handle.remove()
    if any((set(captured[i]) != {'mlp_input', 'mlp_output', 'residual', 'layer_output'} for i in LAYERS)):
        raise StopPilot('Frozen layer hooks did not fire exactly as required.')
    return captured

def unload():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
