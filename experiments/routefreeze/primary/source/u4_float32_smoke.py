# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json
import math
import os
import resource
import time
from pathlib import Path
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeForCausalLM
ROOT = Path(_p1_path('path_f2699b70cfbf'))
assert os.environ.get('HF_HUB_OFFLINE') == '1'
assert os.environ.get('TRANSFORMERS_OFFLINE') == '1'
torch.set_num_threads(8)
start = time.monotonic()
tok = AutoTokenizer.from_pretrained(ROOT, local_files_only=True)
model = OlmoeForCausalLM.from_pretrained(ROOT, local_files_only=True, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager')
model.eval()
assert len(model.model.layers) == 16
assert model.config.num_experts == 64 and model.config.num_experts_per_tok == 8
assert model.config.norm_topk_prob is False
mlp = model.model.layers[15].mlp
assert len(mlp.experts) == 64
for p in model.parameters():
    p.requires_grad_(False)
for p in mlp.experts.parameters():
    p.requires_grad_(True)
ids = tok('The quick brown fox jumps over the lazy dog.', return_tensors='pt', add_special_tokens=False)['input_ids']
assert 2 <= ids.shape[1] <= 16
records = []
for _ in range(2):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, use_cache=False)
    loss = F.cross_entropy(out.logits[:, :-1, :].contiguous().view(-1, out.logits.shape[-1]), ids[:, 1:].contiguous().view(-1))
    assert math.isfinite(float(loss))
    loss.backward()
    norms = [float(torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in e.parameters() if p.grad is not None]))) for e in mlp.experts if any((p.grad is not None for p in e.parameters()))]
    assert norms and all((math.isfinite(n) for n in norms))
    assert all((p.grad is None for p in mlp.gate.parameters()))
    assert all((p.grad is None for p in model.model.layers[0].parameters()))
    assert all((p.grad is None for p in model.lm_head.parameters()))
    records.append({'loss': float(loss), 'gradient_norms': norms})
    print(json.dumps({'run': len(records), 'loss': float(loss), 'active_experts': len(norms), 'rss_bytes': psutil.Process().memory_info().rss, 'elapsed_seconds': time.monotonic() - start}), flush=True)
assert math.isclose(records[0]['loss'], records[1]['loss'], rel_tol=1e-07)
assert len(records[0]['gradient_norms']) == len(records[1]['gradient_norms'])
assert all((math.isclose(a, b, rel_tol=1e-07, abs_tol=1e-12) for a, b in zip(records[0]['gradient_norms'], records[1]['gradient_norms'])))
print(json.dumps({'status': 'PASS', 'dtype': 'float32', 'tokens': ids.shape[1], 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'elapsed_seconds': time.monotonic() - start}), flush=True)
