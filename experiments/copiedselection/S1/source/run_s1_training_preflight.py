# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import resource
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).parent))
from s1_impl import upcycle
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
MODEL = ROOT / 'work/model'

def tensor_sha(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()

def make_model(seed):
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True).to('mps')
    moe = upcycle(model, 8, seed)
    model.train()
    return (model, moe)

def run_step(model, optimizer, ids, mask):
    labels = ids.clone()
    labels[mask == 0] = -100
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    assert bool(torch.isfinite(loss).item())
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 1 and params[0].grad is not None
    assert bool(torch.isfinite(params[0].grad).all().item())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    return float(loss.detach().cpu())

def smoke():
    model, moe = make_model(2026100101)
    before = {name: tensor_sha(p) for name, p in model.named_parameters()}
    buffers_before = {name: tensor_sha(b) for name, b in model.named_buffers()}
    data = torch.load(ROOT / 'work/s1_calibration_gate_256.pt', weights_only=True, map_location='cpu')
    ids = data['input_ids'][:1].to('mps')
    mask = data['attention_mask'][:1].to('mps')
    optimizer = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    loss = run_step(model, optimizer, ids, mask)
    torch.mps.synchronize()
    after = {name: tensor_sha(p) for name, p in model.named_parameters()}
    buffers_after = {name: tensor_sha(b) for name, b in model.named_buffers()}
    changed = [name for name in before if before[name] != after[name]]
    result = {'status': 'PASS' if changed == ['gpt_neox.layers.3.mlp.router.weight'] and buffers_before == buffers_after else 'FAIL', 'loss': loss, 'finite_loss_and_gradient': True, 'changed_tensors': changed, 'unchanged_tensors': len(before) - len(changed), 'experts_unchanged': all((before[n] == after[n] for n in before if '.experts.' in n)), 'other_model_tensors_unchanged': all((before[n] == after[n] for n in before if '.experts.' not in n and '.router.' not in n)), 'model_buffers_unchanged': buffers_before == buffers_after, 'model_buffers_checked': len(buffers_before), 'disposable_copy_discarded': True}
    (OUT / 'IMMUTABILITY_GATE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)

def benchmark():
    seed = 2026100198
    model, moe = make_model(seed)
    data = torch.load(ROOT / 'work/s1_benchmark_100k.pt', weights_only=True, map_location='cpu')
    assert int(data['attention_mask'].sum()) == 100000
    assert data['input_ids'].shape[0] == 392
    optimizer = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mps = torch.mps.current_allocated_memory()
    count = 0
    losses = []
    start = time.perf_counter()
    steps = data['input_ids'].shape[0]
    for index in range(steps):
        u = (index + 1) / steps
        lr = 0.001 * (u / 0.05 if u <= 0.05 else 0.1 + 0.9 * (1 + math.cos(math.pi * (u - 0.05) / 0.95)) / 2)
        optimizer.param_groups[0]['lr'] = lr
        ids = data['input_ids'][index:index + 1].to('mps')
        mask = data['attention_mask'][index:index + 1].to('mps')
        losses.append(run_step(model, optimizer, ids, mask))
        count += int(mask.sum().cpu())
        if (index + 1) % 20 == 0 or index + 1 == steps:
            torch.mps.synchronize()
            peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            peak_mps = max(peak_mps, torch.mps.current_allocated_memory())
            print('step', index + 1, 'tokens', count, 'seconds', round(time.perf_counter() - start, 2), flush=True)
    wall = time.perf_counter() - start
    assert count == 100000
    projected_hours = wall * 8192000 / 100000 / 3600
    result = {'status': 'PASS' if projected_hours <= 6 else 'EXP_S1_BLOCKED_LOCAL_RUNTIME_BUDGET', 'arm': 'S8', 'seed': seed, 'nonpadding_tokens': count, 'sequences': steps, 'wall_seconds': wall, 'tokens_per_second': count / wall, 'peak_mps_allocated_bytes': peak_mps, 'peak_process_rss_bytes': peak_rss, 'projected_32_run_256k_hours': projected_hours, 'budget_hours': 6, 'finite_losses': all((math.isfinite(x) for x in losses)), 'first_loss': losses[0], 'last_loss': losses[-1], 'disposable_copy_discarded': True}
    (OUT / 'RUNTIME_BENCHMARK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    if sys.argv[1] == 'smoke':
        smoke()
    elif sys.argv[1] == 'benchmark':
        benchmark()
    else:
        raise SystemExit('smoke or benchmark')
