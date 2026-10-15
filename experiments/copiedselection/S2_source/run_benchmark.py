# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""100k-token S8 end-to-end MPS benchmark on disjoint preflight-only data.

This command has no scientific-bank or scientific-audit input. It retains only
timing, infrastructure, and finite/numerical-sanity diagnostics.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
from build_banks import filehash
from s2_impl import LAYER_INDEX, assert_router_optimizer_only, upcycle

def synced() -> float:
    torch.mps.synchronize()
    return time.perf_counter()

def s1_lr_factor(tokens: int) -> float:
    """Exact S1 5% warmup, cosine decay to 0.1 peak over 256k tokens."""
    if tokens <= 12800:
        return tokens / 12800
    progress = (tokens - 12800) / (256000 - 12800)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

def response_pass(model, moe, bank: dict, seed: int) -> dict:
    model.eval()
    start = synced()
    captured = []

    def capture(_module, args):
        captured.append(args[0].detach())
    hook = moe.register_forward_pre_hook(capture)
    reps = []
    with torch.inference_mode():
        for i in range(0, 2048, 8):
            ids = bank['input_ids'][i:i + 8].to('mps')
            masks = bank['attention_mask'][i:i + 8].to('mps')
            captured.clear()
            model.model(input_ids=ids, attention_mask=masks, use_cache=False)
            h = captured[0]
            last = masks.sum(1).long() - 1
            reps.append(h[torch.arange(len(h), device='mps'), last].cpu())
    hook.remove()
    h = torch.cat(reps, dim=0).contiguous()
    assert tuple(h.shape) == (2048, 576) and bool(torch.isfinite(h).all())
    representation_seconds = synced() - start
    rng = np.random.default_rng(seed)
    src, tgt, groups = ([], [], [])
    for a in range(4):
        for b in range(4):
            if a == b:
                continue
            group = len(set(groups)) if False else a * 4 + b
            for pos in rng.choice(512 * 512, size=1024, replace=False):
                src.append(a * 512 + int(pos) // 512)
                tgt.append(b * 512 + int(pos) % 512)
                groups.append(group)
    src = np.asarray(src, dtype=np.int32)
    tgt = np.asarray(tgt, dtype=np.int32)
    groups = np.asarray(groups, dtype=np.int8)
    assert len(src) == 12288 and len(np.unique(groups)) == 12
    start = synced()
    with torch.inference_mode():
        e = moe.experts[0]
        acts = []
        hm = h.to('mps')
        for q in range(8):
            child = moe.experts[q]
            acts.append((moe.act_fn(child.gate_proj(hm)) * child.up_proj(hm)).cpu())
        features = torch.cat(acts, dim=1).double().numpy()
        probs = moe.probabilities(hm).cpu().double().numpy()
    k_child = np.empty((len(src), 8), dtype=np.float64)
    for q in range(8):
        sl = slice(q * 192, (q + 1) * 192)
        k_child[:, q] = (features[src, sl] * features[tgt, sl]).mean(axis=1)
    k_centered = k_child - k_child.mean(axis=1, keepdims=True)
    k_c = np.broadcast_to(k_centered[:, None, :], (len(src), 4, 8)).reshape(len(src), 32)
    geometry_seconds = synced() - start
    start = synced()
    raw = probs[src] * probs[tgt]
    centered = raw - raw.mean(axis=1, keepdims=True)
    pq = np.square(centered).sum(axis=1).mean() / 31
    assert np.isfinite(pq) and pq > 0
    qhat = centered / math.sqrt(pq)
    q3 = qhat.reshape(len(src), 4, 8)
    q_c = np.broadcast_to(q3.mean(axis=1, keepdims=True) - q3.mean(axis=(1, 2), keepdims=True), q3.shape).reshape(len(src), 32)
    projected = np.sum(q_c * k_c, axis=1)
    direct = np.sum(qhat * k_c, axis=1)
    projector_error = float(np.max(np.abs(projected - direct)))
    finite = bool(np.isfinite(projected).all() and np.isfinite(q_c).all() and np.isfinite(k_c).all())
    aggregates = [projected[groups == g].mean() for g in np.unique(groups)]
    finite = finite and bool(np.isfinite(aggregates).all())
    del aggregates, projected, direct, q_c, qhat, k_c, k_child
    response_seconds = synced() - start
    assert finite and projector_error < 1e-08
    return {'documents': 2048, 'ordered_pair_coordinates': 12, 'pairs_per_coordinate': 1024, 'representation_seconds': representation_seconds, 'geometry_seconds': geometry_seconds, 'router_response_seconds': response_seconds, 'total_seconds': representation_seconds + geometry_seconds + response_seconds, 'all_finite': finite, 'response_projector_identity_max_abs_error': projector_error, 'scientific_slopes_or_powers_retained': False}

def run(model_dir: Path, bank_dir: Path, out: Path) -> dict:
    assert torch.backends.mps.is_available(), 'MPS_UNAVAILABLE'
    torch.set_num_threads(8)
    index = json.loads((bank_dir / 'BANKS_INDEX.json').read_text())
    assert index['status'] == 'BANK_IDS_FROZEN_PRE_SCIENCE'
    assert index['checks']['all_partitions_pairwise_disjoint']
    assert index['preflight_benchmark_tokens'] == 100000
    tensor_path = bank_dir / 'preflight_benchmark_100k.pt'
    assert filehash(tensor_path) == index['manifest_sha256']['preflight_benchmark_tensor']
    start_total = time.perf_counter()
    start = time.perf_counter()
    data = torch.load(tensor_path, map_location='cpu', weights_only=True)
    ids, masks = (data['input_ids'], data['attention_mask'])
    assert tuple(ids.shape) == tuple(masks.shape) == (392, 256)
    assert int(masks.sum()) == 100000
    data_load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32, local_files_only=True)
    moe = upcycle(model, 8, 2026110298)
    model.to('mps').train()
    setup_seconds = time.perf_counter() - start
    opt = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    assert_router_optimizer_only(model, opt)
    checkpoints = [0, 16000, 64000, 100000]
    checkpoint_dir = out.parent / 'preflight_router_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    checkpoint_seconds = 0.0

    def save_checkpoint(tokens: int) -> None:
        nonlocal checkpoint_seconds
        start = synced()
        path = checkpoint_dir / f'router_{tokens}.safetensors'
        save_file({'router.weight': moe.router.weight.detach().cpu().contiguous()}, str(path))
        saved.append({'tokens': tokens, 'bytes': path.stat().st_size, 'sha256': filehash(path)})
        checkpoint_seconds += synced() - start
    save_checkpoint(0)
    moe_forward_seconds = 0.0
    hook_start = [0.0]

    def pre(_module, _args):
        hook_start[0] = synced()

    def post(_module, _args, _out):
        nonlocal moe_forward_seconds
        moe_forward_seconds += synced() - hook_start[0]
    before = moe.register_forward_pre_hook(pre)
    after = moe.register_forward_hook(post)
    data_transfer_seconds = model_forward_seconds = backward_seconds = optimizer_seconds = 0.0
    consumed = 0
    loss_finite = grad_finite = True
    peak_driver = int(torch.mps.driver_allocated_memory())
    train_start = synced()
    for step in range(len(ids)):
        t = synced()
        x = ids[step:step + 1].to('mps')
        mask = masks[step:step + 1].to('mps')
        labels = x.masked_fill(mask == 0, -100)
        data_transfer_seconds += synced() - t
        opt.zero_grad(set_to_none=True)
        t = synced()
        loss = model(input_ids=x, attention_mask=mask, labels=labels, use_cache=False).loss
        model_forward_seconds += synced() - t
        loss_finite = loss_finite and bool(torch.isfinite(loss).item())
        t = synced()
        loss.backward()
        backward_seconds += synced() - t
        grad_finite = grad_finite and bool(torch.isfinite(moe.router.weight.grad).all().item())
        assert all((p.grad is None for p in model.parameters() if not p.requires_grad))
        t = synced()
        torch.nn.utils.clip_grad_norm_([moe.router.weight], 1.0)
        next_tokens = consumed + int(mask.sum().item())
        opt.param_groups[0]['lr'] = 0.001 * s1_lr_factor(next_tokens)
        opt.step()
        optimizer_seconds += synced() - t
        consumed = next_tokens
        for target in checkpoints[1:]:
            if consumed >= target and (not any((r['tokens'] == target for r in saved))):
                save_checkpoint(target)
        if step % 20 == 0:
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if step and step % 100 == 0:
            print(f'PREFLIGHT_BENCHMARK_PROGRESS steps={step} tokens={consumed}', flush=True)
    train_wall = synced() - train_start
    before.remove()
    after.remove()
    assert consumed == 100000 and len(saved) == 4 and loss_finite and grad_finite
    assert_router_optimizer_only(model, opt)
    peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
    start = time.perf_counter()
    response_bank_path = bank_dir / 'preflight_response_bank.pt'
    bank_meta = json.loads(response_bank_path.with_suffix('.json').read_text())
    assert bank_meta['status'] == 'PREFLIGHT_ONLY'
    assert filehash(response_bank_path) == bank_meta['tensor_sha256']
    response_bank = torch.load(response_bank_path, map_location='cpu', weights_only=True)
    response_data_load_seconds = time.perf_counter() - start
    response = response_pass(model, moe, response_bank, 2026110297)
    peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
    total_wall = time.perf_counter() - start_total
    result = {'status': 'PREFLIGHT_ONLY_COMPLETE', 'arm': 'S8', 'g': 8, 'H': 32, 'm': 192, 'device': 'mps', 'dtype': 'float32', 'benchmark_tokens': consumed, 'optimizer_steps': len(ids), 'source_bank_manifest_sha256': index['manifest_sha256']['preflight_benchmark'], 'optimizer_schedule': 'S1 12,800-token linear warmup; cosine decay to 0.1 peak at 256,000 tokens', 'data_load_seconds': data_load_seconds, 'bank_construction_seconds': bank_meta['total_seconds'], 'response_bank_data_load_seconds': response_data_load_seconds, 'setup_seconds': setup_seconds, 'base_forward_seconds': model_forward_seconds - moe_forward_seconds, 'router_moe_forward_seconds': moe_forward_seconds, 'model_forward_seconds': model_forward_seconds, 'backward_to_router_seconds': backward_seconds, 'optimizer_update_seconds': optimizer_seconds, 'data_transfer_seconds': data_transfer_seconds, 'checkpoint_io_seconds': checkpoint_seconds, 'training_loop_wall_seconds': train_wall, 'response_pass': response, 'total_wall_seconds': total_wall, 'peak_mps_driver_bytes_sampled': peak_driver, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'checkpoints': saved, 'loss_finite': loss_finite, 'router_gradient_finite': grad_finite, 'scientific_training_started': False, 'scientific_audit_observables_opened': False}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'tokens': consumed, 'train_wall_seconds': train_wall, 'response_wall_seconds': response['total_seconds'], 'total_wall_seconds': total_wall}))
    return result
if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
