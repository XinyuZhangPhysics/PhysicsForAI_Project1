# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'work'))
from s1_impl import ARMS, parent_vectors, upcycle
OUT = ROOT / 'outputs/exp_s1'
WORK = ROOT / 'work'
MODEL = WORK / 'model'
SEEDS = tuple(range(2026100101, 2026100109))
HORIZONS = (0, 16000, 64000, 256000)
TOTAL = 256000

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()

def frozen_hashes(model) -> dict:
    hashes = {'expert': hashlib.sha256(), 'base': hashlib.sha256(), 'buffers': hashlib.sha256()}
    for name, p in model.named_parameters():
        if name == 'gpt_neox.layers.3.mlp.router.weight':
            continue
        key = 'expert' if '.experts.' in name else 'base'
        hashes[key].update(name.encode() + b'\x00')
        hashes[key].update(tensor_bytes(p))
    for name, b in model.named_buffers():
        hashes['buffers'].update(name.encode() + b'\x00')
        hashes['buffers'].update(tensor_bytes(b))
    return {key: h.hexdigest() for key, h in hashes.items()}

def save_checkpoint(moe, folder: Path, horizon: int, model, initial_frozen: dict) -> dict:
    hashes = frozen_hashes(model)
    if hashes != initial_frozen:
        raise RuntimeError(f'EXP_S1_BLOCKED_TRAINING_INTEGRITY frozen tensor changed at {horizon}')
    path = folder / f'router_{horizon}.safetensors'
    save_file({'router.weight': moe.router.weight.detach().cpu().contiguous()}, str(path))
    return {'horizon_nonpadding_tokens': horizon, 'router_checkpoint': str(path.relative_to(ROOT)), 'router_checkpoint_sha256': sha_file(path), 'frozen_tensor_hashes': hashes, 'router_finite': bool(torch.isfinite(moe.router.weight).all().item())}

def lr_factor(tokens: int) -> float:
    if tokens <= 12800:
        return tokens / 12800
    progress = (tokens - 12800) / (TOTAL - 12800)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

def segments():
    for i in range(1000):
        if i == 62:
            yield (i, 0, 128)
            yield (i, 128, 256)
        else:
            yield (i, 0, 256)

def static_check():
    tokens = 0
    hits = [0]
    steps = 0
    for i, a, b in segments():
        assert b > a and 0 <= a < b <= 256
        tokens += b - a
        steps += 1
        if tokens in HORIZONS:
            hits.append(tokens)
    assert tokens == TOTAL and steps == 1001
    assert sorted(set(hits)) == list(HORIZONS)
    return {'steps': steps, 'tokens': tokens, 'checkpoint_horizons': list(HORIZONS), 'split_sequence_index': 62, 'split_lengths': [128, 128], 'learning_rate_basis': 'cumulative nonpadding tokens after each optimizer step'}

def run(arm: str, seed: int):
    assert arm in ARMS and seed in SEEDS
    assert (OUT / 'PROVENANCE_PREFLIGHT.json').is_file()
    assert json.loads((OUT / 'PROVENANCE_PREFLIGHT.json').read_text())['status'] == 'EXP_S1_PROVENANCE_VERIFIED'
    protocol = json.loads((OUT / 'RUN_PROTOCOL.json').read_text())
    assert protocol['status'] == 'FROZEN_BEFORE_SCIENTIFIC_EXECUTION'
    assert sha_file(Path(__file__)) == protocol['training_code_sha256']
    assert sha_file(WORK / 's1_impl.py') == protocol['preflight_impl_sha256']
    assert torch.backends.mps.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    frozen_stream = json.loads((ROOT / 'outputs/TRAINING_MANIFEST_HASHES.json').read_text())['seeds'][str(seed)]
    tensor_path = WORK / f'train_seed_{seed}.pt'
    manifest_path = ROOT / 'outputs' / f'TRAIN_SEED_{seed}_MANIFEST.jsonl'
    assert sha_file(tensor_path) == frozen_stream['tensor_sha256']
    assert sha_file(manifest_path) == frozen_stream['manifest_sha256']
    data = torch.load(tensor_path, map_location='cpu', weights_only=True)
    ids, masks = (data['input_ids'], data['attention_mask'])
    assert tuple(ids.shape) == tuple(masks.shape) == (1000, 256)
    assert int(masks.sum()) == TOTAL
    assert bool(masks.all())
    folder = OUT / 'checkpoints' / f'{arm}_seed_{seed}'
    if folder.exists():
        raise RuntimeError(f'EXP_S1_BLOCKED_TRAINING_INTEGRITY refusing duplicate run {folder}')
    folder.mkdir(parents=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True)
    moe = upcycle(model, ARMS[arm], seed)
    init_manifest = json.loads((ROOT / 'outputs/ROUTER_INITIALIZATION.json').read_text())
    init_sha = hashlib.sha256(tensor_bytes(moe.router.weight)).hexdigest()
    assert init_sha == init_manifest['seeds'][str(seed)]['arms'][arm]['router_weight_sha256']
    assert torch.equal(moe.router.weight.detach().cpu(), parent_vectors(seed, torch.float32).repeat_interleave(ARMS[arm], 0))
    model.to('mps')
    model.train()
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ['gpt_neox.layers.3.mlp.router.weight']
    initial_frozen = frozen_hashes(model)
    checkpoints = {'0': save_checkpoint(moe, folder, 0, model, initial_frozen)}
    opt = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    consumed = 0
    steps = 0
    loss_first = None
    loss_last = None
    max_loss = -math.inf
    peak_mps = torch.mps.current_allocated_memory()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    trace = folder / 'training_trace.jsonl'
    torch.mps.synchronize()
    started = time.perf_counter()
    with trace.open('w') as f:
        for i, a, b in segments():
            x = ids[i, a:b].unsqueeze(0).to('mps')
            attention = masks[i, a:b].unsqueeze(0).to('mps')
            assert bool(attention.all())
            labels = x.clone()
            opt.zero_grad(set_to_none=True)
            loss = model(input_ids=x, attention_mask=attention, labels=labels, use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('EXP_S1_BLOCKED_TRAINING_INTEGRITY nonfinite loss')
            loss.backward()
            if moe.router.weight.grad is None or not bool(torch.isfinite(moe.router.weight.grad).all().item()):
                raise RuntimeError('EXP_S1_BLOCKED_TRAINING_INTEGRITY nonfinite router gradient')
            grad_norm = torch.nn.utils.clip_grad_norm_([moe.router.weight], 1.0)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError('EXP_S1_BLOCKED_TRAINING_INTEGRITY nonfinite grad norm')
            consumed += int(attention.sum().item())
            steps += 1
            lr = 0.001 * lr_factor(consumed)
            opt.param_groups[0]['lr'] = lr
            opt.step()
            if not bool(torch.isfinite(moe.router.weight).all().item()):
                raise RuntimeError('EXP_S1_BLOCKED_TRAINING_INTEGRITY nonfinite router parameter')
            loss_value = float(loss.detach().cpu())
            if loss_first is None:
                loss_first = loss_value
            loss_last = loss_value
            max_loss = max(max_loss, loss_value)
            f.write(json.dumps({'step': steps, 'stream_sequence_index': i, 'stream_token_slice': [a, b], 'cumulative_nonpadding_tokens': consumed, 'lm_loss': loss_value, 'router_grad_norm_before_clip': float(grad_norm.detach().cpu()), 'router_lr': lr}, sort_keys=True, separators=(',', ':')) + '\n')
            if consumed in HORIZONS[1:]:
                torch.mps.synchronize()
                checkpoints[str(consumed)] = save_checkpoint(moe, folder, consumed, model, initial_frozen)
                print('CHECKPOINT', arm, seed, consumed, checkpoints[str(consumed)]['router_checkpoint_sha256'], flush=True)
            if steps % 100 == 0:
                torch.mps.synchronize()
                peak_mps = max(peak_mps, torch.mps.current_allocated_memory())
                peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        f.flush()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    assert consumed == TOTAL and steps == 1001 and (sorted((int(x) for x in checkpoints)) == list(HORIZONS))
    assert all((v['router_finite'] and v['frozen_tensor_hashes'] == initial_frozen for v in checkpoints.values()))
    result = {'status': 'PASS', 'arm': arm, 'g': ARMS[arm], 'H': 4 * ARMS[arm], 'm': 2048 // ARMS[arm], 'seed': seed, 'nonpadding_tokens_consumed': consumed, 'optimizer_steps': steps, 'training_manifest_sha256': frozen_stream['manifest_sha256'], 'training_tensor_sha256': frozen_stream['tensor_sha256'], 'router_initialization_sha256': init_sha, 'checkpoints': checkpoints, 'frozen_tensor_hashes': initial_frozen, 'trace_sha256': sha_file(trace), 'loss_first': loss_first, 'loss_last': loss_last, 'max_loss': max_loss, 'finite_loss_gradient_and_router': True, 'only_router_trainable': True, 'peak_mps_allocated_bytes': peak_mps, 'peak_process_rss_bytes': peak_rss, 'training_loop_wall_seconds': elapsed, 'training_loop_tokens_per_second': TOTAL / elapsed}
    (folder / 'RUN_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RUN_PASS', arm, seed, 'wall_s', round(elapsed, 3), flush=True)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--static-check', action='store_true')
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    if args.static_check:
        print(json.dumps(static_check(), indent=2))
        return
    if args.arm is None or args.seed is None:
        raise SystemExit('arm and seed required')
    run(args.arm, args.seed)
if __name__ == '__main__':
    main()
