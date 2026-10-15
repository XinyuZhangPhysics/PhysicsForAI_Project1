# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""EXP-S2 registered router-only training. Added after preregistration freeze."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gc
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
from s2_impl import ARMS, LAYER_INDEX, assert_router_optimizer_only, parent_vectors, upcycle
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OUT = ROOT / 'outputs/science_exp_s2'
BANK = ROOT / 'outputs/reports/exp_s2/preflight/banks'
FREEZE = ROOT / 'outputs/reports/exp_s2/preregistration_freeze/EXP_S2_PREREGISTRATION_FREEZE.json'
MODEL = WORK / 'smollm2-pinned'
SEEDS = tuple(range(2026110201, 2026110209))
HORIZONS = (0, 16000, 64000, 256000)
TRAINABLE = f'model.layers.{LAYER_INDEX}.mlp.router.weight'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def frozen_hashes(model) -> dict[str, str]:
    groups = {x: hashlib.sha256() for x in ('base', 'expert', 'buffers')}
    for name, p in model.named_parameters():
        if name == TRAINABLE:
            continue
        key = 'expert' if '.experts.' in name else 'base'
        groups[key].update(name.encode() + b'\x00')
        groups[key].update(p.detach().cpu().contiguous().numpy().tobytes())
    for name, b in model.named_buffers():
        groups['buffers'].update(name.encode() + b'\x00')
        groups['buffers'].update(b.detach().cpu().contiguous().numpy().tobytes())
    return {k: v.hexdigest() for k, v in groups.items()}

def lr_factor(tokens: int) -> float:
    if tokens <= 12800:
        return tokens / 12800
    progress = (tokens - 12800) / (256000 - 12800)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

def segments():
    for i in range(1000):
        if i == 62:
            yield (i, 0, 128)
            yield (i, 128, 256)
        else:
            yield (i, 0, 256)

def static_check() -> dict:
    tokens = 0
    hits = [0]
    steps = 0
    for i, a, b in segments():
        assert 0 <= a < b <= 256
        tokens += b - a
        steps += 1
        if tokens in HORIZONS:
            hits.append(tokens)
    assert tokens == 256000 and steps == 1001 and (hits == list(HORIZONS))
    assert lr_factor(12800) == 1.0 and abs(lr_factor(256000) - 0.1) < 1e-15
    return {'tokens': tokens, 'steps': steps, 'checkpoint_tokens': hits, 'split_index': 62, 'split_lengths': [128, 128]}

def save_checkpoint(folder: Path, moe, model, initial: dict, horizon: int) -> dict:
    current = frozen_hashes(model)
    if current != initial:
        raise RuntimeError(f'FROZEN_TENSOR_CHANGED at {horizon}')
    path = folder / f'router_{horizon}.safetensors'
    save_file({'router.weight': moe.router.weight.detach().cpu().contiguous()}, str(path))
    return {'horizon_nonpadding_tokens': horizon, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'frozen_tensor_hashes': current, 'finite': bool(torch.isfinite(moe.router.weight).all().item())}

def run(arm: str, seed: int) -> dict:
    assert arm in ARMS and seed in SEEDS and torch.backends.mps.is_available()
    assert static_check()['steps'] == 1001
    frozen = json.loads(FREEZE.read_text())
    stream = frozen['training_streams']['seeds'][str(seed)]
    tensor_path = BANK / f'future_training_seed_{seed}_tokens.pt'
    doc_path = BANK / f'future_training_seed_{seed}.jsonl'
    seq_path = BANK / f'future_training_seed_{seed}_sequences.jsonl'
    assert sha(tensor_path) == stream['token_tensor_sha256']
    assert sha(doc_path) == stream['document_ids_sha256']
    assert sha(seq_path) == stream['sequence_manifest_sha256']
    data = torch.load(tensor_path, map_location='cpu', weights_only=True)
    ids, masks = (data['input_ids'], data['attention_mask'])
    assert ids.shape == masks.shape == (1000, 256)
    assert bool(masks.all()) and int(masks.sum()) == 256000
    folder = OUT / 'checkpoints' / f'{arm}_seed_{seed}'
    if folder.exists():
        raise RuntimeError(f'DUPLICATE_OR_PARTIAL_RUN {folder}')
    folder.mkdir(parents=True)
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True)
    moe = upcycle(model, ARMS[arm], seed)
    assert torch.equal(moe.router.weight.detach().cpu(), parent_vectors(seed, torch.float32).repeat_interleave(ARMS[arm], 0))
    init_sha = hashlib.sha256(moe.router.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    model.to('mps').train()
    assert [n for n, p in model.named_parameters() if p.requires_grad] == [TRAINABLE]
    initial = frozen_hashes(model)
    checkpoints = {'0': save_checkpoint(folder, moe, model, initial, 0)}
    optimizer = torch.optim.AdamW([moe.router.weight], lr=0.001, betas=(0.9, 0.95), eps=1e-08, weight_decay=0)
    assert_router_optimizer_only(model, optimizer)
    consumed = steps = 0
    first = last = None
    max_loss = -math.inf
    peak_mps = int(torch.mps.current_allocated_memory())
    trace = folder / 'training_trace.jsonl'
    torch.mps.synchronize()
    start = time.perf_counter()
    with trace.open('w') as f:
        for i, a, b in segments():
            x = ids[i, a:b].unsqueeze(0).to('mps')
            mask = masks[i, a:b].unsqueeze(0).to('mps')
            assert bool(mask.all())
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=x, attention_mask=mask, labels=x.clone(), use_cache=False).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError('NONFINITE_LOSS')
            loss.backward()
            grad = moe.router.weight.grad
            if grad is None or not bool(torch.isfinite(grad).all().item()):
                raise RuntimeError('NONFINITE_ROUTER_GRADIENT')
            if any((p.grad is not None for p in model.parameters() if not p.requires_grad)):
                raise RuntimeError('FROZEN_GRADIENT')
            norm = torch.nn.utils.clip_grad_norm_([moe.router.weight], 1.0)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError('NONFINITE_GRADIENT_NORM')
            consumed += int(mask.sum().item())
            steps += 1
            lr = 0.001 * lr_factor(consumed)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.step()
            if not bool(torch.isfinite(moe.router.weight).all().item()):
                raise RuntimeError('NONFINITE_ROUTER')
            value = float(loss.detach().cpu())
            if first is None:
                first = value
            last = value
            max_loss = max(max_loss, value)
            f.write(json.dumps({'step': steps, 'stream_sequence_index': i, 'stream_token_slice': [a, b], 'cumulative_nonpadding_tokens': consumed, 'lm_loss': value, 'router_grad_norm_before_clip': float(norm.detach().cpu()), 'router_lr': lr}, sort_keys=True, separators=(',', ':')) + '\n')
            if consumed in HORIZONS[1:]:
                torch.mps.synchronize()
                checkpoints[str(consumed)] = save_checkpoint(folder, moe, model, initial, consumed)
                print('CHECKPOINT', arm, seed, consumed, flush=True)
            if steps % 100 == 0:
                torch.mps.synchronize()
                peak_mps = max(peak_mps, int(torch.mps.current_allocated_memory()))
        f.flush()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    assert consumed == 256000 and steps == 1001
    assert sorted((int(x) for x in checkpoints)) == list(HORIZONS)
    assert all((c['finite'] and c['frozen_tensor_hashes'] == initial for c in checkpoints.values()))
    assert_router_optimizer_only(model, optimizer)
    result = {'status': 'PASS', 'arm': arm, 'g': ARMS[arm], 'H': 4 * ARMS[arm], 'm': 1536 // ARMS[arm], 'seed': seed, 'nonpadding_tokens_consumed': consumed, 'optimizer_steps': steps, 'training_tensor_sha256': stream['token_tensor_sha256'], 'training_document_manifest_sha256': stream['document_ids_sha256'], 'training_sequence_manifest_sha256': stream['sequence_manifest_sha256'], 'router_initialization_sha256': init_sha, 'checkpoints': checkpoints, 'frozen_tensor_hashes': initial, 'trace_sha256': sha(trace), 'loss_first': first, 'loss_last': last, 'max_loss': max_loss, 'finite_loss_gradient_and_router': True, 'only_router_trainable': True, 'peak_mps_allocated_bytes': peak_mps, 'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'training_loop_wall_seconds': elapsed}
    write(folder / 'RUN_RESULT.json', result)
    print('RUN_PASS', arm, seed, round(elapsed, 3), flush=True)
    del model, moe, optimizer
    gc.collect()
    torch.mps.empty_cache()
    return result

def all_runs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            runs.append(run(arm, seed))
    assert len(runs) == 32
    result = {'status': 'PASS', 'runs': 32, 'total_nonpadding_tokens': 8192000, 'optimizer_steps_per_run': 1001, 'run_result_sha256': {f"{r['arm']}_{r['seed']}": sha(OUT / 'checkpoints' / f"{r['arm']}_seed_{r['seed']}" / 'RUN_RESULT.json') for r in runs}}
    write(OUT / 'TRAINING_INTEGRITY.json', result)
    return result
if __name__ == '__main__':
    if sys.argv[1:] == ['--static-check']:
        print(json.dumps(static_check(), indent=2))
    elif sys.argv[1:] == ['--all']:
        print(json.dumps(all_runs(), indent=2))
    else:
        raise SystemExit('usage: science_train.py --static-check|--all')
