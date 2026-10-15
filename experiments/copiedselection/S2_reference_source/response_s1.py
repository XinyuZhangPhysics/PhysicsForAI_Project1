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
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work'
OUT = ROOT / 'outputs/exp_s1'
sys.path.insert(0, str(WORK))
from s1_impl import ARMS
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
SEEDS = tuple(range(2026100101, 2026100109))
HORIZONS = (0, 16000, 64000, 256000)
RESOLUTION_POWER = 1e-20

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def paths(membership: str):
    assert membership in ('calibration', 'audit')
    if membership == 'audit' and (not (OUT / 'AUDIT_OPENING_RECORD.json').exists()):
        raise RuntimeError('AUDIT_NOT_OPENED')
    if membership == 'calibration' and (not (OUT / 'CHECKPOINT_HASHES.json').exists()):
        raise RuntimeError('CHECKPOINT_HASHES_NOT_FROZEN')
    rep = WORK / f's1_{membership}_representation.pt'
    if membership == 'calibration':
        meta = json.loads((ROOT / 'outputs/SECTOR_CONTROLS.json').read_text())
        assert sha(rep) == meta['representation_tensor_sha256']
    else:
        meta = json.loads((OUT / 'AUDIT_REPRESENTATION.json').read_text())
        assert sha(rep) == meta['representation_tensor_sha256']
    bank = ROOT / 'outputs' / f'{membership}_bank_manifest.jsonl'
    pair = ROOT / 'outputs' / f'{membership}_pair_manifest.jsonl'
    pre = json.loads((ROOT / 'outputs/DATA_MANIFEST_HASHES.json').read_text())
    assert sha(bank) == pre['manifest_sha256'][membership]
    assert sha(pair) == pre['manifest_sha256'][membership + '_pairs']
    return (rep, bank, pair)

def pair_indices(bank: Path, pair: Path):
    rows = [json.loads(x) for x in bank.open()]
    assert len(rows) == 2048
    ids = {r['train_row_id']: i for i, r in enumerate(rows)}
    pairs = [json.loads(x) for x in pair.open()]
    assert len(pairs) == 12288
    src = np.asarray([ids[r['source_train_row_id']] for r in pairs], dtype=np.int32)
    tgt = np.asarray([ids[r['target_train_row_id']] for r in pairs], dtype=np.int32)
    keys = list(dict.fromkeys(((r['source_domain'], r['target_domain']) for r in pairs)))
    assert len(keys) == 12 and all((a != b for a, b in keys))
    group = np.asarray([keys.index((r['source_domain'], r['target_domain'])) for r in pairs], dtype=np.int8)
    assert all((np.sum(group == i) == 1024 for i in range(12)))
    return (src, tgt, group, keys)

def prepare(membership: str):
    rep_path, bank, pair = paths(membership)
    h = torch.load(rep_path, map_location='cpu', weights_only=True)['h']
    assert tuple(h.shape) == (2048, 512) and bool(torch.isfinite(h).all())
    src, tgt, group, keys = pair_indices(bank, pair)
    model = AutoModelForCausalLM.from_pretrained(str(WORK / 'model'), dtype=torch.float32, local_files_only=True).eval()
    dense = model.gpt_neox.layers[3].mlp.to('mps')
    with torch.inference_mode():
        activation = dense.act(dense.dense_h_to_4h(h.to('mps'))).cpu().numpy().astype(np.float64)
    assert activation.shape == (2048, 2048)
    del model, dense
    geometry = {}
    for arm, g in ARMS.items():
        m = 2048 // g
        child = np.empty((len(src), g), dtype=np.float64)
        for j in range(g):
            sl = slice(j * m, (j + 1) * m)
            for a in range(0, len(src), 1024):
                b = min(a + 1024, len(src))
                child[a:b, j] = (activation[src[a:b], sl] * activation[tgt[a:b], sl]).mean(axis=1)
        k = np.broadcast_to(child[:, None, :], (len(src), 4, g)).copy()
        mean = k.mean(axis=(1, 2), keepdims=True)
        kc = np.broadcast_to(k.mean(axis=1, keepdims=True) - mean, k.shape).reshape(len(src), 4 * g).copy()
        ka = np.broadcast_to(k.mean(axis=2, keepdims=True) - mean, k.shape)
        km = k - mean - kc.reshape(len(src), 4, g) - ka
        forbidden = float(np.square(ka).sum(axis=(1, 2)).mean() + np.square(km).sum(axis=(1, 2)).mean())
        assert forbidden <= RESOLUTION_POWER
        K = float(np.square(kc).sum(axis=1).mean())
        if g == 1:
            assert K == 0
        else:
            assert K > RESOLUTION_POWER
        path = WORK / 'scientific' / f'{membership}_geometry_{arm}.npz'
        np.savez_compressed(path, h=h.numpy(), src=src, tgt=tgt, group=group, kappa_C=kc)
        geometry[arm] = {'g': g, 'H': 4 * g, 'm': m, 'K_C_0': K, 'forbidden_kappa_A_M_power': forbidden, 'cache_sha256': sha(path), 'cache_path': str(path.relative_to(ROOT))}
        print('GEOMETRY', membership, arm, 'K_C', K, flush=True)
    meta = {'status': 'FROZEN', 'membership': membership, 'representation_tensor_sha256': sha(rep_path), 'bank_manifest_sha256': sha(bank), 'pair_manifest_sha256': sha(pair), 'ordered_pair_keys': [list(x) for x in keys], 'router_field_convention': 'q_raw(pair,i)=softmax(router(h_source))_i * softmax(router(h_target))_i; center each pair over H; Pq=mean_pair(sum_i(centered^2)/(H-1)); qhat=centered/sqrt(Pq)', 'geometry_convention': 'kappa(pair,i)=mean_a(GELU(w_ia h+b_ia) GELU(w_ia h_prime+b_ia)); expert-center each pair; project J4 tensor Pi_g', 'power_convention': 'Q_C and K_C are mean_pair sum_i(field_C^2); B_coordinate is mean of pairwise q_C dot kappa_C within each 1024-pair group; P_B is mean of 12 squared coordinates; Theta=P_B/(Q_C*K_C)', 'resolution_power_threshold': RESOLUTION_POWER, 'arms': geometry}
    (OUT / f'{membership.upper()}_GEOMETRY.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta

def measure(membership: str, arm: str, seed: int, horizon: int):
    if membership == 'audit' and horizon != 64000 and (arm != 'S1') and (not (OUT / 'PRIMARY_FREEZE.json').exists()):
        raise RuntimeError('SECONDARY_AUDIT_BEFORE_PRIMARY_FREEZE')
    if arm not in ARMS or seed not in SEEDS or horizon not in HORIZONS:
        raise ValueError((arm, seed, horizon))
    meta = json.loads((OUT / f'{membership.upper()}_GEOMETRY.json').read_text())
    item = meta['arms'][arm]
    cache = ROOT / item['cache_path']
    assert sha(cache) == item['cache_sha256']
    values = np.load(cache)
    h = values['h']
    src = values['src']
    tgt = values['tgt']
    group = values['group']
    kc = values['kappa_C']
    g = ARMS[arm]
    H = 4 * g
    run = json.loads((OUT / 'checkpoints' / f'{arm}_seed_{seed}' / 'RUN_RESULT.json').read_text())
    ckpt = run['checkpoints'][str(horizon)]
    path = ROOT / ckpt['router_checkpoint']
    assert sha(path) == ckpt['router_checkpoint_sha256']
    state = load_file(str(path))
    weight = state['router.weight']
    assert tuple(weight.shape) == (H, 512)
    with torch.inference_mode():
        probs = torch.softmax(torch.nn.functional.linear(torch.from_numpy(h).to('mps'), weight.to('mps')), dim=-1).cpu().double().numpy()
    raw = probs[src] * probs[tgt]
    centered = raw - raw.mean(axis=1, keepdims=True)
    Pq = float(np.mean(np.square(centered).sum(axis=1) / (H - 1)))
    assert math.isfinite(Pq) and Pq > 0
    qhat = centered / math.sqrt(Pq)
    q3 = qhat.reshape(len(src), 4, g)
    qc = np.broadcast_to(q3.mean(axis=1, keepdims=True) - q3.mean(axis=(1, 2), keepdims=True), q3.shape).reshape(len(src), H)
    Q = float(np.square(qc).sum(axis=1).mean())
    K = float(item['K_C_0'])
    direct_pair = np.sum(qhat * kc, axis=1)
    projected_pair = np.sum(qc * kc, axis=1)
    direct = np.asarray([direct_pair[group == i].mean() for i in range(12)], dtype=np.float64)
    projected = np.asarray([projected_pair[group == i].mean() for i in range(12)], dtype=np.float64)
    response_residual = float(np.max(np.abs(direct - projected)))
    P = float(np.square(projected).mean())
    direct_P = float(np.square(direct).mean())
    denominator = Q * K
    theta = P / denominator if denominator > RESOLUTION_POWER and P > RESOLUTION_POWER else None
    factor_error = abs(P - denominator * theta) if theta is not None else None
    resolved = all((math.isfinite(x) and x > RESOLUTION_POWER for x in (Q, K, P))) and theta is not None and math.isfinite(theta)
    if response_residual > 1e-08 or abs(P - direct_P) > 1e-08:
        raise RuntimeError(f'RESPONSE_PROJECTOR_IDENTITY_FAILED {membership} {arm} {seed} {horizon}')
    result = {'membership': membership, 'arm': arm, 'g': g, 'H': H, 'm': 2048 // g, 'seed': seed, 'horizon_nonpadding_tokens': horizon, 'router_checkpoint_sha256': ckpt['router_checkpoint_sha256'], 'geometry_cache_sha256': item['cache_sha256'], 'Pq_normalizer': Pq, 'Q_C': Q, 'K_C_0': K, 'P_B': P, 'Theta_R': theta, 'resolved_positive': resolved, 'B_coordinates': projected.tolist(), 'B_coordinate_powers': np.square(projected).tolist(), 'direct_response_power': direct_P, 'response_projector_max_abs_error': response_residual, 'factorization_abs_error': factor_error, 'S1_H1_all_coordinates_below_zero_floor': bool(np.all(np.square(projected) <= RESOLUTION_POWER)) if g == 1 else None}
    target = OUT / f'{membership}_responses' / f'{arm}_seed_{seed}_t{horizon}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('RESPONSE', membership, arm, seed, horizon, 'Q', Q, 'K', K, 'P', P, 'Theta', theta, flush=True)
    return result

def bank(membership, stage):
    if membership == 'calibration' and stage != 'all':
        raise ValueError(stage)
    if membership == 'audit' and stage not in ('primary', 'secondary'):
        raise ValueError(stage)
    for seed in SEEDS:
        for arm in ARMS:
            if membership == 'calibration':
                times = HORIZONS
            elif stage == 'primary':
                times = HORIZONS if arm == 'S1' else (64000,)
            else:
                times = () if arm == 'S1' else (16000, 256000)
            for t in times:
                measure(membership, arm, seed, t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'measure-bank'))
    parser.add_argument('membership', choices=('calibration', 'audit'))
    parser.add_argument('stage', nargs='?', default='all')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.membership)
    else:
        bank(args.membership, args.stage)
if __name__ == '__main__':
    main()
