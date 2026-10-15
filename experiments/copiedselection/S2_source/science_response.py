# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""EXP-S2 registered calibration/audit response estimator. Added pre-science."""
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
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from build_banks import read_selected_texts
from s2_impl import ARMS, LAYER_INDEX, projectors
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OUT = ROOT / 'outputs/science_exp_s2'
BANK = ROOT / 'outputs/reports/exp_s2/preflight/banks'
MODEL = WORK / 'smollm2-pinned'
DATA = Path(_p1_path('path_8c5dea5cdb13'))
SEEDS = tuple(range(2026110201, 2026110209))
HORIZONS = (0, 16000, 64000, 256000)
FLOOR = 1e-20
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
RESPONSE_KEYS = ('membership', 'arm', 'g', 'H', 'm', 'seed', 'horizon_nonpadding_tokens', 'router_checkpoint_sha256', 'geometry_cache_sha256', 'Pq_normalizer', 'Q_C', 'K_C_0', 'P_B', 'Theta_R', 'resolved_positive', 'B_coordinates', 'B_coordinate_powers', 'direct_response_power', 'response_projector_max_abs_error', 'factorization_abs_error', 'S2_H1_all_coordinates_below_zero_floor')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def paths(membership: str):
    assert membership in ('calibration', 'audit')
    if membership == 'audit' and (not (OUT / 'AUDIT_OPENING_RECORD.json').exists()):
        raise RuntimeError('AUDIT_NOT_OPENED')
    if membership == 'calibration' and (not (OUT / 'TRAINING_INTEGRITY.json').exists()):
        raise RuntimeError('TRAINING_NOT_COMPLETE')
    index = json.loads((BANK / 'BANKS_INDEX.json').read_text())
    doc = BANK / f'future_{membership}_document_ids.jsonl'
    pair = BANK / f'future_{membership}_pair_ids.jsonl'
    assert sha(doc) == index['manifest_sha256'][f'future_{membership}']
    assert sha(pair) == index['manifest_sha256'][f'future_{membership}_pairs']
    return (doc, pair)

def pair_indices(doc: Path, pair: Path):
    rows = [json.loads(x) for x in doc.open()]
    assert len(rows) == 2048
    ids = {r['train_row_id']: i for i, r in enumerate(rows)}
    pairs = [json.loads(x) for x in pair.open()]
    assert len(pairs) == 12288
    src = np.asarray([ids[r['source_train_row_id']] for r in pairs], dtype=np.int32)
    tgt = np.asarray([ids[r['target_train_row_id']] for r in pairs], dtype=np.int32)
    keys = list(dict.fromkeys(((r['source_domain'], r['target_domain']) for r in pairs)))
    assert len(keys) == 12 and all((a != b for a, b in keys))
    group = np.asarray([keys.index((r['source_domain'], r['target_domain'])) for r in pairs], dtype=np.int8)
    assert all((int(np.sum(group == i)) == 1024 for i in range(12)))
    return (rows, src, tgt, group, keys)

def materialize_bank(membership: str) -> Path:
    doc, pair = paths(membership)
    rows, _, _, _, _ = pair_indices(doc, pair)
    texts = read_selected_texts(DATA, rows)
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    eos = tok.eos_token_id
    ids, masks = ([], [])
    for row in rows:
        tokens = tok(texts[row['text_sha256']], add_special_tokens=False, truncation=True, max_length=256)['input_ids']
        if not tokens:
            tokens = [eos]
        ids.append(tokens + [eos] * (256 - len(tokens)))
        masks.append([1] * len(tokens) + [0] * (256 - len(tokens)))
    path = OUT / f'{membership}_bank_tokens.pt'
    torch.save({'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}, path)
    write(OUT / f'{membership.upper()}_BANK_TOKENS.json', {'status': 'VERIFIED', 'membership': membership, 'documents': 2048, 'document_manifest_sha256': sha(doc), 'pair_manifest_sha256': sha(pair), 'tensor_sha256': sha(path), 'tokenizer_json_sha256': sha(MODEL / 'tokenizer.json')})
    return path

def representation(membership: str) -> Path:
    if membership == 'audit' and (not (OUT / 'AUDIT_OPENING_RECORD.json').exists()):
        raise RuntimeError('AUDIT_NOT_OPENED')
    path = OUT / f'{membership}_bank_tokens.pt'
    meta = json.loads((OUT / f'{membership.upper()}_BANK_TOKENS.json').read_text())
    assert sha(path) == meta['tensor_sha256']
    bank = torch.load(path, map_location='cpu', weights_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True).to('mps').eval()
    layer = model.model.layers[LAYER_INDEX]
    captured = []

    def capture(_module, args):
        captured.append(args[0].detach())
    hook = layer.mlp.register_forward_pre_hook(capture)
    reps = []
    with torch.inference_mode():
        for i in range(0, 2048, 8):
            ids = bank['input_ids'][i:i + 8].to('mps')
            masks = bank['attention_mask'][i:i + 8].to('mps')
            captured.clear()
            model.model(input_ids=ids, attention_mask=masks, use_cache=False)
            assert len(captured) == 1
            h = captured[0]
            last = masks.sum(1).long() - 1
            reps.append(h[torch.arange(len(h), device='mps'), last].cpu())
    hook.remove()
    h = torch.cat(reps, dim=0).contiguous()
    assert h.shape == (2048, 576) and bool(torch.isfinite(h).all())
    path = OUT / f'{membership}_representation.pt'
    torch.save({'h': h}, path)
    write(OUT / f'{membership.upper()}_REPRESENTATION.json', {'status': 'VERIFIED', 'membership': membership, 'shape': [2048, 576], 'tensor_sha256': sha(path), 'bank_tokens_sha256': meta['tensor_sha256'], 'layer_one_indexed': 20, 'final_nonpadding_token': True})
    del model
    torch.mps.empty_cache()
    return path

def geometry(membership: str) -> dict:
    doc, pair = paths(membership)
    rep = OUT / f'{membership}_representation.pt'
    rep_meta = json.loads((OUT / f'{membership.upper()}_REPRESENTATION.json').read_text())
    assert sha(rep) == rep_meta['tensor_sha256']
    h = torch.load(rep, map_location='cpu', weights_only=True)['h']
    assert h.shape == (2048, 576) and bool(torch.isfinite(h).all())
    _, src, tgt, group, keys = pair_indices(doc, pair)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True).eval()
    dense = model.model.layers[LAYER_INDEX].mlp.to('mps')
    with torch.inference_mode():
        hm = h.to('mps')
        features = (dense.act_fn(dense.gate_proj(hm)) * dense.up_proj(hm)).cpu().double().numpy()
    assert features.shape == (2048, 1536) and np.isfinite(features).all()
    arms = {}
    for arm, g in ARMS.items():
        m = 1536 // g
        child = np.empty((len(src), g), dtype=np.float64)
        for j in range(g):
            sl = slice(j * m, (j + 1) * m)
            for a in range(0, len(src), 1024):
                b = min(a + 1024, len(src))
                child[a:b, j] = (features[src[a:b], sl] * features[tgt[a:b], sl]).mean(axis=1)
        k = np.broadcast_to(child[:, None, :], (len(src), 4, g)).copy()
        mean = k.mean(axis=(1, 2), keepdims=True)
        kc = np.broadcast_to(k.mean(axis=1, keepdims=True) - mean, k.shape).reshape(len(src), 4 * g).copy()
        ka = np.broadcast_to(k.mean(axis=2, keepdims=True) - mean, k.shape)
        km = k - mean - kc.reshape(len(src), 4, g) - ka
        forbidden = float(np.square(ka).sum(axis=(1, 2)).mean() + np.square(km).sum(axis=(1, 2)).mean())
        if forbidden > FLOOR:
            raise RuntimeError('KAPPA_FORBIDDEN_SECTOR_NONZERO')
        K = float(np.square(kc).sum(axis=1).mean())
        if g == 1:
            assert K == 0
        else:
            assert K > FLOOR
        cache = OUT / f'{membership}_geometry_{arm}.npz'
        np.savez_compressed(cache, h=h.numpy(), src=src, tgt=tgt, group=group, kappa_C=kc)
        arms[arm] = {'g': g, 'H': 4 * g, 'm': m, 'K_C_0': K, 'forbidden_kappa_A_M_power': forbidden, 'cache_path': str(cache.relative_to(ROOT)), 'cache_sha256': sha(cache)}
    result = {'status': 'FROZEN', 'membership': membership, 'representation_tensor_sha256': sha(rep), 'bank_manifest_sha256': sha(doc), 'pair_manifest_sha256': sha(pair), 'ordered_pair_keys': [list(x) for x in keys], 'router_field_convention': 'q_raw=softmax(router(h_source))*softmax(router(h_target)); pair-center; Pq=mean_pair(sum_i(centered_i^2)/(H-1)); qhat=centered/sqrt(Pq)', 'geometry_convention': 'SwiGLU phi; child coordinate mean of phi(source)*phi(target); repeated across four parents; C=J4 tensor Pi_g', 'power_convention': 'Q_C,K_C=mean_pair sum_i(field_C^2); B_coordinate=mean_1024pairs(q_C dot kappa_C); P_B=mean_12(B_coordinate^2); Theta=P_B/(Q_C*K_C)', 'projection_dtype': 'CPU float64', 'resolution_power_threshold': FLOOR, 'arms': arms}
    write(OUT / f'{membership.upper()}_GEOMETRY.json', result)
    del model
    torch.mps.empty_cache()
    return result

def measure(membership: str, arm: str, seed: int, horizon: int) -> dict:
    if membership == 'audit' and (not (OUT / 'AUDIT_OPENING_RECORD.json').exists()):
        raise RuntimeError('AUDIT_NOT_OPENED')
    if membership == 'audit' and arm != 'S1' and (horizon != 64000) and (not (OUT / 'PRIMARY_FREEZE.json').exists()):
        raise RuntimeError('SECONDARY_BEFORE_PRIMARY_FREEZE')
    assert membership in ('calibration', 'audit') and arm in ARMS and (seed in SEEDS) and (horizon in HORIZONS)
    meta = json.loads((OUT / f'{membership.upper()}_GEOMETRY.json').read_text())
    item = meta['arms'][arm]
    cache = ROOT / item['cache_path']
    assert sha(cache) == item['cache_sha256']
    with np.load(cache) as values:
        h, src, tgt, group, kc = (values[x] for x in ('h', 'src', 'tgt', 'group', 'kappa_C'))
    g = ARMS[arm]
    H = 4 * g
    run_path = OUT / 'checkpoints' / f'{arm}_seed_{seed}' / 'RUN_RESULT.json'
    run = json.loads(run_path.read_text())
    assert run['status'] == 'PASS' and run['seed'] == seed and (run['arm'] == arm)
    ckpt = run['checkpoints'][str(horizon)]
    ckpt_path = ROOT / ckpt['path']
    assert sha(ckpt_path) == ckpt['sha256']
    weight = load_file(str(ckpt_path))['router.weight']
    assert weight.shape == (H, 576)
    with torch.inference_mode():
        probs = torch.softmax(torch.nn.functional.linear(torch.from_numpy(h).to('mps'), weight.to('mps')), dim=-1).cpu().double().numpy()
    raw = probs[src] * probs[tgt]
    centered = raw - raw.mean(axis=1, keepdims=True)
    Pq = float(np.mean(np.square(centered).sum(axis=1) / (H - 1)))
    if not math.isfinite(Pq) or Pq <= 0:
        raise RuntimeError('PQ_UNRESOLVED')
    qhat = centered / math.sqrt(Pq)
    q3 = qhat.reshape(len(src), 4, g)
    qc = np.broadcast_to(q3.mean(axis=1, keepdims=True) - q3.mean(axis=(1, 2), keepdims=True), q3.shape).reshape(len(src), H).copy()
    matrices = projectors(g)
    q_t = torch.from_numpy(qhat)
    for name in ('A', 'C', 'M'):
        assert matrices[name].dtype == torch.float64
    explicit_c = (q_t @ matrices['C'].T).numpy()
    if float(np.max(np.abs(explicit_c - qc))) > 1e-12:
        raise RuntimeError('C_PROJECTOR_MISMATCH')
    if float(np.max(np.abs(qhat - (q_t @ matrices['A'].T + q_t @ matrices['C'].T + q_t @ matrices['M'].T).numpy()))) > 1e-12:
        raise RuntimeError('A_C_M_DECOMPOSITION_MISMATCH')
    Q = float(np.square(qc).sum(axis=1).mean())
    K = float(item['K_C_0'])
    direct_pair = np.sum(qhat * kc, axis=1)
    projected_pair = np.sum(qc * kc, axis=1)
    direct = np.asarray([direct_pair[group == i].mean() for i in range(12)], dtype=np.float64)
    projected = np.asarray([projected_pair[group == i].mean() for i in range(12)], dtype=np.float64)
    residual = float(np.max(np.abs(direct - projected)))
    P = float(np.square(projected).mean())
    direct_P = float(np.square(direct).mean())
    denominator = Q * K
    theta = P / denominator if denominator > FLOOR and P > FLOOR else None
    factor_error = abs(P - denominator * theta) if theta is not None else None
    resolved = all((math.isfinite(x) and x > FLOOR for x in (Q, K, P))) and theta is not None and math.isfinite(theta)
    if residual > 1e-08 or abs(P - direct_P) > 1e-08:
        raise RuntimeError('RESPONSE_PROJECTOR_IDENTITY_FAILED')
    result = {'membership': membership, 'arm': arm, 'g': g, 'H': H, 'm': 1536 // g, 'seed': seed, 'horizon_nonpadding_tokens': horizon, 'router_checkpoint_sha256': ckpt['sha256'], 'geometry_cache_sha256': item['cache_sha256'], 'Pq_normalizer': Pq, 'Q_C': Q, 'K_C_0': K, 'P_B': P, 'Theta_R': theta, 'resolved_positive': resolved, 'B_coordinates': projected.tolist(), 'B_coordinate_powers': np.square(projected).tolist(), 'direct_response_power': direct_P, 'response_projector_max_abs_error': residual, 'factorization_abs_error': factor_error, 'S2_H1_all_coordinates_below_zero_floor': bool(np.all(np.square(projected) <= FLOOR)) if g == 1 else None}
    assert tuple(result) == RESPONSE_KEYS
    target = OUT / f'{membership}_responses' / f'{arm}_seed_{seed}_t{horizon}.json'
    if target.exists():
        raise RuntimeError(f'DUPLICATE_RESPONSE {target}')
    write(target, result)
    print('RESPONSE', membership, arm, seed, horizon, flush=True)
    return result

def measure_bank(membership: str, stage: str):
    if membership == 'calibration':
        assert stage == 'all'
    else:
        assert stage in ('primary', 'secondary')
    for seed in SEEDS:
        for arm in ARMS:
            times = HORIZONS if membership == 'calibration' or (stage == 'primary' and arm == 'S1') else (64000,) if stage == 'primary' else () if arm == 'S1' else (16000, 256000)
            for t in times:
                measure(membership, arm, seed, t)
if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit('usage: science_response.py materialize|representation|geometry MEMBERSHIP | measure-bank MEMBERSHIP STAGE')
    command, membership, *args = sys.argv[1:]
    if command == 'materialize':
        materialize_bank(membership)
    elif command == 'representation':
        representation(membership)
    elif command == 'geometry':
        geometry(membership)
    elif command == 'measure-bank':
        measure_bank(membership, args[0])
    else:
        raise SystemExit(command)
