# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen-protocol J1-C local target-layer SGD runner.

No science mode can run without a separate, post-freeze authorization record.
This module has no work on import. The current preflight uses excluded data only.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from j1_core import response, router_q
from j1_torch_backend import CoTrainMoE
from run_real_preflight import features, geometry, pairs
ORIGINAL_J1_FREEZE_SHA256 = 'de1d181356aba62f526893dd61a5b3a348e4207397b45c9dd4c795fa810560ef'
AMENDMENT_SHA256 = '84834538af6753fc5f1adf88260b2a4770418107195299522924dc106c94438f'
STREAM_INDEX_SHA256 = '65de5239b93af80beb22d1a01c6d715d09f5ec59a6a2eeac5b615b75f9b994be'
SEEDS = tuple(range(2026110201, 2026110209))
GRANULARITIES = (1, 4)
ARMS = ('R', 'E', 'J')
CHECKPOINT_UPDATES = (0, 1, 4, 16)
CHECKPOINT_TOKENS = (0, 1024, 4096, 16384)
DOMAIN_ORDER = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_sha(x: torch.Tensor | np.ndarray) -> str:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def frozen_model_sha(model: torch.nn.Module) -> str:
    """Hash all non-target parameters and buffers for C0 isolation evidence."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if 'model.layers.19.mlp.' in name:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def save_target_parameters(moe: CoTrainMoE, path: Path) -> None:
    payload = {'router': moe.router.weight.detach().cpu().numpy()}
    for i, expert in enumerate(moe.experts):
        for block in ('gate_proj', 'up_proj', 'down_proj'):
            payload[f'expert_{i}_{block}'] = getattr(expert, block).weight.detach().cpu().numpy()
    np.savez_compressed(path, **payload)

def validate_authorities(original_freeze: Path, amendment: Path, stream_index: Path) -> dict:
    if sha(original_freeze) != ORIGINAL_J1_FREEZE_SHA256:
        raise ValueError('original frozen J1 authority hash mismatch')
    if sha(amendment) != AMENDMENT_SHA256:
        raise ValueError('v1.1 amendment hash mismatch')
    if sha(stream_index) != STREAM_INDEX_SHA256:
        raise ValueError('inherited training stream index hash mismatch')
    frozen, amended, index = (json.loads(p.read_text()) for p in (original_freeze, amendment, stream_index))
    if frozen['seeds'] != list(SEEDS) or frozen['primary_g'] != list(GRANULARITIES):
        raise ValueError('frozen J1 seed/g mismatch')
    if amended['effective_checkpoint_tokens'] != list(CHECKPOINT_TOKENS) or amended['checkpoint_update_indices'] != list(CHECKPOINT_UPDATES):
        raise ValueError('v1.1 checkpoint mismatch')
    if amended['original_j1_freeze_sha256'] != ORIGINAL_J1_FREEZE_SHA256:
        raise ValueError('amendment does not anchor original J1')
    if index['domain_order'] != list(DOMAIN_ORDER) or index['updates'] != 16:
        raise ValueError('stream protocol mismatch')
    for seed in SEEDS:
        batches = index['seeds'][str(seed)]['updates']
        if len(batches) != 16:
            raise ValueError(f'missing update for {seed}')
        for n, row in enumerate(batches, 1):
            if row['update'] != n or row['sequence_indices'] != list(range(4 * (n - 1), 4 * n)) or [x['domain'] for x in row['sequences']] != list(DOMAIN_ORDER) or any((x['nonpadding_tokens'] != 256 for x in row['sequences'])) or (row['cumulative_nonpadding_tokens'] != 1024 * n):
                raise ValueError(f'bad stream update {seed}/{n}')
    return index

def require_science_authorization(path: Path | None, repo: Path, prereg_path: Path) -> None:
    """A future human authorization must name this clean frozen runner and prereg."""
    if path is None:
        raise PermissionError('J1-C science requires a separate authorization after runner freeze')
    a = json.loads(path.read_text())
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    needed = {'j1_c_science_authorized': True, 'runner_commit': commit, 'preregistration_sha256': sha(prereg_path), 'original_j1_freeze_sha256': ORIGINAL_J1_FREEZE_SHA256, 'amendment_sha256': AMENDMENT_SHA256, 'training_stream_index_sha256': STREAM_INDEX_SHA256}
    if dirty or any((a.get(k) != v for k, v in needed.items())):
        raise PermissionError('science authorization or frozen runner mismatch')

def set_arm(moe: CoTrainMoE, arm: str) -> list[torch.nn.Parameter]:
    if arm not in ARMS:
        raise ValueError('unsupported arm')
    selected = []
    for name, parameter in moe.named_parameters():
        enabled = name == 'router.weight' and arm in ('R', 'J') or (name.startswith('experts.') and arm in ('E', 'J'))
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(parameter)
    expected = (1 if arm in ('R', 'J') else 0) + (3 * 4 * moe.g if arm in ('E', 'J') else 0)
    if len(selected) != expected:
        raise AssertionError('trainable parameter block mismatch')
    return selected

def sgd_update_lm(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: dict) -> float:
    ids = batch['input_ids']
    mask = batch['attention_mask']
    if ids.shape != (4, 256) or mask.shape != ids.shape or (not bool(mask.all())):
        raise ValueError('expected four complete 256-token sequences')
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, attention_mask=mask, labels=ids).loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite training loss')
    loss.backward()
    optimizer.step()
    return float(loss.detach())

def fixed_bank_qk(moe: CoTrainMoE, hidden: np.ndarray, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preserve the confirmed J1 router and expert geometry implementation."""
    wr = moe.router.weight.detach().cpu().double().numpy()
    gate = np.stack([e.gate_proj.weight.detach().cpu().double().numpy() for e in moe.experts])
    up = np.stack([e.up_proj.weight.detach().cpu().double().numpy() for e in moe.experts])
    q = router_q(wr, hidden[src], hidden[tgt])
    k = geometry(features(gate, up, hidden), src, tgt)
    return (q, k)

def serialize_response(q: np.ndarray, k: np.ndarray, group: np.ndarray, g: int) -> dict:
    b = response(q, k, group, g)
    return {name: values.tolist() for name, values in b.items()}

def factorized_from_saved(q_r: np.ndarray, k_e: np.ndarray, b_j: dict, group: np.ndarray, g: int) -> dict:
    composite = serialize_response(q_r, k_e, group, g)
    residual = {name: (np.asarray(b_j[name]) - np.asarray(composite[name])).tolist() for name in ('A', 'C', 'M', 'total')}
    return {'B_oplus': composite, 'C_co': residual}

def science_run(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent.parent
    require_science_authorization(args.authorization, repo, args.preregistration)
    index = validate_authorities(args.original_freeze, args.amendment, args.stream_index)
    prereg = json.loads(args.preregistration.read_text())
    if prereg['original_j1_freeze_sha256'] != ORIGINAL_J1_FREEZE_SHA256 or prereg['amendment_sha256'] != AMENDMENT_SHA256:
        raise ValueError('effective preregistration authority mismatch')
    if sha(repo / 'protocol/J1_C_OUTPUT_SCHEMA_v1_1.json') != prereg['output_schema_sha256']:
        raise ValueError('frozen output schema changed')
    for rel, expected in prereg['source_sha256'].items():
        if sha(repo / rel) != expected:
            raise ValueError(f'frozen source changed: {rel}')
    if sha(args.batch_index) != prereg['training_batch_hash_index_sha256']:
        raise ValueError('frozen training batch hash index changed')
    batch_index = json.loads(args.batch_index.read_text())
    if batch_index['training_stream_index_sha256'] != STREAM_INDEX_SHA256:
        raise ValueError('training batch index stream anchor mismatch')
    if sha(args.formal_tokens) != prereg['response_bank']['audit_token_tensor_sha256'] or sha(args.formal_hidden) != prereg['response_bank']['audit_hidden_tensor_sha256']:
        raise ValueError('formal bank hash mismatch')
    if sha(args.formal_documents) != prereg['response_bank']['audit_document_manifest_sha256'] or sha(args.formal_pairs) != prereg['response_bank']['audit_pair_manifest_sha256']:
        raise ValueError('formal bank manifest mismatch')
    if sha(args.model_dir / 'model.safetensors') != prereg['primary_model']['model_safetensors_sha256']:
        raise ValueError('frozen model asset hash mismatch')
    src, tgt, group = pairs(args.formal_documents, args.formal_pairs)
    hidden_tensor = torch.load(args.formal_hidden, map_location='cpu', weights_only=True)['h']
    if tensor_sha(hidden_tensor) != prereg['response_bank']['audit_hidden_value_sha256']:
        raise ValueError('formal fixed hidden values changed')
    hidden = hidden_tensor.double().numpy()
    if hidden.shape != (2048, 576):
        raise ValueError('formal hidden bank shape mismatch')
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), dtype=torch.float32, local_files_only=True).eval()
    model.requires_grad_(False)
    dense = model.model.layers[19].mlp
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / 'group.npy', group)
    run_manifest = {'original_j1_freeze_sha256': ORIGINAL_J1_FREEZE_SHA256, 'amendment_sha256': AMENDMENT_SHA256, 'preregistration_sha256': sha(args.preregistration), 'training_stream_index_sha256': STREAM_INDEX_SHA256, 'training_batch_hash_index_sha256': sha(args.batch_index), 'model_safetensors_sha256': sha(args.model_dir / 'model.safetensors'), 'formal_token_tensor_sha256': sha(args.formal_tokens), 'formal_hidden_tensor_sha256': sha(args.formal_hidden), 'formal_document_manifest_sha256': sha(args.formal_documents), 'formal_pair_manifest_sha256': sha(args.formal_pairs), 'group_file_sha256': sha(args.output / 'group.npy'), 'runner_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(), 'checkpoint_updates': list(CHECKPOINT_UPDATES), 'checkpoint_tokens': list(CHECKPOINT_TOKENS)}
    (args.output / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + '\n')
    for g in GRANULARITIES:
        for seed in SEEDS:
            moe = CoTrainMoE(dense, g, seed)
            model.model.layers[19].mlp = moe
            initial = {k: v.detach().clone() for k, v in moe.state_dict().items()}
            initial_frozen_model_sha = frozen_model_sha(model)
            row = index['seeds'][str(seed)]
            expected_batch_hashes = batch_index['seeds'][str(seed)]['batch_input_ids_sha256']
            stream_path = args.stream_dir / f'future_training_seed_{seed}_tokens.pt'
            if sha(stream_path) != row['token_tensor_sha256']:
                raise ValueError(f'science stream file hash mismatch: {seed}')
            stream = torch.load(stream_path, map_location='cpu', weights_only=True)
            if stream['input_ids'].shape != (1000, 256) or not bool(stream['attention_mask'][:64].all()):
                raise ValueError('science stream shape/completeness mismatch')
            run_dir = args.output / f'g{g}_seed{seed}'
            run_dir.mkdir()
            arm_results = {}
            for arm in ARMS:
                moe.load_state_dict(initial)
                params = set_arm(moe, arm)
                optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)
                if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]['lr'] != 0.01:
                    raise AssertionError('optimizer protocol mismatch')
                records = []
                losses = []
                batch_hashes = []
                for n in range(17):
                    if n in CHECKPOINT_UPDATES:
                        q, k = fixed_bank_qk(moe, hidden, src, tgt)
                        np.save(run_dir / f'{arm}_u{n}_q.npy', q)
                        np.save(run_dir / f'{arm}_u{n}_kappa.npy', k)
                        if n in (0, 1):
                            save_target_parameters(moe, run_dir / f'{arm}_u{n}_params.npz')
                        records.append({'arm': arm, 'g': g, 'seed': seed, 'update': n, 'nonpadding_tokens': n * 1024, 'B': serialize_response(q, k, group, g), 'q_sha256': tensor_sha(q), 'kappa_sha256': tensor_sha(k), 'param_file_sha256': sha(run_dir / f'{arm}_u{n}_params.npz') if n in (0, 1) else None, 'router_sha256': tensor_sha(moe.router.weight), 'expert_sha256': [tensor_sha(p) for name, p in moe.named_parameters() if name.startswith('experts.')], 'frozen_model_sha256': frozen_model_sha(model) if n == 16 else initial_frozen_model_sha})
                    if n == 16:
                        break
                    i = row['updates'][n]['sequence_indices']
                    batch = {key: value[i] for key, value in stream.items() if key in ('input_ids', 'attention_mask')}
                    batch_hash = tensor_sha(batch['input_ids'])
                    if batch_hash != expected_batch_hashes[n]:
                        raise ValueError(f'inherited batch content mismatch: {seed}/{n}')
                    batch_hashes.append(batch_hash)
                    loss = sgd_update_lm(model, optimizer, batch)
                    losses.append(loss)
                arm_results[arm] = {'checkpoints': records, 'batch_sha256': batch_hashes, 'loss_per_update': losses, 'initial_target_state_sha256': tensor_sha(torch.cat([v.flatten() for v in initial.values()])), 'initial_frozen_model_sha256': initial_frozen_model_sha}
            for n in CHECKPOINT_UPDATES:
                q_r = np.load(run_dir / f'R_u{n}_q.npy', allow_pickle=False)
                k_e = np.load(run_dir / f'E_u{n}_kappa.npy', allow_pickle=False)
                j = next((r for r in arm_results['J']['checkpoints'] if r['update'] == n))
                j.update(factorized_from_saved(q_r, k_e, j['B'], group, g))
            (run_dir / 'responses.json').write_text(json.dumps({'original_j1_freeze_sha256': ORIGINAL_J1_FREEZE_SHA256, 'amendment_sha256': AMENDMENT_SHA256, 'stream_index_sha256': STREAM_INDEX_SHA256, 'preregistration_sha256': sha(args.preregistration), 'source_token_tensor_sha256': row['token_tensor_sha256'], 'g': g, 'seed': seed, 'arms': arm_results, 'C0_C7': 'PENDING_INDEPENDENT_CHECKER'}, sort_keys=True) + '\n')
            model.model.layers[19].mlp = dense

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'science'), required=True)
    for name in ('authorization', 'preregistration', 'original_freeze', 'amendment', 'stream_index', 'batch_index', 'formal_tokens', 'formal_hidden', 'formal_documents', 'formal_pairs', 'model_dir', 'stream_dir', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path)
    args = parser.parse_args()
    if args.mode == 'preflight':
        raise SystemExit('Run source/j1c_preflight.py for synthetic and excluded-bank preflight')
    science_run(args)
if __name__ == '__main__':
    main()
