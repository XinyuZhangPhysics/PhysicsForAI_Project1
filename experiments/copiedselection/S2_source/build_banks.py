# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Freeze document-ID partitions; materialize only the preflight 100k stream.

No scientific response statistic is computed. The historical global-doc index
is read-only and every selected text is checked against the pinned parquet.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import argparse
import hashlib
import heapq
import json
import sqlite3
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
from s2_impl import DATASET_ID, DATASET_REVISION, MODEL_REVISION
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
N_CAL, N_AUD, N_BENCH, N_TRAIN = (512, 512, 512, 300)
SEEDS = tuple(range(2026110201, 2026110209))
SALT = 'EXP-S2-v1-fresh-banks|'

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def filehash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    return filehash(path)

def old_banks(paths: list[Path]) -> tuple[set[str], dict[str, str]]:
    hashes: set[str] = set()
    manifests = {}
    for path in paths:
        manifests[str(path)] = filehash(path)
        for line in path.open():
            row = json.loads(line)
            hashes.add(row['text_sha256'])
    return (hashes, manifests)

def read_selected_texts(data_dir: Path, rows: list[dict]) -> dict[str, str]:
    wanted: dict[int, dict[int, str]] = {}
    for r in rows:
        wanted.setdefault(r['shard'], {})[r['row_ordinal']] = r['text_sha256']
    texts = {}
    for shard, ordinals in sorted(wanted.items()):
        matches = list(data_dir.glob(f'train-{shard:05d}-of-00048-*.parquet'))
        assert len(matches) == 1
        pf = pq.ParquetFile(matches[0])
        offset = 0
        for rg in range(pf.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            local = [(ordinal, h) for ordinal, h in ordinals.items() if offset <= ordinal < offset + size]
            if local:
                col = pf.read_row_group(rg, columns=['text']).column(0)
                for ordinal, h in local:
                    text = col[ordinal - offset].as_py()
                    assert digest(text.encode('utf-8')) == h
                    texts[h] = text
            offset += size
    assert len(texts) == len({r['text_sha256'] for r in rows})
    return texts

def main(args: argparse.Namespace) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    old, old_manifest_sha = old_banks(args.old_bank)
    con = sqlite3.connect(f'file:{args.index}?mode=ro', uri=True)
    picks = {'future_calibration': [], 'future_audit': [], 'preflight_benchmark': [], 'future_training': []}
    training_by_seed = {seed: [] for seed in SEEDS}
    training_union = {}
    eligible = {}
    for domain in DOMAINS:
        ranked = []
        for h, shard, ordinal, original in con.execute('select text_sha,shard,row_ordinal,original_index from global_docs where domain=?', (domain,)):
            if h in old:
                continue
            rank = digest(f'{SALT}{domain}|{h}'.encode())
            ranked.append((rank, h, shard, ordinal, original))
        ranked.sort()
        eligible[domain] = len(ranked)
        assert len(ranked) >= N_CAL + N_AUD + N_BENCH + N_TRAIN
        offsets = (0, N_CAL, N_CAL + N_AUD, N_CAL + N_AUD + N_BENCH)
        for membership, first, last in zip(('future_calibration', 'future_audit', 'preflight_benchmark'), offsets[:-1], offsets[1:]):
            for rank, h, shard, ordinal, original in ranked[first:last]:
                picks[membership].append({'bank': membership, 'domain': domain, 'train_row_id': f'train-{shard:05d}:row-{ordinal}', 'shard': shard, 'row_ordinal': ordinal, 'original_index': original, 'text_sha256': h, 'rank_key': rank})
        candidates = ranked[offsets[-1]:]
        for seed in SEEDS:
            chosen = heapq.nsmallest(N_TRAIN, ((digest(f'EXP-S2-v1-train|{seed}|{domain}|{h}'.encode()), rank, h, shard, ordinal, original) for rank, h, shard, ordinal, original in candidates))
            for order, (training_rank, rank, h, shard, ordinal, original) in enumerate(chosen):
                row = {'bank': 'future_training', 'domain': domain, 'train_row_id': f'train-{shard:05d}:row-{ordinal}', 'shard': shard, 'row_ordinal': ordinal, 'original_index': original, 'text_sha256': h, 'rank_key': rank}
                training_union[h] = row
                training_by_seed[seed].append({'seed': seed, 'domain': domain, 'order': order, 'train_row_id': row['train_row_id'], 'text_sha256': h, 'selection_rank_key': training_rank})
    picks['future_training'] = sorted(training_union.values(), key=lambda r: (DOMAINS.index(r['domain']), r['text_sha256']))
    sets = {k: {r['text_sha256'] for r in rows} for k, rows in picks.items()}
    assert all((len(v) == len(picks[k]) for k, v in sets.items()))
    assert not set().union(*sets.values()).intersection(old)
    assert all((not sets[a].intersection(sets[b]) for a in sets for b in sets if a != b))
    manifests = {}
    for membership, rows in picks.items():
        manifests[membership] = write_jsonl(out / f'{membership}_document_ids.jsonl', rows)
    pair_seeds = {'future_calibration': 2026110291, 'future_audit': 2026110292}
    for membership, seed in pair_seeds.items():
        by_domain = {d: [r for r in picks[membership] if r['domain'] == d] for d in DOMAINS}
        rng = np.random.default_rng(seed)
        rows = []
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue
                for i, pos in enumerate(rng.choice(512 * 512, 1024, replace=False)):
                    a = by_domain[source][int(pos) // 512]
                    b = by_domain[target][int(pos) % 512]
                    rows.append({'bank': membership, 'source_domain': source, 'target_domain': target, 'pair_index': i, 'source_train_row_id': a['train_row_id'], 'target_train_row_id': b['train_row_id']})
        manifests[membership + '_pairs'] = write_jsonl(out / f'{membership}_pair_ids.jsonl', rows)
    for seed in SEEDS:
        rows = training_by_seed[seed]
        assert len(rows) == 4 * N_TRAIN
        manifests[f'future_training_seed_{seed}'] = write_jsonl(out / f'future_training_seed_{seed}.jsonl', rows)
    texts = read_selected_texts(args.data, picks['preflight_benchmark'])
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    eos = tok.eos_token_id
    by_domain = {d: sorted((r for r in picks['preflight_benchmark'] if r['domain'] == d), key=lambda r: r['rank_key']) for d in DOMAINS}
    streams, consumed_ids = ({}, [])
    for domain in DOMAINS:
        stream = []
        for row in by_domain[domain]:
            ids = tok(texts[row['text_sha256']], add_special_tokens=False)['input_ids']
            if not ids:
                continue
            stream.extend(ids)
            stream.append(eos)
            consumed_ids.append(row['train_row_id'])
            if len(stream) >= 25000:
                break
        assert len(stream) >= 25000, domain
        streams[domain] = stream[:25000]
    blocks = {}
    for domain, stream in streams.items():
        blocks[domain] = [(stream[i:i + 256] + [eos] * (256 - len(stream[i:i + 256])), [1] * len(stream[i:i + 256]) + [0] * (256 - len(stream[i:i + 256]))) for i in range(0, 25000, 256)]
        assert len(blocks[domain]) == 98
    ids, masks = ([], [])
    for i in range(98):
        for domain in DOMAINS:
            a, b = blocks[domain][i]
            ids.append(a)
            masks.append(b)
    tensor = {'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}
    assert int(tensor['attention_mask'].sum()) == 100000
    tensor_path = out / 'preflight_benchmark_100k.pt'
    torch.save(tensor, tensor_path)
    manifests['preflight_benchmark_tensor'] = filehash(tensor_path)
    checks = {'calibration_audit_overlap': 0, 'preflight_future_overlap': 0, 'future_banks_prior_holdout_overlap': 0, 'prior_holdout_hash_count': len(old), 'all_partitions_pairwise_disjoint': True}
    result = {'status': 'BANK_IDS_FROZEN_PRE_SCIENCE', 'dataset_id': DATASET_ID, 'dataset_revision': DATASET_REVISION, 'tokenizer_model_revision': MODEL_REVISION, 'salt': SALT, 'pair_seeds': pair_seeds, 'training_seeds': SEEDS, 'counts': {k: len(v) for k, v in picks.items()}, 'eligible_counts': eligible, 'checks': checks, 'historical_bank_manifest_sha256': old_manifest_sha, 'manifest_sha256': manifests, 'preflight_benchmark_tokens': 100000, 'preflight_benchmark_source_docs_used': len(set(consumed_ids)), 'scientific_response_observables_computed': False}
    (out / 'BANKS_INDEX.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'benchmark_tokens': 100000, 'checks': checks}))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--old-bank', type=Path, action='append', required=True)
    main(p.parse_args())
