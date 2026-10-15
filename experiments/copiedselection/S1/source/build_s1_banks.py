# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import heapq
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import zstandard as zstd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path(_p1_path('path_33f6c02e7543'))
OLD = PRIOR / 'outputs/EXP_G_V3_LAPTOP_PREFLIGHT'
DB = PRIOR / 'work/v3_train_unique.sqlite'
DATA = Path(_p1_path('path_8c5dea5cdb13'))
OUT = ROOT / 'outputs'
WORK = ROOT / 'work'
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')

def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def line(obj: dict) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()

def main():
    OUT.mkdir(exist_ok=True)
    old_hashes = set()
    old_manifest_hashes = {}
    for name in ('calibration', 'audit'):
        p = OLD / f'{name}_bank_manifest.jsonl'
        old_manifest_hashes[name] = sha(p.read_bytes())
        old_hashes.update((json.loads(x)['text_sha256'] for x in p.open()))
    assert len(old_hashes) == 4096
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    assert con.execute('select count(*),count(distinct text_sha) from global_docs').fetchone() == (5488984, 5488984)
    chosen = {'calibration': [], 'audit': []}
    benchmark = []
    selected_hashes = set()
    eligible_counts = {}
    for domain in DOMAINS:
        ranked = []
        for h, shard, ordinal, original_index in con.execute('select text_sha,shard,row_ordinal,original_index from global_docs where domain=?', (domain,)):
            if h in old_hashes:
                continue
            rank = sha(f'EXP-S1-v1-bank|{domain}|{h}'.encode('ascii'))
            ranked.append((rank, h, shard, ordinal, original_index))
        ranked.sort()
        eligible_counts[domain] = len(ranked)
        assert len(ranked) >= 1024
        for membership, part in (('calibration', ranked[:512]), ('audit', ranked[512:1024])):
            for rank, h, shard, ordinal, original_index in part:
                chosen[membership].append({'membership': membership, 'domain': domain, 'train_row_id': f'train-{shard:05d}:row-{ordinal}', 'shard': shard, 'row_ordinal': ordinal, 'original_index': original_index, 'text_sha256': h, 'rank_key': rank})
                selected_hashes.add(h)
        candidates = ((sha(f'EXP-S1-v1-benchmark|2026100198|{domain}|{h}'.encode('ascii')), h, shard, ordinal, original_index) for _, h, shard, ordinal, original_index in ranked[1024:])
        for rank, h, shard, ordinal, original_index in heapq.nsmallest(500, candidates):
            benchmark.append({'domain': domain, 'train_row_id': f'train-{shard:05d}:row-{ordinal}', 'shard': shard, 'row_ordinal': ordinal, 'text_sha256': h, 'rank_key': rank})
    assert len(selected_hashes) == 4096 and (not selected_hashes & old_hashes)
    wanted = defaultdict(dict)
    for rows in chosen.values():
        for row in rows:
            wanted[row['shard']][row['row_ordinal']] = row['text_sha256']
    for row in benchmark:
        wanted[row['shard']][row['row_ordinal']] = row['text_sha256']
    texts = {}
    for shard, ordinals in sorted(wanted.items()):
        p = next(DATA.glob(f'train-{shard:05d}-of-00048-*.parquet'))
        pf = pq.ParquetFile(p)
        offset = 0
        for rg in range(pf.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            local = [(ordinal, h) for ordinal, h in ordinals.items() if offset <= ordinal < offset + size]
            if local:
                column = pf.read_row_group(rg, columns=['text']).column(0)
                for ordinal, h in local:
                    content = column[ordinal - offset].as_py()
                    assert sha(content.encode('utf-8')) == h
                    texts[h] = content
            offset += size
        print('read shard', shard, flush=True)
    assert len(texts) == len(set((h for d in wanted.values() for h in d.values())))
    tokenizer = AutoTokenizer.from_pretrained(str(WORK / 'model'), local_files_only=True)
    eos = tokenizer.eos_token_id
    manifest_hashes = {}
    gate_ids, gate_mask = ([], [])
    for membership in ('calibration', 'audit'):
        p = OUT / f'{membership}_bank_manifest.jsonl'
        with p.open('wb') as f:
            for i, row in enumerate(chosen[membership]):
                ids = tokenizer(texts[row['text_sha256']], add_special_tokens=False, truncation=True, max_length=256)['input_ids']
                if not ids:
                    ids = [eos]
                mask = [1] * len(ids) + [0] * (256 - len(ids))
                padded = ids + [eos] * (256 - len(ids))
                row['tokenized_sequence_sha256'] = sha(json.dumps({'input_ids': padded, 'attention_mask': mask}, sort_keys=True, separators=(',', ':')).encode())
                row['nonpadding_tokens'] = len(ids)
                f.write(line(row))
                if membership == 'calibration' and i % 512 < 64:
                    gate_ids.append(padded)
                    gate_mask.append(mask)
        manifest_hashes[membership] = sha(p.read_bytes())
    assert len(gate_ids) == 256
    torch.save({'input_ids': torch.tensor(gate_ids), 'attention_mask': torch.tensor(gate_mask)}, WORK / 's1_calibration_gate_256.pt')
    for membership, seed in (('calibration', 2026100191), ('audit', 2026100192)):
        by_domain = {d: [r for r in chosen[membership] if r['domain'] == d] for d in DOMAINS}
        rng = np.random.default_rng(seed)
        p = OUT / f'{membership}_pair_manifest.jsonl'
        with p.open('wb') as f:
            for source in DOMAINS:
                for target in DOMAINS:
                    if source == target:
                        continue
                    positions = rng.choice(512 * 512, size=1024, replace=False)
                    for index, pos in enumerate(positions):
                        a, b = (by_domain[source][int(pos) // 512], by_domain[target][int(pos) % 512])
                        f.write(line({'membership': membership, 'source_domain': source, 'target_domain': target, 'pair_index': index, 'source_train_row_id': a['train_row_id'], 'target_train_row_id': b['train_row_id'], 'source_text_sha256': a['text_sha256'], 'target_text_sha256': b['text_sha256']}))
        manifest_hashes[membership + '_pairs'] = sha(p.read_bytes())
    p = OUT / 'training_pool_manifest.jsonl.zst'
    content_hash = hashlib.sha256()
    pool_rows = 0
    with p.open('wb') as raw, zstd.ZstdCompressor(level=6).stream_writer(raw) as writer:
        for domain in DOMAINS:
            for h, shard, ordinal, original_index in con.execute('select text_sha,shard,row_ordinal,original_index from global_docs where domain=? order by text_sha', (domain,)):
                if h in selected_hashes or h in old_hashes:
                    continue
                payload = line({'domain': domain, 'train_row_id': f'train-{shard:05d}:row-{ordinal}', 'original_index': original_index, 'text_sha256': h})
                writer.write(payload)
                content_hash.update(payload)
                pool_rows += 1
    manifest_hashes['training_pool_content'] = content_hash.hexdigest()
    manifest_hashes['training_pool_compressed'] = sha(p.read_bytes())
    seqs = {}
    bench_rows = []
    for domain in DOMAINS:
        stream = []
        for row in sorted((x for x in benchmark if x['domain'] == domain), key=lambda x: x['rank_key']):
            ids = tokenizer(texts[row['text_sha256']], add_special_tokens=False)['input_ids']
            if not ids:
                continue
            stream.extend(ids)
            stream.append(eos)
            bench_rows.append(row)
            if len(stream) >= 25000:
                break
        assert len(stream) >= 25000
        stream = stream[:25000]
        seqs[domain] = [(stream[i:i + 256] + [eos] * max(0, 256 - len(stream[i:i + 256])), [1] * len(stream[i:i + 256]) + [0] * max(0, 256 - len(stream[i:i + 256]))) for i in range(0, 25000, 256)]
    ids, masks = ([], [])
    for i in range(98):
        for domain in DOMAINS:
            a, b = seqs[domain][i]
            ids.append(a)
            masks.append(b)
    benchmark_tensor = {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks)}
    assert int(benchmark_tensor['attention_mask'].sum()) == 100000
    torch.save(benchmark_tensor, WORK / 's1_benchmark_100k.pt')
    p = OUT / 'benchmark_source_manifest.jsonl'
    p.write_bytes(b''.join((line(row) for row in bench_rows)))
    manifest_hashes['benchmark_source'] = sha(p.read_bytes())
    result = {'status': 'BANKS_FROZEN', 'dataset_revision': 'b5f90f419b7489cdba26fdbc8c022fcb5562f968', 'old_bank_manifest_sha256': old_manifest_hashes, 'old_holdout_hashes': len(old_hashes), 'eligible_unique_docs_per_domain': eligible_counts, 'calibration_rows': 2048, 'audit_rows': 2048, 'calibration_audit_old_text_hash_overlap': 0, 'training_pool_rows': pool_rows, 'training_pool_holdout_hash_overlap': 0, 'gate_sequences': 256, 'benchmark_training_tokens': 100000, 'benchmark_seed': 2026100198, 'pair_seeds': {'calibration': 2026100191, 'audit': 2026100192}, 'manifest_sha256': manifest_hashes}
    (OUT / 'DATA_MANIFEST_HASHES.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
