# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import heapq
import io
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import torch
import zstandard as zstd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
WORK = ROOT / 'work'
DATA = Path(_p1_path('path_8c5dea5cdb13'))
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')
SEEDS = tuple(range(2026100101, 2026100109))
CANDIDATES = 300

def sha(b):
    return hashlib.sha256(b).hexdigest()

def canon(r):
    return (json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n').encode()

def main():
    pool = OUT / 'training_pool_manifest.jsonl.zst'
    data_hashes = json.loads((OUT / 'DATA_MANIFEST_HASHES.json').read_text())
    assert sha(pool.read_bytes()) == data_hashes['manifest_sha256']['training_pool_compressed']
    selected = {seed: {d: [] for d in DOMAINS} for seed in SEEDS}
    with pool.open('rb') as raw, zstd.ZstdDecompressor().stream_reader(raw) as stream:
        for line in io.TextIOWrapper(stream, encoding='utf-8'):
            row = json.loads(line)
            d = row['domain']
            h = row['text_sha256']
            for seed in SEEDS:
                rank = sha(f'EXP-S1-v1-train|{seed}|{d}|{h}'.encode('ascii'))
                item = (-int(rank, 16), rank, row)
                heap = selected[seed][d]
                if len(heap) < CANDIDATES:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
    for seed in SEEDS:
        for d in DOMAINS:
            assert len(selected[seed][d]) == CANDIDATES
            selected[seed][d] = [(rank, row) for _, rank, row in sorted(selected[seed][d], key=lambda x: x[1])]
    wanted = defaultdict(dict)
    for seed in SEEDS:
        for d in DOMAINS:
            for rank, row in selected[seed][d]:
                shard, ordinal = row['train_row_id'].removeprefix('train-').split(':row-')
                wanted[int(shard)][int(ordinal)] = row['text_sha256']
    texts = {}
    for shard, ordinals in sorted(wanted.items()):
        p = next(DATA.glob(f'train-{shard:05d}-of-00048-*.parquet'))
        pf = pq.ParquetFile(p)
        offset = 0
        for rg in range(pf.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            local = [(o, h) for o, h in ordinals.items() if offset <= o < offset + size]
            if local:
                col = pf.read_row_group(rg, columns=['text']).column(0)
                for ordinal, h in local:
                    value = col[ordinal - offset].as_py()
                    assert sha(value.encode('utf-8')) == h
                    texts[h] = value
            offset += size
        print('retrieved shard', shard, flush=True)
    assert len(texts) == len({h for vals in wanted.values() for h in vals.values()})
    tok = AutoTokenizer.from_pretrained(str(WORK / 'model'), local_files_only=True)
    eos = tok.eos_token_id
    tokenized = {h: tok(value, add_special_tokens=False)['input_ids'] for h, value in texts.items()}
    summary = {'status': 'FROZEN', 'selection': 'SHA256(EXP-S1-v1-train|seed|domain|text_sha256) ascending on eligible globally unique train pool', 'model_revision': 'a39f36b100fe8a5377810d56c3f4789b9c53ac42', 'dataset_revision': 'b5f90f419b7489cdba26fdbc8c022fcb5562f968', 'pool_sha256': sha(pool.read_bytes()), 'sequence_length': 256, 'tokens_per_domain_per_seed': 64000, 'nonpadding_tokens_per_seed': 256000, 'domain_interleave_order': DOMAINS, 'seeds': {}}
    for seed in SEEDS:
        domain_sequences = {}
        for d in DOMAINS:
            stream = []
            spans = []
            for rank, row in selected[seed][d]:
                h = row['text_sha256']
                ids = tokenized[h]
                if not ids:
                    continue
                start = len(stream)
                stream.extend(ids)
                stream.append(eos)
                spans.append((start, len(stream), h, row['train_row_id'], rank))
                if len(stream) >= 64000:
                    break
            assert len(stream) >= 64000, (seed, d, len(stream))
            stream = stream[:64000]
            domain_sequences[d] = []
            for start in range(0, 64000, 256):
                chunk = stream[start:start + 256]
                docs = [{'train_row_id': rid, 'text_sha256': h, 'selection_rank_key': rank} for a, b, h, rid, rank in spans if a < start + 256 and b > start]
                domain_sequences[d].append((chunk, docs))
        ids = []
        masks = []
        path = OUT / f'TRAIN_SEED_{seed}_MANIFEST.jsonl'
        with path.open('wb') as f:
            for index in range(250):
                for d in DOMAINS:
                    chunk, docs = domain_sequences[d][index]
                    assert len(chunk) == 256
                    ids.append(chunk)
                    masks.append([1] * 256)
                    sequence_sha = sha(json.dumps({'input_ids': chunk, 'attention_mask': [1] * 256}, sort_keys=True, separators=(',', ':')).encode())
                    f.write(canon({'seed': seed, 'sequence_index': len(ids) - 1, 'domain': d, 'nonpadding_tokens': 256, 'sequence_sha256': sequence_sha, 'source_documents': docs}))
        tensor = {'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}
        tensor_path = WORK / f'train_seed_{seed}.pt'
        torch.save(tensor, tensor_path)
        assert int(tensor['attention_mask'].sum()) == 256000
        summary['seeds'][str(seed)] = {'manifest_sha256': sha(path.read_bytes()), 'tensor_sha256': sha(tensor_path.read_bytes()), 'sequences': 1000, 'nonpadding_tokens': 256000}
        print('frozen seed', seed, flush=True)
    (OUT / 'TRAINING_MANIFEST_HASHES.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('TRAINING_STREAMS_FROZEN', flush=True)
if __name__ == '__main__':
    main()
