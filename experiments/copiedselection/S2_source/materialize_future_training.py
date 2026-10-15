# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Freeze prospective S2 token streams without running any training."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from build_banks import DOMAINS, SEEDS, filehash, read_selected_texts, write_jsonl

def sequence_hash(ids: list[int]) -> str:
    import hashlib
    payload = json.dumps({'input_ids': ids, 'attention_mask': [1] * 256}, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()

def main(bank_dir: Path, data_dir: Path, model_dir: Path) -> None:
    start = time.perf_counter()
    index = json.loads((bank_dir / 'BANKS_INDEX.json').read_text())
    union_path = bank_dir / 'future_training_document_ids.jsonl'
    assert filehash(union_path) == index['manifest_sha256']['future_training']
    union = [json.loads(line) for line in union_path.open()]
    texts = read_selected_texts(data_dir, union)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer.model_max_length = 10 ** 9
    eos = tokenizer.eos_token_id
    tokens = {h: tokenizer(text, add_special_tokens=False)['input_ids'] for h, text in texts.items()}
    summary = {'status': 'FUTURE_SCIENTIFIC_STREAMS_FROZEN_PRE_SCIENCE', 'selection': '300 per domain and seed by SHA256(EXP-S2-v1-train|seed|domain|text_sha256) ascending', 'model_revision': index['tokenizer_model_revision'], 'dataset_revision': index['dataset_revision'], 'source_union_manifest_sha256': filehash(union_path), 'sequence_length': 256, 'tokens_per_domain_per_seed': 64000, 'nonpadding_tokens_per_seed': 256000, 'domain_interleave_order': DOMAINS, 'seeds': {}}
    for seed in SEEDS:
        seed_path = bank_dir / f'future_training_seed_{seed}.jsonl'
        assert filehash(seed_path) == index['manifest_sha256'][f'future_training_seed_{seed}']
        selected = [json.loads(line) for line in seed_path.open()]
        assert len(selected) == 1200
        sequences = {}
        for domain in DOMAINS:
            stream, spans = ([], [])
            rows = [r for r in selected if r['domain'] == domain]
            assert len(rows) == 300 and [r['order'] for r in rows] == list(range(300))
            for row in rows:
                ids = tokens[row['text_sha256']]
                if not ids:
                    continue
                begin = len(stream)
                stream.extend(ids)
                stream.append(eos)
                spans.append((begin, len(stream), row))
                if len(stream) >= 64000:
                    break
            assert len(stream) >= 64000, (seed, domain, len(stream))
            stream = stream[:64000]
            sequences[domain] = []
            for pos in range(0, 64000, 256):
                chunk = stream[pos:pos + 256]
                docs = [{'train_row_id': row['train_row_id'], 'text_sha256': row['text_sha256'], 'selection_rank_key': row['selection_rank_key']} for a, b, row in spans if a < pos + 256 and b > pos]
                sequences[domain].append((chunk, docs))
        ids, manifest = ([], [])
        for i in range(250):
            for domain in DOMAINS:
                chunk, docs = sequences[domain][i]
                assert len(chunk) == 256
                ids.append(chunk)
                manifest.append({'seed': seed, 'sequence_index': len(ids) - 1, 'domain': domain, 'nonpadding_tokens': 256, 'sequence_sha256': sequence_hash(chunk), 'source_documents': docs})
        seq_manifest = bank_dir / f'future_training_seed_{seed}_sequences.jsonl'
        manifest_sha = write_jsonl(seq_manifest, manifest)
        tensor_path = bank_dir / f'future_training_seed_{seed}_tokens.pt'
        torch.save({'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.ones((1000, 256), dtype=torch.long)}, tensor_path)
        summary['seeds'][str(seed)] = {'document_ids_sha256': filehash(seed_path), 'sequence_manifest_sha256': manifest_sha, 'token_tensor_sha256': filehash(tensor_path), 'sequences': 1000, 'nonpadding_tokens': 256000}
    summary['elapsed_seconds'] = time.perf_counter() - start
    summary['scientific_training_started'] = False
    (bank_dir / 'TRAINING_STREAMS.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'seeds': len(summary['seeds']), 'elapsed_seconds': summary['elapsed_seconds']}))
if __name__ == '__main__':
    main(*(Path(p) for p in sys.argv[1:4]))
