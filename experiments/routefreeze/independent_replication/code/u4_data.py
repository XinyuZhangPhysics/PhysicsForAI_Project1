# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Deterministic source/target sample construction — preregistration Section 6.

Every rule below is taken verbatim from the frozen preregistration:

1. h_text = SHA256(UTF8(text)) for every validation/test row;
2. remove every row whose h_text occurs in both validation and test
   (an h_text that occurs in both splits disqualifies *all* rows carrying it — declared
   reading of "every row whose h_text occurs in both validation and test", see
   replication_config.json -> uncertain_or_undocumented_details);
3. k_row = SHA256(split || 0x00 || domain || 0x00 || str(index) || 0x00 || text);
4. sort rows lexicographically by k_row;
5. tokenize with the pinned local OLMoE tokenizer, add_special_tokens=False;
6. retain rows with >= 128 token ids;
7. take the first 16 eligible rows, first 128 token ids.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from typing import Dict, List
from u4_common import PATHS, log, sha256_text_utf8
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']
SPLITS = ['validation', 'test']
PARQUET = {'validation': 'validation-00000-of-00001-4fb685c22a3f91ef.parquet', 'test': 'test-00000-of-00001-9f769cf7ce219017.parquet'}

def _sep() -> bytes:
    return b'\x00'

def k_row(split: str, domain: str, index, text: str) -> str:
    payload = split.encode('utf-8') + _sep() + domain.encode('utf-8') + _sep() + str(index).encode('utf-8') + _sep() + text.encode('utf-8')
    return __import__('hashlib').sha256(payload).hexdigest()

def _read_split(split: str):
    import pyarrow.parquet as pq
    path = PATHS['data_dir'] / PARQUET[split]
    tbl = pq.read_table(str(path), columns=['text', 'meta', '__index_level_0__'])
    texts = tbl.column('text').to_pylist()
    meta = tbl.column('meta').to_pylist()
    idx = tbl.column('__index_level_0__').to_pylist()
    sets = [m['redpajama_set_name'] for m in meta]
    return (texts, sets, idx)

def select(cfg: dict, tokenizer) -> dict:
    """Return the frozen selection plus full diagnostics."""
    raw: Dict[str, dict] = {}
    for split in SPLITS:
        texts, sets, idx = _read_split(split)
        log(f'loaded {split}: {len(texts)} rows')
        hh = [sha256_text_utf8(t) for t in texts]
        raw[split] = {'texts': texts, 'sets': sets, 'idx': idx, 'h_text': hh}
    cross = set(raw['validation']['h_text']) & set(raw['test']['h_text'])
    log(f'cross-split duplicate text hashes: {len(cross)}')
    kept = {}
    for split in SPLITS:
        r = raw[split]
        keep = [i for i, h in enumerate(r['h_text']) if h not in cross]
        kept[split] = keep
        log(f"{split}: kept {len(keep)} / {len(r['h_text'])} rows after cross-split dedup")
    selection: Dict[str, list] = {}
    n_short = {}
    for split in SPLITS:
        r = raw[split]
        keep = kept[split]
        for domain in DOMAINS:
            cand = [(k_row(split, domain, r['idx'][i], r['texts'][i]), i) for i in keep if r['sets'][i] == domain]
            cand.sort(key=lambda t: t[0])
            elig, short = ([], 0)
            for k, i in cand:
                ids = tokenizer(r['texts'][i], add_special_tokens=False)['input_ids']
                if len(ids) >= 128:
                    elig.append((k, i, ids[:128]))
                else:
                    short += 1
                if len(elig) >= 16:
                    break
            n_short[split, domain] = short
            if len(elig) < 16:
                raise RuntimeError(f'DATA_BLOCKED: {split}/{domain} has <16 eligible rows')
            selection[f'{split}|{domain}'] = [{'split': split, 'domain': domain, 'row_index': int(r['idx'][i]), 'text_sha256': r['h_text'][i], 'selection_key': k, 'token_ids': [int(t) for t in ids]} for k, i, ids in elig]
            log(f'{split}/{domain}: eligible={len(elig)} scanned_short={short} candidates={len(cand)}')
    diag = {'cross_split_duplicate_hashes': len(cross), 'rows': {s: int(len(raw[s]['texts'])) for s in SPLITS}, 'kept_after_dedup': {s: int(len(kept[s])) for s in SPLITS}, 'domain_row_counts': {f'{s}|{d}': int(sum((1 for i in kept[s] if raw[s]['sets'][i] == d))) for s in SPLITS for d in DOMAINS}, 'short_rows_skipped_before_16th_hit': {f'{s}|{d}': int(n_short[s, d]) for s in SPLITS for d in DOMAINS}, 'parquet': {s: PARQUET[s] for s in SPLITS}}
    return {'selection': selection, 'diagnostics': diag}
