# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen SlimPajama row identity and duplicate exclusion."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
ROOT = Path(_p1_path('path_f6b2b72181c3'))
OUT = Path(__file__).parent / 'u4_candidates.jsonl'
DOMAINS = {'RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4'}
FILES = {'validation': ROOT / 'validation-00000-of-00001-4fb685c22a3f91ef.parquet', 'test': ROOT / 'test-00000-of-00001-9f769cf7ce219017.parquet'}
rows = {}
for split, path in FILES.items():
    table = pq.read_table(path)
    assert {'text', 'meta', '__index_level_0__'} <= set(table.schema.names)
    texts = table['text'].to_pylist()
    metas = table['meta'].to_pylist()
    indices = table['__index_level_0__'].to_pylist()
    rows[split] = [dict(split=split, domain=m['redpajama_set_name'], index=i, text=t, text_sha256=hashlib.sha256(t.encode('utf-8')).hexdigest()) for t, m, i in zip(texts, metas, indices)]
shared = {r['text_sha256'] for r in rows['validation']} & {r['text_sha256'] for r in rows['test']}
filtered = [r for split in FILES for r in rows[split] if r['text_sha256'] not in shared and r['domain'] in DOMAINS]
for r in filtered:
    key_input = '\x00'.join([r['split'], r['domain'], str(r['index']), r['text']])
    r['selection_key'] = hashlib.sha256(key_input.encode('utf-8')).hexdigest()
filtered.sort(key=lambda r: (r['split'], r['domain'], r['selection_key']))
with OUT.open('w') as f:
    for r in filtered:
        f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
print(json.dumps({'schema': 'PASS', 'rows': {k: len(v) for k, v in rows.items()}, 'cross_split_duplicate_hashes': len(shared), 'remaining_domain_counts': {f'{s}/{d}': sum((r['split'] == s and r['domain'] == d for r in filtered)) for s in FILES for d in sorted(DOMAINS)}, 'candidate_file': str(OUT)}, indent=2))
