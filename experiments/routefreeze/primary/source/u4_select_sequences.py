# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Select frozen 16 by 128-token samples per split and domain."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
BASE = Path(__file__).parent
MODEL = _p1_path('path_f2699b70cfbf')
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = defaultdict(list)
for line in (BASE / 'u4_candidates.jsonl').open():
    r = json.loads(line)
    rows[r['split'], r['domain']].append(r)
selected = []
eligibility = {}
for split in ('validation', 'test'):
    for domain in ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4'):
        group = sorted(rows[split, domain], key=lambda r: r['selection_key'])
        count = 0
        for r in group:
            ids = tok.encode(r['text'], add_special_tokens=False)
            if len(ids) < 128:
                continue
            count += 1
            if count <= 16:
                selected.append({k: r[k] for k in ('split', 'domain', 'index', 'text_sha256', 'selection_key')} | {'token_ids': ids[:128]})
            if count >= 16:
                break
        eligibility[f'{split}/{domain}'] = count
        if count < 16:
            raise RuntimeError(f'DATA_BLOCKED: {split}/{domain} has only {count} eligible rows')
assert len(selected) == 128 and all((len(r['token_ids']) == 128 for r in selected))
data = json.dumps(selected, ensure_ascii=False, indent=2) + '\n'
path = BASE / 'u4_selected_sequences.json'
path.write_text(data)
print(json.dumps({'status': 'PASS', 'count': len(selected), 'per_group': eligibility, 'sha256': hashlib.sha256(data.encode()).hexdigest(), 'path': str(path)}, indent=2))
