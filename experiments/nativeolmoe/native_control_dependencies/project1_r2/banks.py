# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from . import DOMAINS, StopPilot
from .reports import sha256

def document_half(document_id):
    return int.from_bytes(hashlib.sha256(document_id.encode('utf8')).digest()[:8], 'big') % 2

def read_manifest(path, allow_extra=False):
    if not path or not Path(path).is_file():
        raise StopPilot('An explicitly supplied engineering bank manifest is required.')
    path = Path(path).resolve()
    if path.suffix == '.csv':
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
        contract = None
    else:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get('scope') != 'engineering':
            raise StopPilot('JSON bank must explicitly declare scope=engineering, not formal A/B/C.')
        rows, contract = (data.get('documents', []), data.get('preprocessing'))
    if not rows:
        raise StopPilot('Empty engineering manifest.')
    seen, extra = (set(), set())
    for row in rows:
        if any((not isinstance(row.get(k), str) or not row[k].strip() for k in ('document_id', 'domain', 'text_or_token_path'))):
            raise StopPilot('Manifest rows require document_id, domain, text_or_token_path.')
        if row['document_id'] in seen:
            raise StopPilot('Duplicate document_id: documents are the independent unit.')
        seen.add(row['document_id'])
        if row['domain'] not in DOMAINS:
            extra.add(row['domain'])
        if row.get('bank', 'engineering') != 'engineering':
            raise StopPilot('Formal-bank row rejected.')
        source = (path.parent / row['text_or_token_path']).resolve()
        if not source.is_file():
            raise StopPilot(f"Missing supplied document: {row['document_id']}")
        row['source_path'] = str(source)
        row['content_sha256'] = sha256(source)
        row['half'] = document_half(row['document_id'])
    if extra and (not allow_extra):
        raise StopPilot('Unknown domains rejected: ' + ', '.join(sorted(extra)))
    return dict(documents=rows, manifest_sha256=sha256(path), preprocessing=contract, registerable=not bool(extra), extra_domains=sorted(extra))

def validate_split(bank):
    domains = sorted({r['domain'] for r in bank['documents']})
    for domain in domains:
        if {r['half'] for r in bank['documents'] if r['domain'] == domain} != {0, 1}:
            raise StopPilot(f'Domain {domain} needs documents in both deterministic halves.')
    if not bank['extra_domains'] and set(domains) != set(DOMAINS):
        raise StopPilot('SNR pilot requires all seven frozen domains.')
    return domains

def tokens_for(row, tokenizer, max_context, preprocessing):
    source = Path(row['source_path'])
    if sha256(source) != row['content_sha256']:
        raise StopPilot('Supplied document changed after manifest validation.')
    if source.suffix == '.npy':
        tokens = np.load(source, allow_pickle=False)
    elif source.suffix == '.json':
        tokens = np.asarray(json.loads(source.read_text()))
    else:
        if not isinstance(preprocessing, dict) or preprocessing.get('add_special_tokens') is None:
            raise StopPilot('Text needs supplied preprocessing.add_special_tokens; no invented preprocessing.')
        tokens = np.asarray(tokenizer(source.read_text(), add_special_tokens=preprocessing['add_special_tokens'])['input_ids'])
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer) or len(tokens) == 0:
        raise StopPilot('Token file must be a nonempty one-dimensional integer array.')
    if len(tokens) > max_context:
        raise StopPilot('Document exceeds model context; no silent truncation/chunking.')
    if tokens.min() < 0 or tokens.max() >= len(tokenizer):
        raise StopPilot('Token IDs outside frozen tokenizer vocabulary.')
    return tokens.astype(np.int64)
