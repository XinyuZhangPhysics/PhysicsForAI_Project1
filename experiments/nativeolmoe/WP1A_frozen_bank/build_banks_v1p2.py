# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Response-blind, deterministic multi-shard formal bank construction."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import math
import subprocess
import unicodedata
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(__file__).parent
OLD = Path(_p1_path('path_643af9e8d449'))
REV = 'efe872cfd4bcd7b9233fdb1f7289b3295f120f3a'
MODEL = 'allenai/OLMoE-1B-7B-0924'
MODEL_SHA = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
DOMAINS = ['dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki']
SIZES = [('G', 64), ('S', 96), ('P', 96), ('E', 96)]
BASE = f'https://huggingface.co/datasets/allenai/OLMoE-mix-0924/resolve/{REV}/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rank(label):
    return int(sha(label.encode('utf-8')), 16)

def load_exclusions():
    paths = [Path(_p1_path('path_23cf439563e4')), Path(_p1_path('path_98cd919fbc38'))]
    ids, texts, tokens, manifests = (set(), set(), set(), [])
    for path in paths:
        raw = path.read_bytes()
        obj = json.loads(raw)
        for groups in obj['banks'].values():
            for group in groups.values() if isinstance(groups, dict) else [groups]:
                for item in group:
                    ids.add(item['source_id'])
                    texts.add(item['source_sha256'])
                    tokens.add(item['token_sha256'])
        manifests.append({'path': str(path), 'sha256': sha(raw)})
    return (ids, texts, tokens, manifests)

def decompress_prefix(path, raw):
    if path.endswith('.zst'):
        proc = subprocess.run(['zstd', '-dcq'], input=raw, capture_output=True, check=False)
        clear = proc.stdout
    else:
        clear = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    if not clear:
        raise RuntimeError(f'no decompressed bytes: {path}')
    return clear

def fetch_prefix(path, size, wanted):
    url = BASE + urllib.parse.quote(path)
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{wanted - 1}', 'User-Agent': 'formal-prep-multishard/1.2'})
    with urllib.request.urlopen(request, timeout=120) as response:
        expected = f'bytes 0-{wanted - 1}/{size}'
        if response.status != 206 or response.headers.get('Content-Range') != expected:
            raise RuntimeError(f"unexpected range response: {path}: {response.status}, {response.headers.get('Content-Range')}")
        raw = response.read(wanted)
    if len(raw) != wanted:
        raise RuntimeError(f'incomplete prefix: {path}')
    return raw

def choose_shard(entry, q, tokenizer, prior_ids, prior_texts, prior_tokens, old_ids, old_texts, old_tokens):
    path = entry['path']
    size = entry['size']
    wanted = min(size, 1048576)
    while True:
        raw = fetch_prefix(path, size, wanted)
        clear = decompress_prefix(path, raw)
        lines = clear.split(b'\n')
        if wanted < size:
            lines = lines[:-1]
        elif lines and (not lines[-1]):
            lines = lines[:-1]
        selected_candidates = []
        local_ids, local_texts, local_tokens = (set(), set(), set())
        rejections = {'invalid_text': 0, 'short_tokens': 0, 'prior_overlap': 0, 'duplicate': 0}
        inspected = 0
        for index, raw_line in enumerate(lines):
            inspected += 1
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'malformed complete line: {path}#{index}') from exc
            body = obj.get('text') if isinstance(obj, dict) else None
            if not isinstance(body, str) or not body.strip():
                rejections['invalid_text'] += 1
                continue
            norm = unicodedata.normalize('NFC', body.replace('\r\n', '\n').replace('\r', '\n'))
            raw_hash, norm_hash = (sha(body.encode()), sha(norm.encode()))
            stable = obj.get('id')
            source_key = str(stable) if stable is not None and str(stable) else norm_hash
            identity = f'allenai/OLMoE-mix-0924@{REV}/{path}#{source_key}'
            if identity in old_ids or raw_hash in old_texts or norm_hash in old_texts:
                rejections['prior_overlap'] += 1
                continue
            if identity in prior_ids or norm_hash in prior_texts or identity in local_ids or (norm_hash in local_texts):
                rejections['duplicate'] += 1
                continue
            token_ids = tokenizer(norm, add_special_tokens=False)['input_ids']
            if len(token_ids) < 256:
                rejections['short_tokens'] += 1
                continue
            offset = rank(f'20260929|offset|{identity}') % (len(token_ids) - 255)
            segment = token_ids[offset:offset + 256]
            token_hash = sha(json.dumps(segment, separators=(',', ':')).encode())
            if token_hash in old_tokens:
                rejections['prior_overlap'] += 1
                continue
            if token_hash in prior_tokens or token_hash in local_tokens:
                rejections['duplicate'] += 1
                continue
            item = {'source_id': identity, 'source_id_rule': 'record.id + pinned shard path' if stable is not None and str(stable) else 'normalized text SHA256 + pinned shard path', 'source_shard': path, 'record_line_zero_based': index, 'raw_record_sha256': sha(raw_line), 'source_sha256': norm_hash, 'raw_text_sha256': raw_hash, 'token_offset': offset, 'source_tokens': len(token_ids), 'token_ids': segment, 'token_sha256': token_hash, 'shard_lfs_sha256': entry.get('lfs', {}).get('sha256')}
            selected_candidates.append(item)
            local_ids.add(identity)
            local_texts.add(norm_hash)
            local_tokens.add(token_hash)
            if len(selected_candidates) == 4 * q:
                break
        if len(selected_candidates) >= 4 * q or wanted == size:
            break
        wanted = min(size, wanted * 2)
    selected_candidates.sort(key=lambda x: (rank('20260929|formal-v1p2|record|' + x['source_id'] + '|' + x['raw_text_sha256']), x['source_id']))
    retained = selected_candidates[:q]
    stat = {'path': path, 'size': size, 'lfs_sha256': entry.get('lfs', {}).get('sha256'), 'xet_hash': entry.get('xet_hash'), 'quota': q, 'prefix_bytes_read': wanted, 'prefix_sha256': sha(raw), 'records_inspected': inspected, 'candidate_count_inspected': len(selected_candidates), 'selected_document_count': len(retained), 'rejections': rejections}
    return (retained, stat)

def main():
    audit = json.loads((OLD / 'Historical_Source_Audit_v1p1.json').read_text())
    tree_raw = (OLD / 'historical_tree.json').read_bytes()
    assert audit['audit_passed'] and audit['requested_revision'] == audit['resolved_revision'] == REV
    assert sha(tree_raw) == audit['tree_json_sha256']
    tree = json.loads(tree_raw)
    old_ids, old_texts, old_tokens, exclusions = load_exclusions()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA, local_files_only=True)
    all_ids, all_texts, all_tokens = (set(), set(), set())
    banks, sampling = ({}, {})
    for domain in DOMAINS:
        files = [e for e in tree if e['path'].startswith(f'data/{domain}/') and 'size' in e]
        assert len(files) == audit['canonical_file_counts'][domain]
        files.sort(key=lambda e: (rank(f"20260929|formal-v1p2|shard|{domain}|{e['path']}"), e['path']))
        R = min(32, len(files))
        q0, remainder = divmod(352, R)
        selected, stats, deficiencies = ([], [], [])
        for j, entry in enumerate(files[:R]):
            q = q0 + (j < remainder)
            got, stat = choose_shard(entry, q, tokenizer, all_ids, all_texts, all_tokens, old_ids, old_texts, old_tokens)
            selected += got
            stats.append(stat)
            all_ids.update((x['source_id'] for x in got))
            all_texts.update((x['source_sha256'] for x in got))
            all_tokens.update((x['token_sha256'] for x in got))
            if len(got) < q:
                deficiencies.append({'selected_shard': entry['path'], 'missing': q - len(got)})
            print('SHARD', domain, j + 1, '/', R, entry['path'], len(got), '/', q, 'prefix', stat['prefix_bytes_read'], flush=True)
        next_unused = R
        while len(selected) < 352:
            if next_unused >= len(files):
                raise RuntimeError(f'{domain}: no unused shard for deficiency fill')
            entry = files[next_unused]
            missing = 352 - len(selected)
            got, stat = choose_shard(entry, missing, tokenizer, all_ids, all_texts, all_tokens, old_ids, old_texts, old_tokens)
            stat['deficiency_fill'] = True
            selected += got
            stats.append(stat)
            all_ids.update((x['source_id'] for x in got))
            all_texts.update((x['source_sha256'] for x in got))
            all_tokens.update((x['token_sha256'] for x in got))
            deficiencies.append({'fill_shard': entry['path'], 'requested': missing, 'filled': len(got)})
            next_unused += 1
        assert len(selected) == 352
        selected.sort(key=lambda x: (rank('20260929|formal-v1p2|bank|' + x['source_id'] + '|' + x['token_sha256']), x['source_id']))
        offset = 0
        banks[domain] = {}
        for name, n in SIZES:
            banks[domain][name] = selected[offset:offset + n]
            offset += n
        sampling[domain] = {'complete_frozen_shard_count': len(files), 'selected_R': R, 'selected_paths_in_hash_rank': [e['path'] for e in files[:R]], 'per_shard': stats, 'deficiency_fill_events': deficiencies}
        (ROOT / 'sampling_progress.json').write_text(json.dumps(sampling, indent=2) + '\n')
    sampling_obj = {'schema': 'Formal_Shard_Sampling_v1p2', 'dataset_revision': REV, 'historical_tree_sha256': sha(tree_raw), 'domains': sampling}
    (ROOT / 'Formal_Shard_Sampling_v1p2.json').write_text(json.dumps(sampling_obj, indent=2) + '\n')
    manifest = {'schema': 'Formal_Banks_v1p2', 'dataset_repository': 'allenai/OLMoE-mix-0924', 'dataset_revision': REV, 'model': MODEL, 'model_sha': MODEL_SHA, 'sequence_length': 256, 'sampling_artifact_sha256': sha((ROOT / 'Formal_Shard_Sampling_v1p2.json').read_bytes()), 'normalization_rule': 'Unicode NFC after CRLF and CR conversion to LF', 'token_hash_rule': 'SHA256(compact JSON array of 256 integer token IDs)', 'token_offset_rule': 'SHA256(20260929|offset|source_id) modulo (token_count-255)', 'exclusion_manifests': exclusions, 'banks': banks}
    out = ROOT / 'Formal_Banks_v1p2.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':')) + '\n')
    print('SEALED', out, sha(out.read_bytes()), flush=True)
if __name__ == '__main__':
    main()
