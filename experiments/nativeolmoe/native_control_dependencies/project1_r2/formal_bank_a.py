# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Response-blind v154 data construction. No model or response dependencies."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gzip
import hashlib
import importlib.abc
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
DATASET = 'allenai/OLMoE-mix-0924'
REVISION = 'efe872cfd4bcd7b9233fdb1f7289b3295f120f3a'
TOKENIZER = 'allenai/OLMoE-1B-7B-0924'
TOKENIZER_REVISION = 'e5642d2d7b37c138775ddf59f685f91788e173b7'
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
SEED = 'R2A-20261005'
REQUIRED = {'Formal_Banks_v1p1.json': 'c5055703bb81d16de1f7f194e9c670540a1984719bd01c9f8dcacbfe8f3fac70', 'Formal_Banks_v1p2.json': '80eac9dd99453359aa38a10c2b726bf4d21dc67e708682aa5ca3f0b68db8727e', 'WP1_Pilot_Banks_v1.json': 'a550ecb2fe2653e03a6b89b56299d71e4114a9eb468dc21d99abe8512796f397', 'WP1_FP32_Extension_Banks_v1.json': 'cead8c47c7cf22471880c392ae58788b1cc9910c4e2ba26669ed2352f2e98053'}
SCHEMAS = {'Formal_Banks_v1p1', 'Formal_Banks_v1p2', 'wp1-pilot-banks-v1', 'wp1-fp32-extension-banks-v1', 'M1P5_BankManifest_v2'}
CHANNELS = ('source_ids', 'document_ids', 'text_sha256', 'raw_record_sha256', 'token_array_sha256', 'opened_shards')

class BankStop(RuntimeError):

    def __init__(self, code, reason, details=None):
        super().__init__(reason)
        self.code, self.details = (code, details or {})

def require(condition, code, reason, details=None):
    if not condition:
        raise BankStop(code, reason, details)

def compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf8')

def digest(value):
    return hashlib.sha256(value).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def normalize(text):
    require(isinstance(text, str), 'INVALID_TEXT', 'Text must be a string.')
    return unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n'))

def rule_hash(kind, source_id):
    require(kind in ('select', 'offset', 'half'), 'RULE', 'Unknown frozen hash rule.')
    return digest((SEED + '|' + kind + '|' + source_id).encode('utf8'))

def offset(source_id, n):
    require(n >= 256, 'SHORT_DOCUMENT', 'Document has fewer than 256 tokens.')
    return int(rule_hash('offset', source_id), 16) % (n - 255)

class ModelImportGuard(importlib.abc.MetaPathFinder):
    forbidden = ('torch', 'transformers', 'tensorflow', 'jax', 'project1_r2.model_io', 'project1_r2.expert_internal', 'project1_r2.pilot', 'project1_r2.sketch', 'project1_r2.hard_routing_diagnostic', 'project1_r2.shadow_oracle_repair')

    def find_spec(self, fullname, path=None, target=None):
        if any((fullname == x or fullname.startswith(x + '.') for x in self.forbidden)):
            raise BankStop('MODEL_PROHIBITION', 'Forbidden model/response import: ' + fullname)

    def install(self):
        require(not any((x in sys.modules for x in self.forbidden)), 'MODEL_PROHIBITION', 'Model dependency already imported.')
        sys.meta_path.insert(0, self)

class BuildLog:

    def __init__(self, path):
        self.path = Path(path)
        require(not self.path.exists(), 'OUTPUT_EXISTS', 'Never overwrite a prior build log.')
        self.previous, self.seq = (None, 0)

    def event(self, event, **values):
        row = dict(seq=self.seq, utc=datetime.now(timezone.utc).isoformat(), event=event, previous_sha256=self.previous, **values)
        row['event_sha256'] = digest(compact(row))
        with self.path.open('a', encoding='utf8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
        self.previous = row['event_sha256']
        self.seq += 1
        print(event, values.get('path', values.get('domain', '')), flush=True)

def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            if isinstance(value, (dict, list)):
                yield from walk(value)

def collect(row, identities, root=None):
    for name, channel in [('source_id', 'source_ids'), ('document_id', 'document_ids')]:
        if name in row:
            require(isinstance(row[name], str) and bool(row[name]), 'HISTORY_SCHEMA', 'Invalid historical identity.')
            identities[channel].add(row[name])
            match = re.match(re.escape(DATASET + '@' + REVISION + '/') + '(data/[^#]+)#', row[name])
            if match:
                identities['opened_shards'].add(match.group(1))
    for name in ('source_shard', 'shard_id'):
        if isinstance(row.get(name), str) and row[name].startswith('data/'):
            identities['opened_shards'].add(row[name])
    fields = {'source_sha256': 'text_sha256', 'raw_text_sha256': 'text_sha256', 'normalized_text_sha256': 'text_sha256', 'raw_record_sha256': 'raw_record_sha256', 'token_sha256': 'token_array_sha256', 'token_window_sha256': 'token_array_sha256', 'token_array_sha256': 'token_array_sha256'}
    for name, channel in fields.items():
        if row.get(name) is not None:
            require(isinstance(row[name], str) and bool(re.fullmatch('[0-9a-f]{64}', row[name])), 'HISTORY_SCHEMA', 'Malformed historical hash: ' + name)
            identities[channel].add(row[name])
    ids = row.get('token_ids')
    if ids is None and root and row.get('text_or_token_path') and ('source_id' in row):
        p = (root / row['text_or_token_path']).resolve()
        require(p.is_file(), 'HISTORY_MISSING', 'Missing historical token payload: ' + str(p))
        ids = json.loads(p.read_text())
    if ids is not None:
        require(isinstance(ids, list) and len(ids) == 256 and all((type(v) is int and v >= 0 for v in ids)), 'HISTORY_SCHEMA', 'Invalid historical 256-token array.')
        actual = digest(compact(ids))
        for name in ('token_sha256', 'token_window_sha256', 'token_array_sha256'):
            require(row.get(name, actual) == actual, 'HISTORY_HASH', 'Historical token-array hash mismatch.')
        identities['token_array_sha256'].add(actual)
    for window in row.get('windows', {}).values():
        if window.get('sha256'):
            require(bool(re.fullmatch('[0-9a-f]{64}', window['sha256'])), 'HISTORY_SCHEMA', 'Bad historical window hash.')
            identities['token_array_sha256'].add(window['sha256'])

def historical_ledger(paths, frame_seal=None, log=None, require_authorities=True):
    identities = {k: set() for k in CHANNELS}
    groups, inputs = ({}, [])
    for value in sorted(set((str(Path(p).resolve()) for p in paths))):
        p = Path(value)
        require(p.is_file(), 'HISTORY_MISSING', 'Missing historical manifest: ' + value)
        h = file_sha(p)
        groups.setdefault(p.name, {}).setdefault(h, []).append(p)
    if require_authorities:
        for name, h in REQUIRED.items():
            require(name in groups and set(groups[name]) == {h}, 'HISTORY_HASH', 'Missing/conflicting pinned authority: ' + name)
    for name, alternatives in groups.items():
        require(len(alternatives) == 1, 'HISTORY_CONFLICT', 'Conflicting copies of ' + name)
        h, aliases = next(iter(alternatives.items()))
        p = aliases[0]
        data = json.loads(p.read_text())
        require(isinstance(data, dict) and (data.get('schema') in SCHEMAS or data.get('scope') == 'engineering'), 'HISTORY_SCHEMA', 'Unknown historical bank schema: ' + name)
        for reference in data.get('exclusion_manifests', []):
            q = Path(reference['path'])
            require(q.is_file() and file_sha(q) == reference['sha256'], 'HISTORY_HASH', 'Known historical exclusion reference missing/conflicting: ' + str(q))
        if data.get('pilot_manifest_sha256'):
            require(data['pilot_manifest_sha256'] == REQUIRED['WP1_Pilot_Banks_v1.json'], 'HISTORY_HASH', 'Pilot reference hash conflict.')
        rows = 0
        for row in walk(data):
            collect(row, identities, p.parent)
            rows += int('source_id' in row)
        require(rows > 0, 'HISTORY_SCHEMA', 'Historical manifest has no recoverable source identities: ' + name)
        inputs.extend((dict(path=str(q), sha256=h, schema=data.get('schema', 'engineering'), records=rows) for q in aliases))
        if log:
            log.event('history_manifest', path=str(p), aliases=len(aliases), records=rows)
    if frame_seal:
        p = Path(frame_seal).resolve()
        data = json.loads(p.read_text())
        require(data.get('schema') == 'M1P5_CandidateFrameSeal_v2', 'HISTORY_SCHEMA', 'Unknown historical frame seal.')
        checksum_file = p.parent / 'SHA256SUMS'
        checks = dict(((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in checksum_file.read_text().splitlines() if '  ' in line))
        for q in [p, p.parent / 'M1P5_BankManifest_v2.json']:
            require(q.name in checks and file_sha(q) == checks[q.name], 'HISTORY_HASH', 'Historical seal/manifest checksum conflict.')
        inputs.append(dict(path=str(p), sha256=file_sha(p), schema=data['schema']))
        identities['opened_shards'].update(data['processing_order'])
        for frame in data['frames']:
            q = p.parent / frame['metadata_file']
            require(q.is_file() and file_sha(q) == frame['metadata_file_sha256'], 'HISTORY_HASH', 'Historical frame metadata missing/corrupt: ' + str(q))
            count = 0
            with gzip.open(q, 'rt', encoding='utf8') as f:
                for line in f:
                    collect(json.loads(line), identities)
                    count += 1
            require(count == frame['frame_count'], 'HISTORY_SCHEMA', 'Historical frame count conflict.')
            inputs.append(dict(path=str(q), sha256=frame['metadata_file_sha256'], records=count, schema='sealed historical candidate metadata'))
            if log:
                log.event('history_frame', path=str(q), records=count)
    return dict(schema='R2_Historical_Exclusion_Ledger_v154', complete=True, input_files=inputs, counts={k: len(v) for k, v in identities.items()}, **{k: sorted(v) for k, v in identities.items()})

def shard_availability(tree, ledger):
    require(tree.get('repository') == DATASET and tree.get('revision') == REVISION, 'DATASET_REVISION', 'Unverified pinned dataset revision.')
    opened = set(ledger['opened_shards'])
    files = [v['path'] for v in tree['files']]
    require(len(files) == len(set(files)), 'DATASET_TREE', 'Duplicate file paths in dataset tree.')
    rows, selected = ([], {})
    for domain in DOMAINS:
        all_paths = sorted((p for p in files if p.startswith('data/' + domain + '/') and p.endswith(('.json.gz', '.jsonl.gz', '.json.zst', '.jsonl.zst'))))
        remaining = [p for p in all_paths if p not in opened]
        rows.append(dict(domain=domain, total_shards=len(all_paths), opened_shards=[p for p in all_paths if p in opened], remaining_shards=remaining, remaining_count=len(remaining), proposed_first_four=remaining[:4], passes=len(remaining) >= 4))
        selected[domain] = remaining[:4]
    return dict(dataset_repository=DATASET, dataset_revision=REVISION, all_domains_frozen=all((r['passes'] for r in rows)), record_reading_started=False, domains=rows, selected_shards=selected)

def freeze_four(tree, ledger):
    report = shard_availability(tree, ledger)
    failed = next((r for r in report['domains'] if not r['passes']), None)
    require(failed is None, 'INSUFFICIENT_FRESH_SHARDS', 'Cannot freeze four never-opened shards for ' + str(failed['domain'] if failed else ''), report)
    return report

def eligible_frame(records, shard, domain, tokenize, ledger):
    require(shard not in set(ledger['opened_shards']), 'OPENED_SHARD', 'Opened historical shard rejected.')
    identities = {k: set(ledger[k]) for k in CHANNELS}
    frame = []
    for line_index, raw in records:
        obj = json.loads(raw)
        text = obj.get('text')
        if not isinstance(text, str) or not text.strip():
            continue
        text = normalize(text)
        ids = tokenize(text)
        require(all((type(x) is int and x >= 0 for x in ids)), 'TOKEN_IDS', 'Tokenizer returned invalid token IDs.')
        if len(ids) < 256:
            continue
        source_id = DATASET + '@' + REVISION + '/' + shard + '#' + str(obj.get('id', 'line:' + str(line_index)))
        o = offset(source_id, len(ids))
        tokens = ids[o:o + 256]
        text_hash, token_hash, raw_hash = (digest(text.encode('utf8')), digest(compact(tokens)), digest(raw))
        if source_id in identities['source_ids'] or source_id in identities['document_ids'] or text_hash in identities['text_sha256'] or (token_hash in identities['token_array_sha256']) or (raw_hash in identities['raw_record_sha256']):
            continue
        frame.append(dict(bank='A', scope='formal_discovery', document_id='A-' + digest(source_id.encode('utf8')), source_id=source_id, domain=domain, source_shard=shard, record_line_zero_based=line_index, raw_record_sha256=raw_hash, normalized_text_sha256=text_hash, source_token_count=len(ids), token_offset=o, token_count=256, token_array_sha256=token_hash, token_ids=tokens, dataset_repository=DATASET, dataset_revision=REVISION, tokenizer_repository=TOKENIZER, tokenizer_revision=TOKENIZER_REVISION, selection_seed=SEED, normalized_text=text, raw_record=raw.decode('utf8')))
        if len(frame) == 64:
            break
    require(len(frame) == 64, 'FRAME_SHORTAGE', 'Frozen shard cannot supply the first 64 eligible records: ' + shard)
    return frame

def choose_and_split(frames):
    require(set(frames) == set(DOMAINS), 'DOMAIN_COUNT', 'Exactly seven frozen domains required.')
    selected = []
    for domain in DOMAINS:
        require(len(frames[domain]) == 4, 'SHARD_COUNT', 'Exactly four frozen shards/domain required.')
        rows = []
        for shard, candidates in frames[domain].items():
            require(len(candidates) == 64 and all((v['source_shard'] == shard and v['domain'] == domain for v in candidates)), 'FRAME_COUNT', 'Exactly 64 candidates per frozen shard required.')
            rows.extend((dict(v) for v in sorted(candidates, key=lambda v: rule_hash('select', v['source_id']))[:16]))
        rows.sort(key=lambda v: rule_hash('half', v['source_id']))
        for i, row in enumerate(rows):
            row['half'] = int(i >= 32)
        selected.extend(rows)
    for key in ('document_id', 'source_id', 'normalized_text_sha256', 'token_array_sha256'):
        require(len({v[key] for v in selected}) == 448, 'A_DUPLICATE', 'Duplicate within A: ' + key)
    return selected

def guard_tokenizer_revision(repository, revision):
    require((repository, revision) == (TOKENIZER, TOKENIZER_REVISION), 'TOKENIZER_REVISION', 'Only the pinned 41B tokenizer is permitted.')

def pinned_tokenizer(cache):
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from tokenizers import Tokenizer
    guard_tokenizer_revision(TOKENIZER, TOKENIZER_REVISION)
    files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        meta = get_hf_file_metadata(hf_hub_url(TOKENIZER, name, revision=TOKENIZER_REVISION))
        require(meta.commit_hash == TOKENIZER_REVISION, 'TOKENIZER_REVISION', 'Tokenizer file commit contradiction.')
        p = Path(hf_hub_download(TOKENIZER, name, revision=TOKENIZER_REVISION, cache_dir=str(cache)))
        files[name] = dict(path=str(p), sha256=file_sha(p), commit_sha=meta.commit_hash)
    tok = Tokenizer.from_file(files['tokenizer.json']['path'])
    tok.no_truncation()
    tok.no_padding()
    return (lambda text: tok.encode(text, add_special_tokens=False).ids, files)

def frozen_records(shard, selection, cache):
    require(selection['all_domains_frozen'] and shard in sum(selection['selected_shards'].values(), []), 'UNFROZEN_SHARD', 'Refusing a substitute/unfrozen shard.')
    from huggingface_hub import hf_hub_download
    try:
        p = Path(hf_hub_download(DATASET, shard, repo_type='dataset', revision=REVISION, cache_dir=str(cache)))
        if shard.endswith('.gz'):
            with gzip.open(p, 'rb') as f:
                for i, raw in enumerate(f):
                    yield (i, raw.rstrip(b'\r\n'))
        else:
            import zstandard
            import io
            with p.open('rb') as f, zstandard.ZstdDecompressor().stream_reader(f) as stream, io.BufferedReader(stream) as reader:
                for i, raw in enumerate(reader):
                    yield (i, raw.rstrip(b'\r\n'))
    except Exception as exc:
        raise BankStop('FROZEN_SHARD_INACCESSIBLE', 'Frozen shard inaccessible/undecodable; no substitution: ' + shard) from exc

def fetch_tree():
    from huggingface_hub import HfApi
    api = HfApi()
    actual = api.dataset_info(DATASET, revision=REVISION).sha
    require(actual == REVISION, 'DATASET_REVISION', 'Pinned dataset commit contradiction.')
    files = [dict(path=v.path, size=v.size, blob_id=v.blob_id, lfs_sha256=getattr(v.lfs, 'sha256', None) if v.lfs else None) for v in api.list_repo_tree(DATASET, repo_type='dataset', revision=REVISION, recursive=True) if hasattr(v, 'size')]
    return dict(repository=DATASET, revision=REVISION, actual_repository_commit=actual, files=files)

def write_bank(out, frames, selected, tokenizer_files):
    out = Path(out)
    for name in ('tokens', 'texts', 'records'):
        (out / name).mkdir()
    public_frames = {}
    for domain, shards in frames.items():
        public_frames[domain] = {}
        for shard, candidates in shards.items():
            public_frames[domain][shard] = []
            for original in candidates:
                row = dict(original)
                for name, key, payload in [('texts', 'normalized_text_sha256', row.pop('normalized_text').encode('utf8')), ('records', 'raw_record_sha256', row.pop('raw_record').encode('utf8'))]:
                    p = out / name / (row[key] + ('.txt' if name == 'texts' else '.json'))
                    if p.exists():
                        require(p.read_bytes() == payload, 'CONTENT_ADDRESS', 'Content-address conflict.')
                    else:
                        p.write_bytes(payload)
                    row['normalized_text_path' if name == 'texts' else 'raw_record_path'] = p.relative_to(out).as_posix()
                public_frames[domain][shard].append(row)
    write_json(out / 'candidate_frame_v154.json', public_frames)
    manifest = []
    for original in selected:
        row = {k: v for k, v in original.items() if k not in ('token_ids', 'normalized_text', 'raw_record')}
        p = out / 'tokens' / (row['token_array_sha256'] + '.json')
        p.write_bytes(compact(original['token_ids']))
        row['text_or_token_path'] = p.relative_to(out).as_posix()
        manifest.append(row)
    with (out / 'Formal_Bank_A_v1.jsonl').open('x', encoding='utf8') as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    immutable = ['Formal_Bank_A_v1.jsonl', 'candidate_frame_v154.json', 'historical_exclusion_ledger_v154.json', 'fresh_shard_selection_v154.json', 'dataset_tree_v154.json']
    immutable += [p.relative_to(out).as_posix() for d in ('tokens', 'texts', 'records') for p in (out / d).iterdir()]
    return dict(schema='Formal_Bank_A_Seal_v1', FORMAL_BANK_A_SEALED='PENDING_VERIFICATION', FORMAL_BANK_A_RESPONSE_BLIND='YES', A_DISCOVERY_AUTHORIZED='NO', FRESH_BANK_B_AUTHORIZED='NO', FRESH_BANK_C_AUTHORIZED='NO', dataset_repository=DATASET, dataset_revision=REVISION, tokenizer_repository=TOKENIZER, tokenizer_revision=TOKENIZER_REVISION, tokenizer_files=tokenizer_files, hashes={p: file_sha(out / p) for p in sorted(immutable)}, model_probe_computed=False)
