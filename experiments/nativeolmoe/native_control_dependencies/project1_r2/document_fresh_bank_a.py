# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""v155 response-blind document construction; additive to the immutable v154 node."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gzip
import io
import json
import sys
from pathlib import Path
from .formal_bank_a import DATASET, REVISION, TOKENIZER, TOKENIZER_REVISION, DOMAINS, CHANNELS, BankStop, require, compact, digest, normalize, file_sha, write_json, ModelImportGuard
SEED = 'R2A-v155'
FORBIDDEN = ModelImportGuard.forbidden + ('project1_r2.dual_channel_engineering', 'project1_r2.candidate', 'project1_r2.routing', 'project1_r2.jvp')

class DocumentGuard(ModelImportGuard):
    forbidden = FORBIDDEN

def rule(kind, identity):
    require(kind in ('offset', 'select', 'half'), 'RULE', 'Unknown v155 rule')
    return digest((SEED + '|' + kind + '|' + identity).encode())

def freeze_traversal(tree):
    require(tree['repository'] == DATASET and tree['revision'] == REVISION and (tree['actual_repository_commit'] == REVISION), 'DATASET_REVISION', 'Pinned tree commit mismatch')
    paths = [v['path'] for v in tree['files']]
    require(len(paths) == len(set(paths)), 'DATASET_TREE', 'Duplicate paths')
    domains = {}
    for domain in DOMAINS:
        selected = [p for p in paths if p.startswith('data/' + domain + '/') and p.endswith(('.json.gz', '.jsonl.gz', '.json.zst', '.jsonl.zst'))]
        selected.sort(key=lambda p: digest((SEED + '|shard|' + domain + '|' + p).encode()))
        require(bool(selected), 'EMPTY_DOMAIN', domain)
        domains[domain] = selected
    return dict(dataset_repository=DATASET, dataset_revision=REVISION, seed=SEED, all_domains_frozen=True, record_reading_started=False, domains=domains)

def new_sets(ledger):
    return {k: set(ledger[k]) for k in CHANNELS if k != 'opened_shards'}

def evaluate(raw, line, shard, domain, tokenize, history, seen):
    obj = json.loads(raw)
    require(isinstance(obj, dict), 'RECORD_SCHEMA', 'Record is not an object')
    identity = DATASET + '@' + REVISION + '/' + shard + '#' + str(obj.get('id', 'line:' + str(line)))
    doc = 'A-' + digest(identity.encode())
    evidence = dict(source_id=identity, document_id=doc, domain=domain, source_shard=shard, record_line_zero_based=line, raw_record_sha256=digest(raw))
    if obj.get('document_id') is not None:
        evidence['raw_document_id'] = str(obj['document_id'])
    text = obj.get('text')
    if not isinstance(text, str) or not text.strip():
        return (None, dict(evidence, disposition='unusable_text'))
    normalized = normalize(text)
    evidence.update(normalized_text_sha256=digest(normalized.encode()), raw_text_sha256=digest(text.encode()))
    ids = tokenize(normalized)
    require(all((type(v) is int and v >= 0 for v in ids)), 'TOKEN_IDS', 'Invalid tokenizer IDs')
    evidence['source_token_count'] = len(ids)
    if len(ids) < 256:
        return (None, dict(evidence, disposition='short_document'))
    off = int(rule('offset', identity), 16) % (len(ids) - 255)
    tokens = ids[off:off + 256]
    evidence.update(token_offset=off, token_array_sha256=digest(compact(tokens)))
    checks = [('source_ids', identity), ('document_ids', identity), ('document_ids', doc), ('document_ids', evidence.get('raw_document_id')), ('text_sha256', evidence['normalized_text_sha256']), ('text_sha256', evidence['raw_text_sha256']), ('raw_record_sha256', evidence['raw_record_sha256']), ('token_array_sha256', evidence['token_array_sha256'])]
    for channel, value in checks:
        if value is not None and value in history[channel]:
            return (None, dict(evidence, disposition='historical_collision', exclusion_channel=channel))
    for channel, value in checks:
        if value is not None and value in seen[channel]:
            return (None, dict(evidence, disposition='current_frame_duplicate', exclusion_channel=channel))
    for channel, value in checks:
        if value is not None:
            seen[channel].add(value)
    row = dict(evidence, bank='A', scope='response_blind_candidate_frame', token_count=256, token_ids=tokens, dataset_repository=DATASET, dataset_revision=REVISION, tokenizer_repository=TOKENIZER, tokenizer_revision=TOKENIZER_REVISION, selection_seed=SEED)
    return (row, dict(evidence, disposition='candidate_frame'))

def take_domain(domain, traversal, records, tokenize, history, seen, accept, expose, quota=256):
    frame = []
    for shard in traversal:
        stream = records(shard)
        try:
            for i, raw in stream:
                row, evidence = evaluate(raw, i, shard, domain, tokenize, history, seen)
                expose(evidence)
                if row:
                    frame.append(row)
                    accept(row)
                    if len(frame) == quota:
                        return frame
        finally:
            if hasattr(stream, 'close'):
                stream.close()
    raise BankStop('FRAME_SHORTAGE', 'Pinned domain exhausted before 256 eligible documents', {'domain': domain, 'eligible': len(frame)})

def select_a(frames):
    require(set(frames) == set(DOMAINS), 'DOMAIN_COUNT', 'Seven domains required')
    selected = []
    for domain in DOMAINS:
        require(len(frames[domain]) == 256, 'FRAME_COUNT', '256 candidates/domain required')
        rows = [dict(v) for v in sorted(frames[domain], key=lambda v: rule('select', v['source_id']))[:64]]
        rows.sort(key=lambda v: rule('half', v['source_id']))
        for i, row in enumerate(rows):
            row.update(half=int(i >= 32), scope='formal_discovery')
        selected.extend(rows)
    return selected

class PinnedStream:
    """Complete records only; saves every decompressed record actually examined for replay."""

    def __init__(self, out, tree, traversal, log):
        self.out, self.traversal, self.log = (Path(out), traversal, log)
        self.files = {v['path']: v for v in tree['files']}
        self.access = []
        (self.out / 'scan_records').mkdir()

    def __call__(self, shard):
        require(shard in sum(self.traversal['domains'].values(), []), 'UNFROZEN_SHARD', 'Unfrozen path')
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        import requests
        import zstandard
        url = hf_hub_url(DATASET, shard, repo_type='dataset', revision=REVISION)
        meta = get_hf_file_metadata(url)
        expected = self.files[shard]
        require(meta.commit_hash == REVISION and meta.etag == expected['lfs_sha256'] and (meta.size == expected['size']), 'TRANSPORT_IDENTITY', 'Pinned commit/LFS identity mismatch', {'shard': shard})
        p = self.out / 'scan_records' / (digest(shard.encode()) + '.jsonl.gz')
        state = dict(source_shard=shard, commit_sha=meta.commit_hash, lfs_sha256=meta.etag, full_compressed_size=meta.size, scan_path=p.relative_to(self.out).as_posix(), records=0, eof=False, transport='pinned HTTPS streaming prefix', full_shard_sha256_verified=False)
        self.access.append(state)
        self.log.event('records_begin', path=shard, commit_sha=meta.commit_hash, lfs_sha256=meta.etag)
        try:
            with requests.get(url, stream=True, timeout=(30, 120), headers={'Accept-Encoding': 'identity'}) as response:
                response.raise_for_status()
                response.raw.decode_content = False
                decoder = gzip.GzipFile(fileobj=response.raw) if shard.endswith('.gz') else zstandard.ZstdDecompressor().stream_reader(response.raw)
                with decoder, io.BufferedReader(decoder) as reader, gzip.open(p, 'xb') as saved:
                    for i, line in enumerate(reader):
                        saved.write(line)
                        saved.flush()
                        state['records'] += 1
                        yield (i, line.rstrip(b'\r\n'))
                    state['eof'] = True
        except GeneratorExit:
            raise
        except Exception as exc:
            raise BankStop('PINNED_STREAM_FAILURE', 'Cannot read next frozen shard; no substitution', {'shard': shard, 'error': str(exc)}) from exc
        finally:
            state['scan_sha256'] = file_sha(p) if p.exists() else None
            self.log.event('records_end', path=shard, records=state['records'], eof=state['eof'], scan_sha256=state['scan_sha256'])

def append_jsonl(path, row):
    with Path(path).open('a', encoding='utf8') as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        f.flush()

def exposure_export(frame, scanned, ledger):
    sets = {k: set() for k in CHANNELS if k != 'opened_shards'}
    for row in frame:
        for c, f in [('source_ids', 'source_id'), ('document_ids', 'document_id'), ('document_ids', 'raw_document_id'), ('text_sha256', 'normalized_text_sha256'), ('text_sha256', 'raw_text_sha256'), ('raw_record_sha256', 'raw_record_sha256'), ('token_array_sha256', 'token_array_sha256')]:
            if row.get(f) is not None:
                sets[c].add(row[f])
    return dict(schema='v155_candidate_frame_exposure_ledger', candidate_count=len(frame), complete=len(frame) == 1792, all_candidate_rows=frame, counts={k: len(v) for k, v in sets.items()}, **{k: sorted(v) for k, v in sets.items()}, historical_authority='historical_exclusion_ledger_v155.json', historical_counts=ledger['counts'], additional_scanned_exposures=scanned, future_exclusion_requires_all_scanned_records=True)
