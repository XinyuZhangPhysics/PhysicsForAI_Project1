# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Operational continuation of the exact v155 attempt; no scientific changes."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import gzip
import json
import os
import re
import subprocess
import time
from pathlib import Path
from .document_fresh_bank_a import evaluate, append_jsonl, new_sets, DocumentGuard
from .formal_bank_a import BankStop, require, file_sha, digest, write_json, DATASET, REVISION
CHUNK = 33554432

def inspect_sets(rows):
    sets = {k: set() for k in ('source_ids', 'document_ids', 'text_sha256', 'raw_record_sha256', 'token_array_sha256')}
    fields = {'source_ids': ['source_id'], 'document_ids': ['source_id', 'document_id', 'raw_document_id'], 'text_sha256': ['normalized_text_sha256', 'raw_text_sha256'], 'raw_record_sha256': ['raw_record_sha256'], 'token_array_sha256': ['token_array_sha256']}
    for row in rows:
        for k, fs in fields.items():
            for f in fs:
                if row.get(f) is not None:
                    sets[k].add(row[f])
    return sets

def continuation_domain(domain, paths, records, tokenize, history, seen, accept, expose):
    frame = []
    for shard in paths:
        stream = records(shard)
        try:
            for i, raw in stream:
                row, evidence = evaluate(raw, i, shard, domain, tokenize, history, seen)
                expose(evidence)
                delta = inspect_sets([evidence])
                for k in seen:
                    seen[k].update(delta[k])
                if row:
                    frame.append(row)
                    accept(row)
                    if len(frame) == 256:
                        return frame
        finally:
            if hasattr(stream, 'close'):
                stream.close()
    raise BankStop('FRAME_SHORTAGE', 'Pinned domain exhausted before 256 eligible documents', {'domain': domain, 'eligible': len(frame)})

def parse_final_headers(text):
    blocks = re.split('\\r?\\n\\r?\\n', text.strip())
    finals = [b for b in blocks if b.startswith('HTTP/')]
    require(bool(finals), 'TRANSPORT_HEADERS', 'No HTTP status')
    lines = finals[-1].splitlines()
    status = int(lines[0].split()[1])
    headers = dict(((line.split(':', 1)[0].strip().lower(), line.split(':', 1)[1].strip()) for line in lines[1:] if ':' in line))
    return (status, headers)

def append_range(part, chunk, headers, start, end, total):
    status, h = parse_final_headers(headers)
    require(status == 206 and h.get('content-range') == f'bytes {start}-{end}/{total}', 'TRANSPORT_RANGE', 'Exact Content-Range required')
    require(chunk.stat().st_size == end - start + 1, 'TRANSPORT_RANGE', 'Incomplete range cannot be committed')
    require((part.stat().st_size if part.exists() else 0) == start, 'TRANSPORT_RANGE', 'Cache offset mismatch')
    with part.open('ab') as f, chunk.open('rb') as src:
        for b in iter(lambda: src.read(1 << 20), b''):
            f.write(b)
        f.flush()
        os.fsync(f.fileno())

def validate_complete(path, size, expected):
    require(path.stat().st_size == size, 'TRANSPORT_SIZE', 'Compressed object is incomplete')
    require(file_sha(path) == expected, 'TRANSPORT_SHA256', 'Compressed object hash contradiction')
    return path

class ExactCache:

    def __init__(self, cache, out, tree, log):
        self.cache = Path(cache)
        self.out = Path(out)
        self.log = log
        self.files = {v['path']: v for v in tree['files']}
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.out / 'transport').mkdir(exist_ok=True)
        self.completed = []

    def complete(self, shard):
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        expected = self.files[shard]
        identity = expected['lfs_sha256']
        size = expected['size']
        require(isinstance(identity, str) and bool(re.fullmatch('[0-9a-f]{64}', identity)), 'TRANSPORT_IDENTITY', 'LFS SHA256 required')
        obj = self.cache / identity
        part = self.cache / (identity + '.incomplete')
        row = dict(source_shard=shard, revision=REVISION, expected_size=size, expected_sha256=identity)
        base = hf_hub_url(DATASET, shard, repo_type='dataset', revision=REVISION)
        start_time = time.monotonic()
        journal = self.out / 'transport' / (digest(shard.encode()) + '.jsonl')
        attempts = 0
        if obj.exists():
            validate_complete(obj, size, identity)
            row.update(path=str(obj.resolve()), bytes=size, sha256=identity, status='PASS_CACHE_REUSE')
            self.completed.append(row)
            append_jsonl(journal, row)
            return obj
        if part.exists():
            require(part.stat().st_size <= size, 'TRANSPORT_SIZE', 'Oversize partial cache')
        while (part.stat().st_size if part.exists() else 0) < size:
            offset = part.stat().st_size if part.exists() else 0
            end = min(offset + CHUNK, size) - 1
            succeeded = False
            for trial in range(1, 5):
                require(time.monotonic() - start_time < 3600, 'TRANSPORT_RETRY_EXHAUSTED', 'One-hour pinned-object deadline exhausted', row)
                attempts += 1
                chunk = self.cache / (identity + '.chunk')
                hdr = self.cache / (identity + '.headers')
                record = dict(source_shard=shard, offset=offset, end=end, trial=trial, attempt=attempts)
                try:
                    url = base + '?download=true&transport_nonce=' + str(time.time_ns())
                    meta = get_hf_file_metadata(url, timeout=30)
                    require(meta.commit_hash == REVISION and meta.etag == identity and (meta.size == size), 'TRANSPORT_IDENTITY', 'Remote pinned metadata contradiction')
                    if chunk.exists():
                        chunk.unlink()
                    cmd = ['curl', '--silent', '--show-error', '--fail', '--location', '--connect-timeout', '30', '--max-time', '120', '--header', 'Accept-Encoding: identity', '--range', f'{offset}-{end}', '--dump-header', str(hdr), '--output', str(chunk), url]
                    p = subprocess.run(cmd, capture_output=True, text=True)
                    require(p.returncode == 0, 'TRANSPORT_RETRY', 'curl raw-byte transfer failed', {'curl_exit': p.returncode, 'error': p.stderr[-1500:]})
                    append_range(part, chunk, hdr.read_text(), offset, end, size)
                    record.update(status='RANGE_COMMITTED', committed_bytes=part.stat().st_size, range_sha256=file_sha(chunk), commit_sha=meta.commit_hash, lfs_sha256=meta.etag)
                    succeeded = True
                except Exception as exc:
                    record.update(status='RETRY_FAILURE', error_class=type(exc).__name__, error=str(exc), details=getattr(exc, 'details', {}))
                    if isinstance(exc, BankStop) and exc.code in ('TRANSPORT_IDENTITY', 'TRANSPORT_SIZE', 'TRANSPORT_SHA256', 'TRANSPORT_RANGE'):
                        append_jsonl(journal, record)
                        raise
                append_jsonl(journal, record)
                if succeeded:
                    self.log.event('cache_range_committed', path=shard, bytes=part.stat().st_size, total=size)
                    break
                self.log.event('cache_range_retry', path=shard, offset=offset, trial=trial, error=record['error'])
            require(succeeded, 'TRANSPORT_RETRY_EXHAUSTED', 'Bounded retries exhausted for exact pinned range', dict(row, offset=offset, attempts=attempts))
        validate_complete(part, size, identity)
        part.rename(obj)
        row.update(path=str(obj.resolve()), bytes=size, sha256=identity, status='PASS', attempts=attempts)
        self.completed.append(row)
        append_jsonl(journal, row)
        self.log.event('compressed_object_verified', path=shard, sha256=identity, bytes=size)
        return obj

class CachedRecords:

    def __init__(self, cache, out, traversal, log):
        self.cache, self.out, self.traversal, self.log = (cache, Path(out), traversal, log)
        self.access = []
        (self.out / 'scan_records').mkdir()

    def __call__(self, shard):
        require(shard in sum(self.traversal['domains'].values(), []), 'UNFROZEN_SHARD', 'Unfrozen source')
        p = self.cache.complete(shard)
        expected = self.cache.files[shard]
        saved = self.out / 'scan_records' / (digest(shard.encode()) + '.jsonl.gz')
        state = dict(source_shard=shard, commit_sha=REVISION, lfs_sha256=expected['lfs_sha256'], full_compressed_size=expected['size'], full_shard_sha256_verified=True, complete_cache_path=str(p.resolve()), cache_sha256=file_sha(p), scan_path=saved.relative_to(self.out).as_posix(), records=0, eof=False, transport='complete content-addressed raw-byte cache')
        self.access.append(state)
        self.log.event('records_begin', path=shard, compressed_object_complete=True)
        try:
            with gzip.open(p, 'rb') as source, gzip.open(saved, 'xb') as dest:
                for i, line in enumerate(source):
                    dest.write(line)
                    dest.flush()
                    state['records'] += 1
                    yield (i, line.rstrip(b'\r\n'))
                state['eof'] = True
        finally:
            state['scan_sha256'] = file_sha(saved)
            self.log.event('records_end', path=shard, records=state['records'], eof=state['eof'], scan_sha256=state['scan_sha256'])
