# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""v149 engineering-only legacy-G binding; no general tokenizer mismatch bypass."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
import subprocess
from pathlib import Path
from . import DOMAINS, MODEL_REPO, StopPilot
from .banks import document_half
from .reports import sha256, write_json
LEGACY_SHA = '6d84c48581ece794365f2b8e9cfb043c68ade9c5'
TARGET_SHA = 'e5642d2d7b37c138775ddf59f685f91788e173b7'
IDENTITY = dict(schema='Formal_Banks_v1p1', dataset_repository='allenai/OLMoE-mix-0924', dataset_revision='efe872cfd4bcd7b9233fdb1f7289b3295f120f3a', model=MODEL_REPO, model_sha=LEGACY_SHA, sequence_length=256, seed=20260929)
CHECKS = ('tokenizer_class_equal', 'effective_config_equal', 'vocabulary_size_equal', 'exact_vocabulary_token_id_equal', 'special_token_map_equal', 'semantic_asset_hashes_equal', 'deterministic_corpus_encoding_equal')

def locate_legacy(roots, output, meta):
    paths = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        result = subprocess.run(['rg', '--files', '--hidden', str(root), '-g', 'Formal_Banks_v1p1.json', '-g', '!.git', '-g', '!node_modules'], capture_output=True, text=True)
        if result.returncode not in (0, 1):
            raise StopPilot('Exact filename search failed: ' + result.stderr)
        paths.update((Path(line).resolve() for line in result.stdout.splitlines()))
    candidates = []
    for path in sorted(paths):
        record = dict(path=str(path), sha256=sha256(path))
        try:
            data = json.loads(path.read_text())
            record['identity_checks'] = {key: data.get(key) == value for key, value in IDENTITY.items()}
            record['actual_identity'] = {key: data.get(key) for key in IDENTITY}
            record['valid'] = all(record['identity_checks'].values())
        except (ValueError, OSError) as error:
            record.update(valid=False, error=str(error))
        candidates.append(record)
    valid = [c for c in candidates if c['valid']]
    hashes = {c['sha256'] for c in valid}
    chosen = min(valid, key=lambda c: c['path']) if len(hashes) == 1 else None
    write_json(output, dict(exact_filename='Formal_Banks_v1p1.json', search_roots=[str(r) for r in roots], candidates=candidates, selected=chosen, selection_rule='one unique valid content SHA256; identical copies: lexical absolute path', inspected_banks=[]), meta)
    if chosen is None:
        raise StopPilot('Expected exactly one valid legacy content hash; see candidate path/SHA256 report.')
    return chosen

def tokenizer_contract(preprocessing, target_sha, manifest_path):
    if not isinstance(preprocessing, dict):
        raise StopPilot('Missing tokenizer provenance.')
    if preprocessing.get('tokenizer_revision_commit_sha') == target_sha:
        return dict(mode='exact_target', target_revision_commit_sha=target_sha)
    if target_sha != TARGET_SHA or preprocessing.get('tokenizer_source_revision_commit_sha') != LEGACY_SHA or preprocessing.get('tokenizer_equivalent_target_revision_commit_sha') != TARGET_SHA:
        raise StopPilot('Certified equivalence requires exactly the frozen legacy and 41B revisions.')
    report_name = preprocessing.get('tokenizer_equivalence_report_path')
    if not isinstance(report_name, str) or not report_name:
        raise StopPilot('Tokenizer equivalence report path is missing.')
    report_path = (Path(manifest_path).resolve().parent / report_name).resolve()
    if not report_path.is_file():
        raise StopPilot('Tokenizer equivalence report does not exist.')
    if sha256(report_path) != preprocessing.get('tokenizer_equivalence_report_sha256'):
        raise StopPilot('Tokenizer equivalence report SHA256 mismatch.')
    report = json.loads(report_path.read_text())
    if report.get('source_revision_commit_sha') != LEGACY_SHA or report.get('target_revision_commit_sha') != TARGET_SHA or report.get('model_repo') != MODEL_REPO or (report.get('TOKENIZER_EQUIVALENT') is not True) or any((report.get('checks', {}).get(check) is not True for check in CHECKS)):
        raise StopPilot('Tokenizer report does not certify exact semantic equivalence for frozen revisions.')
    return dict(mode='certified_equivalence', source_revision_commit_sha=LEGACY_SHA, target_revision_commit_sha=TARGET_SHA, report_path=str(report_path), report_sha256=sha256(report_path))

def materialize_g112(selected, equivalence_path, directory, meta):
    source = Path(selected['path'])
    if sha256(source) != selected['sha256']:
        raise StopPilot('Legacy source changed after identity gate.')
    data = json.loads(source.read_text())
    if any((data.get(k) != v for k, v in IDENTITY.items())):
        raise StopPilot('Legacy identity changed.')
    directory = Path(directory)
    preprocessing = dict(tokenizer_source_revision_commit_sha=LEGACY_SHA, tokenizer_equivalent_target_revision_commit_sha=TARGET_SHA, tokenizer_equivalence_report_path='../tokenizer_equivalence_report.json', tokenizer_equivalence_report_sha256=sha256(equivalence_path), add_special_tokens=False, normalization='CRLF/CR -> LF, then Unicode NFC', sequence_length=256, legacy_source='Formal_Banks_v1p1.json', legacy_source_path=str(source), legacy_source_sha256=selected['sha256'], legacy_bank='G', engineering_only=True)
    tokenizer_contract(preprocessing, TARGET_SHA, directory / 'engineering_manifest.json')
    selections = []
    seen = set()
    counts = {}
    for domain in DOMAINS:
        records = data['banks'][domain]['G']
        half_counts = {0: 0, 1: 0}
        for index, record in enumerate(records):
            doc_id = record['source_id']
            half = document_half(doc_id)
            if half_counts[half] >= 8:
                continue
            if doc_id in seen:
                raise StopPilot('Duplicate source_id in frozen G112 selection.')
            tokens = record['token_ids']
            if len(tokens) != 256 or any((type(t) is not int for t in tokens)):
                raise StopPilot('Legacy selected token array must contain exactly 256 integers.')
            raw = json.dumps(tokens, separators=(',', ':'), ensure_ascii=False).encode()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != record['token_sha256']:
                raise StopPilot('Legacy compact token-array SHA256 mismatch.')
            selections.append((domain, index, half, doc_id, record, raw, digest))
            seen.add(doc_id)
            half_counts[half] += 1
        if half_counts != {0: 8, 1: 8}:
            raise StopPilot(f'Domain {domain} lacks 8 G documents in each deterministic half.')
        counts[domain] = {str(k): v for k, v in half_counts.items()}
    (directory / 'tokens').mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, index, half, doc_id, record, raw, digest in selections:
        token_path = directory / 'tokens' / f'{domain}_G{index:03d}.json'
        token_path.write_bytes(raw)
        if sha256(token_path) != digest:
            raise StopPilot('Materialized token file SHA256 mismatch.')
        rows.append(dict(document_id=doc_id, source_id=record['source_id'], domain=domain, text_or_token_path=str(token_path.relative_to(directory)), bank='engineering', half=half, token_sha256=digest, legacy_G_index=index, legacy_record_provenance={k: v for k, v in record.items() if k != 'token_ids'}))
    manifest = dict(scope='engineering', preprocessing=preprocessing, documents=rows)
    write_json(directory / 'engineering_manifest.json', manifest, meta)
    report = dict(scope='engineering_only', legacy_source_path=str(source), legacy_source_sha256=selected['sha256'], identity=IDENTITY, legacy_bank='G', document_id_rule='source_id (explicit pinned-source unique identifier)', selection_rule='stored G order; first 8 per hash half per domain', document_count=len(rows), half_domain_counts=counts, all_token_hashes_valid=True, token_count_per_document=256, total_tokens=256 * len(rows), engineering_manifest_sha256=sha256(directory / 'engineering_manifest.json'), selected_document_ids=[row['document_id'] for row in rows], selected_G_indices={d: [row['legacy_G_index'] for row in rows if row['domain'] == d] for d in DOMAINS}, formal_R2_banks_opened=False, other_legacy_banks_inspected=False)
    write_json(directory / 'materialization_report.json', report, {**meta, 'bank_manifest_sha256': report['engineering_manifest_sha256']})
    return report
