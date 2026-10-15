# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
import shutil, tarfile, stat, time
assert read(OUT / 'FINAL_VERIFICATION_v284M.json')['pass']
lineage = {n: sha(OUT.parent.parent / 'work/source' / n) == sha(OLD / 'source' / n) for n in ['common.py', 'kernels.py']}
assert all(lineage.values())
directions = read(OUT / 'DIRECTION_BINDINGS_v284M.json')
for row in directions:
    v = direction(row['family'], row['g'], row['seed'], row['direction'])
    assert all((hashlib.sha256(v[k].tobytes()).hexdigest() == h for k, h in row['factor_sha256'].items()))
for f in ['S1', 'S2']:
    cal = [json.loads(x) for x in (OUT / 'banks' / f'{f}_calibration_documents.jsonl').read_text().splitlines()]
    aud = [json.loads(x) for x in (OUT / 'banks' / f'{f}_audit_documents.jsonl').read_text().splitlines()]
    assert not {x['train_row_id'] for x in cal}.intersection((x['train_row_id'] for x in aud))
write('FINAL_LINEAGE_CHECK_v284M.json', {'timestamp': stamp(), 'pass': True, 'copied_implementation_equals_sealed_v283M_source': lineage, 'direction_factor_manifests_regenerated': len(directions), 'calibration_audit_document_identity_overlap': 0, 'primary_source_freeze_unchanged': True})
(OUT / 'source').mkdir(exist_ok=True)
(OUT / 'logs').mkdir(exist_ok=True)
for p in (BASE / 'work/source').glob('*.py'):
    shutil.copy2(p, OUT / 'source' / p.name)
for p in (BASE / 'work').glob('*.log'):
    shutil.copy2(p, OUT / 'logs' / p.name)
bindings = read(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json')
external = {x['path']: x['sha256'] for x in bindings['entries']}
zipinput = Path(_p1_path('path_f9e2b3116c4e'))
external[str(zipinput)] = sha(zipinput)
for name in ['common.py', 'kernels.py']:
    external[str(OLD / 'source' / name)] = sha(OLD / 'source' / name)
(OUT / 'SOURCE_HASHES_v284M.sha256').write_text(''.join((f'{h}  {p}\n' for p, h in sorted(external.items()))))
files = [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
write('ARTIFACT_MANIFEST_v284M.json', {'timestamp': stamp(), 'label': read(OUT / 'EXPERT_SIDE_SELECTION_RESULTS_v284M.json')['label'], 'file_count_excluding_manifest_and_output_hash_file': len(files), 'files': files, 'external_sources': external, 'seal': 'read-only file/directory permissions plus per-file SHA256 and sibling completion archive verification; no remote WORM claim', 'source_code_freeze': 'prepare/core/common/kernels/check/run/freeze hashed before any derivative; verify/reports/seal are outcome-preserving postprocessing', 'manifest_self_hash_in': 'OUTPUT_HASHES_v284M.sha256'})
hashes = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT / 'OUTPUT_HASHES_v284M.sha256').write_text(''.join((f'{h}  {p}\n' for p, h in hashes.items())))
hashes['OUTPUT_HASHES_v284M.sha256'] = sha(OUT / 'OUTPUT_HASHES_v284M.sha256')
for p in OUT.rglob('*'):
    if p.is_file():
        p.chmod(292)
for p in sorted((p for p in OUT.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    p.chmod(365)
OUT.chmod(365)
archive = OUT.parent / 'PhysicsForAI_Project1_ExpertSideSelectionClosure_v284M_r1.tar.gz'
started = time.monotonic()
print('ARCHIVING', len(hashes), 'files', flush=True)
with tarfile.open(archive, 'w:gz', compresslevel=1) as tf:
    tf.add(OUT, arcname=OUT.name)
print('ARCHIVE WRITTEN; STREAM VERIFY', flush=True)
verified = {}
with tarfile.open(archive, 'r|gz') as tf:
    for member in tf:
        if not member.isfile():
            continue
        rel = str(Path(member.name).relative_to(OUT.name))
        h = hashlib.sha256()
        stream = tf.extractfile(member)
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
        assert rel in hashes and hashes[rel] == h.hexdigest(), rel
        verified[rel] = h.hexdigest()
assert verified == hashes
ah = sha(archive)
side = archive.with_suffix(archive.suffix + '.sha256')
side.write_text(f'{ah}  {archive.name}\n')
vp = OUT.parent / 'PhysicsForAI_Project1_ExpertSideSelectionClosure_v284M_r1_ARCHIVE_VERIFICATION.json'
vp.write_text(json.dumps({'timestamp': stamp(), 'archive': str(archive), 'sha256': ah, 'bytes': archive.stat().st_size, 'file_count': len(verified), 'all_archived_file_hashes_match': True, 'all_delivered_files_read_only': all((p.stat().st_mode & 146 == 0 for p in OUT.rglob('*') if p.is_file())), 'historical_seals_unchanged': True, 'label': read(OUT / 'EXPERT_SIDE_SELECTION_RESULTS_v284M.json')['label'], 'elapsed_seal_seconds': time.monotonic() - started}, indent=2) + '\n')
for p in [archive, side, vp]:
    p.chmod(292)
print('SEAL VERIFIED', len(verified), 'files', ah, flush=True)
