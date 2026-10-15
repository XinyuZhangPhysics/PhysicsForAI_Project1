# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, tarfile, datetime, shutil
BASE = Path.cwd()
OUT = BASE / 'outputs/block_attribution_post_v283M_r1'
ORIGINAL = BASE / 'outputs/PhysicsForAI_Project1_FullTangentBenchmark_v283M_r1.tar.gz'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, x):
    p.write_text(json.dumps(x, indent=2) + '\n')
req = ['FULL_TANGENT_BLOCK_ATTRIBUTION_RESULTS_post_v283M.json', 'FULL_TANGENT_BLOCK_ATTRIBUTION_COORDS_post_v283M.csv', 'FULL_TANGENT_BLOCK_ATTRIBUTION_SUMMARY_post_v283M.csv', 'FULL_TANGENT_BLOCK_ATTRIBUTION_AUDIT_post_v283M.md', 'FULL_TANGENT_BLOCK_ATTRIBUTION_SOURCE_HASHES_post_v283M.sha256', 'MANUSCRIPT_RELEVANCE_BLOCK_ATTRIBUTION_post_v283M.md']
assert all(((OUT / n).is_file() for n in req))
r = json.loads((OUT / req[0]).read_text())
assert r['primary_classification_unchanged'] == 'PARTIAL_CHANNEL_AGREEMENT' and (not r['strict_closure_pass'])
assert r['technical_completion_assessment']['isotropic_response_coordinate_closure_all_64_records_pass']
bind = json.loads((OUT / 'INPUT_BINDING_MANIFEST_post_v283M.json').read_text())
assert sha(ORIGINAL) == bind['archive_sha256']
assert ORIGINAL.stat().st_mode & 146 == 0
(OUT / 'REPRODUCE_BLOCK_ATTRIBUTION_post_v283M.md').write_text('# Reproduction\n\nUse a new writable project workspace. Restore the source-bound original full-tangent archive under outputs/v283M_r1/ and its compressed archive/checksum under outputs/. Restore the exact supplied post-full-tangent input package under work/PhysicsForAI_Project1_PostFullTangent_v283M/.\n\nCopy this source/ snapshot into work/post_attribution/. Use the Python environment recorded in RUNTIME_post_v283M.json and the unchanged absolute external model/checkpoint locators in the original source manifest. If assets moved, relocate only by the identical SHA256. No new model/sample may be substituted.\n\nRun integrity.py, compute.py, summarize.py, verify.py, reports.py, seal.py sequentially from the new workspace. All computation runs on CPU. The prior original q_C fields and four directional probes are read from the sealed run, never regenerated. The closure plan must retain its actual-achieved-accuracy tolerances; reproducing the strict all-probe failure is part of reproducing this outcome, not a reason to loosen limits or redistribute numerical remainders.\n\nsummary.csv has aggregate RMS/association results; BLOCK_ATTRIBUTION_SEED_SUMMARY has seed-resolved versions. BLOCK_GRAM_COHERENCE contains every 3x3 raw/coherence matrix, and BLOCK_(SEED_)POWER_ACCOUNTING records exact rational power identities and the separate comparison to sealed full powers. No new confidence intervals were inferred.\n\nThe compressed result bundle excludes original model/historical caches because they remain externally source-bound; all newly computed block kernels, pair contributions and data are included. OUTPUT_HASHES_post_v283M.sha256 covers every completed run file except itself. Archive SHA256 and read-back verification are sibling deliverables.\n')
script = Path(__file__).resolve()
shutil.copy2(script, OUT / 'source' / script.name)
with (OUT / req[4]).open('a') as f:
    f.write(f"{sha(OUT / 'source' / script.name)}  {OUT / 'source' / script.name}\n")
files = sorted((p for p in OUT.rglob('*') if p.is_file()))
included = [{'path': str(p.relative_to(OUT)), 'sha256': sha(p), 'bytes': p.stat().st_size, 'included': True} for p in files]
write(OUT / 'ARTIFACT_MANIFEST_BLOCK_ATTRIBUTION_post_v283M.json', {'scientific_status': 'OUTCOME_INFORMED_EXPLORATORY_DIAGNOSTIC', 'diagnostic_status': 'STRICT_CLOSURE_TOLERANCE_NOT_MET', 'primary_label_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'included_artifacts': included, 'external_inputs': [{'path': e['path'], 'sha256': e['sha256'], 'role': e['role'], 'included': False} for e in bind['entries'] if e['sha256']], 'all_new_caches_included': True, 'original_source_assets_duplicated': False, 'checksum_policy': 'OUTPUT_HASHES_post_v283M.sha256 includes manifest and all other files except itself. Manifest inventory excludes itself/checksum to avoid self-reference. Sibling SHA256 covers compressed archive.', 'seal_policy': 'files 0444, directories 0555; checksums provide tamper evidence'})
files = sorted((p for p in OUT.rglob('*') if p.is_file()))
hashes = {str(p.relative_to(OUT)): sha(p) for p in files}
(OUT / 'OUTPUT_HASHES_post_v283M.sha256').write_text(''.join((f'{h}  {n}\n' for n, h in hashes.items())))
assert all((sha(OUT / n) == h for n, h in hashes.items()))
for p in OUT.rglob('*'):
    if p.is_file():
        p.chmod(292)
for p in sorted((p for p in OUT.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
    p.chmod(365)
OUT.chmod(365)
archive = BASE / 'outputs/PhysicsForAI_Project1_FullTangentBlockAttribution_post_v283M_r1.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=6) as t:
    t.add(OUT, arcname=OUT.name)
ah = sha(archive)
archive.with_suffix(archive.suffix + '.sha256').write_text(f'{ah}  {archive.name}\n')
count = 0
with tarfile.open(archive, 'r:gz') as t:
    members = {m.name: m for m in t.getmembers() if m.isfile()}
    assert len(members) == len(hashes) + 1
    for n, h in hashes.items():
        m = members[OUT.name + '/' + n]
        assert m.mode == 292
        f = t.extractfile(m)
        actual = hashlib.sha256()
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            actual.update(b)
        assert actual.hexdigest() == h, n
        count += 1
    f = t.extractfile(OUT.name + '/OUTPUT_HASHES_post_v283M.sha256')
    assert hashlib.sha256(f.read()).hexdigest() == sha(OUT / 'OUTPUT_HASHES_post_v283M.sha256')
assert sha(ORIGINAL) == bind['archive_sha256']
archive.chmod(292)
write(BASE / 'outputs/PhysicsForAI_Project1_FullTangentBlockAttribution_post_v283M_r1_ARCHIVE_VERIFICATION.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'archive_integrity_pass': True, 'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'archive_sha256': ah, 'artifact_hashes_verified': count, 'read_only_modes_verified': True, 'required_output_count': len(req), 'strict_scientific_decomposition_closure_pass': False, 'isotropic_response_coordinate_closure_all_64_pass': True, 'diagnostic_status': 'STRICT_CLOSURE_TOLERANCE_NOT_MET', 'primary_classification_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'original_archive_hash_unchanged': True, 'original_archive_read_only': True, 'manuscript_modified': False})
print(json.dumps({'archive_integrity': 'PASS', 'diagnostic_status': 'STRICT_CLOSURE_TOLERANCE_NOT_MET', 'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': ah, 'artifact_hashes_verified': count}))
