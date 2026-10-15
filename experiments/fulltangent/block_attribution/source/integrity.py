# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, hashlib, tarfile, datetime, shutil
BASE = Path.cwd()
OUT = BASE / 'outputs/block_attribution_post_v283M_r1'
SEALED = BASE / 'outputs/v283M_r1'
ARCH = BASE / 'outputs/PhysicsForAI_Project1_FullTangentBenchmark_v283M_r1.tar.gz'
PACK = BASE / 'work/PhysicsForAI_Project1_PostFullTangent_v283M'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(n, j):
    (OUT / n).write_text(json.dumps(j, indent=2) + '\n')
expected = ARCH.with_suffix(ARCH.suffix + '.sha256').read_text().split()[0]
assert sha(ARCH) == expected == json.loads((BASE / 'outputs/PhysicsForAI_Project1_FullTangentBenchmark_v283M_r1_ARCHIVE_VERIFICATION.json').read_text())['archive_sha256']
hashes = {line.split('  ', 1)[1]: line.split('  ', 1)[0] for line in (SEALED / 'OUTPUT_HASHES_v283M.sha256').read_text().splitlines()}
assert len(hashes) == 188
with tarfile.open(ARCH, 'r:gz') as t:
    members = {m.name: m for m in t.getmembers() if m.isfile()}
    assert len(members) == 189
    for n, h in hashes.items():
        assert sha(SEALED / n) == h, n
        f = t.extractfile(members['v283M_r1/' + n])
        actual = hashlib.sha256()
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            actual.update(b)
        assert actual.hexdigest() == h, n
    assert hashlib.sha256(t.extractfile('v283M_r1/OUTPUT_HASHES_v283M.sha256').read()).hexdigest() == sha(SEALED / 'OUTPUT_HASHES_v283M.sha256')
preg = sha(SEALED / 'FULL_TANGENT_PREREGISTRATION_v283M.md')
assert preg == '0b5b04b7aa2d29b6630095cad95ac91d37ce3fc8e4be609d2bc3b15802a132d7'
freeze = json.loads((SEALED / 'PRIMARY_CLASSIFICATION_FREEZE_v283M.json').read_text())
assert sha(SEALED / 'FULL_TANGENT_RESULTS_v283M.json') == freeze['results_sha256']
assert freeze['classification'] == 'PARTIAL_CHANNEL_AGREEMENT'
source = json.loads((SEALED / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json').read_text())
assert all((sha(e['path']) == e['expected_sha256'] for e in source['entries']))
bind = json.loads((SEALED / 'EXPERT_BLOCK_BINDING_v283M.json').read_text())
for r in bind['records']:
    expectedblocks = {'up.weight', 'up.bias', 'down.weight'} if r['family'] == 'S1' else {'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'}
    actual = {p['parameter_name'].split('].')[-1] for p in r['parameters'] if p['included']}
    assert actual == expectedblocks
    assert all((not p['included'] for p in r['parameters'] if p['parameter_name'].endswith('down.bias')))
checks = json.loads((SEALED / 'FULL_TANGENT_IMPLEMENTATION_CHECK_v283M.json').read_text())
write('DECOMPOSITION_NUMERICS_PLAN_post_v283M.json', {'utc_before_computation': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'absolute_closure_tolerance': checks['max_full_abs_error'], 'aggregate_relative_closure_tolerance': checks['max_full_relative_error'], 'tolerance_source': 'actual achieved full-kernel independent autograd accuracy in sealed run, not its looser acceptance limits', 'primary_dtype': 'float64', 'probes': ['isotropic', 'directional_0', 'directional_1', 'directional_2', 'directional_3'], 'sign_floors': 'reuse sealed per-cell restricted/full floors; block floors use same full floor', 'confidence_intervals': 'not computed; descriptive post hoc summaries and seed-resolved metrics only', 'source_preregistration_modified': False, 'original_primary_label': 'PARTIAL_CHANNEL_AGREEMENT'})
entries = [{'path': str(ARCH), 'sha256': expected, 'role': 'sealed primary archive'}, {'path': str(BASE / 'work/PhysicsForAI_Project1_PostFullTangent_v283M'), 'sha256': None, 'role': 'post hoc task package'}]
for p in sorted(PACK.iterdir()):
    if p.is_file():
        entries.append({'path': str(p), 'sha256': sha(p), 'role': 'post hoc task input'})
        shutil.copy2(p, OUT / p.name)
entries += [{'path': str(SEALED / n), 'sha256': h, 'role': 'sealed run input'} for n, h in hashes.items()]
entries += [{'path': e['path'], 'sha256': e['expected_sha256'], 'role': 'unchanged external source'} for e in source['entries']]
write('INPUT_BINDING_MANIFEST_post_v283M.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS', 'archive_sha256': expected, 'archive_files_verified': 189, 'sealed_data_files_verified': 188, 'external_sources_verified': len(source['entries']), 'preregistration_sha256': preg, 'primary_results_sha256': freeze['results_sha256'], 'expert_block_binding_sha256': sha(SEALED / 'EXPERT_BLOCK_BINDING_v283M.json'), 'entries': entries})
(OUT / 'FULL_TANGENT_BLOCK_ATTRIBUTION_SOURCE_HASHES_post_v283M.sha256').write_text(''.join((f"{e['sha256']}  {e['path']}\n" for e in entries if e['sha256'])))
ARCH.chmod(292)
write('RUN_STATUS_post_v283M.json', {'status': 'SEALED_INPUTS_VERIFIED', 'primary_label_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'no_new_block_outcome_computed': True, 'archive_content_modified': False, 'archive_read_only': True})
print(json.dumps({'integrity': 'PASS', 'archive_files': 189, 'external_sources': len(source['entries']), 'closure_abs_tolerance': checks['max_full_abs_error'], 'closure_rel_tolerance': checks['max_full_relative_error']}))
