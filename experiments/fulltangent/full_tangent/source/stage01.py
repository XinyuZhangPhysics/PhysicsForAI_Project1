# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, hashlib, shutil, datetime, zoneinfo, platform, subprocess, sys
import torch, numpy as np
BASE = Path.cwd()
OUT = BASE / 'outputs/v283M_r1'
PK = BASE / 'work/PhysicsForAI_Project1_FullTangentLaunch_v283M'
MIG = BASE / 'work/PhysicsForAI_Project1_Migration_v283M'
ROOT = Path(_p1_path('path_8c32eaf570f6'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def git(p):

    def run(*a):
        r = subprocess.run(['git', '-C', str(p), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {'root': str(p), 'commit': run('rev-parse', 'HEAD'), 'status': run('status', '--porcelain')}
for n in ['FULL_TANGENT_PREREGISTRATION_v283M.md', 'FULL_TANGENT_PREREGISTRATION_v283M.sha256', 'CODEX_FullTangentBenchmark_EXECUTION_v283M.md']:
    shutil.copy2(PK / n, OUT / n)
expected = (PK / 'FULL_TANGENT_PREREGISTRATION_v283M.sha256').read_text().split()[0]
assert sha(OUT / 'FULL_TANGENT_PREREGISTRATION_v283M.md') == expected
now = datetime.datetime.now(datetime.timezone.utc)
write('ENVIRONMENT_v283M.json', {'utc': now.isoformat(), 'asia_shanghai': now.astimezone(zoneinfo.ZoneInfo('Asia/Shanghai')).isoformat(), 'python': sys.version, 'python_executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'macos': platform.mac_ver(), 'platform': platform.platform(), 'devices': {'cpu': True, 'mps_built': torch.backends.mps.is_built(), 'mps_available': torch.backends.mps.is_available(), 'cuda_available': torch.cuda.is_available()}, 'git': [git(ROOT), git(ROOT / '04_prospective_experiments/EXP_S2/repository_full')], 'output_location_reason': 'User AGENTS default: deliverables under current Codex project outputs; historical research root accessed read-only.'})
write('RUN_STATUS.json', {'status': 'PREREGISTRATION_VERIFIED', 'preregistration_sha256': expected, 'utc': now.isoformat(), 'no_new_full_tangent_outcome_computed': True})
print('STAGE0 PASS', expected, flush=True)
provpath = MIG / 'EVIDENCE/selection_response/SELECTION_RESPONSE_PROVENANCE_v281M.json'
prov = json.loads(provpath.read_text())
refs = prov['evidence_sources']
records = prov['primary_records']
assert len(records) == 64
manifest = []
fails = []
seen = {}

def bind(p, expected, role):
    p = Path(p)
    if str(p) in seen:
        got = seen[str(p)]
        assert got['expected_sha256'] == expected
        return
    present = p.is_file()
    actual = sha(p) if present else None
    item = {'path': str(p), 'expected_sha256': expected, 'actual_sha256': actual, 'bytes': p.stat().st_size if present else None, 'role': role, 'pass': actual == expected}
    manifest.append(item)
    seen[str(p)] = item
    if not item['pass']:
        fails.append(item)
for k in ['s1_protocol', 's2_protocol', 's1_response', 's2_response', 's1_construction', 's2_construction', 's1_representation_source', 's1_audit_geometry', 's2_audit_geometry', 's1_bank', 's1_pairs', 's2_bank', 's2_pairs']:
    v = refs[k]
    bind(v['path'], v['sha256'], k)
for family in ['S1', 'S2']:
    rr = [r for r in records if r['family'] == family]
    seed0 = 2026100101 if family == 'S1' else 2026110201
    assert {(r['g'], r['seed']) for r in rr} == {(g, s) for g in [1, 2, 4, 8] for s in range(seed0, seed0 + 8)}
    for r in rr:
        assert r['horizon_nonpadding_tokens'] == 64000
        for pathkey, hashkey in [('source_path', 'source_sha256'), ('run_metadata_path', 'run_metadata_sha256'), ('aggregate_path', 'aggregate_sha256')]:
            bind(r[pathkey], r[hashkey], family + '_' + pathkey)
        g = r['g']
        arm = r['arm']
        cache = ROOT / ('04_prospective_experiments/EXP_S1/scientific_full/package/EXP_S1_RESULTS/derived' if family == 'S1' else '04_prospective_experiments/EXP_S2/science_full') / f'audit_geometry_{arm}.npz'
        bind(cache, r['geometry_cache_sha256'], family + '_geometry')
        if Path(r['run_metadata_path']).exists():
            run = json.loads(Path(r['run_metadata_path']).read_text())
            ck = run['checkpoints']['64000']
            rel = ck['router_checkpoint'] if family == 'S1' else ck['path']
            candidates = [ROOT / rel, cache.parent / rel, Path(r['run_metadata_path']).parent / Path(rel).name]
            if family == 'S1':
                candidates += [cache.parent.parent / 'scientific/checkpoints' / f"{arm}_seed_{r['seed']}" / Path(rel).name]
            else:
                candidates += [cache.parent / 'checkpoints' / f"{arm}_seed_{r['seed']}" / Path(rel).name]
            path = next((p for p in candidates if p.is_file()), candidates[0])
            bind(path, r['router_checkpoint_sha256'], family + '_router')
            r['bound_router_path'] = str(path)
            r['bound_geometry_path'] = str(cache)
for k, p in [('migration_provenance', provpath), ('launch_zip', Path(_p1_path('path_15a9c2a56381'))), ('migration_zip', Path(_p1_path('path_1997da014ac8')))]:
    bind(p, sha(p), k)
write('FULL_TANGENT_SOURCE_MANIFEST_v283M.json', {'stage': 1, 'entries': manifest, 'failures': fails, 'primary_records': records, 'all_required_hashes_pass': not fails})
(OUT / 'FULL_TANGENT_SOURCE_HASHES_v283M.sha256').write_text(''.join((f"{x['actual_sha256']}  {x['path']}\n" for x in manifest if x['actual_sha256'])))
shutil.copy2(provpath, OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.json')
print('STAGE1 HASHES', len(manifest), 'FAILURES', len(fails), flush=True)
for f in fails:
    print(json.dumps(f), flush=True)
write('RUN_STATUS.json', {'status': 'TECHNICAL_PROTOCOL_FAILURE' if fails else 'SOURCE_HASHES_VERIFIED', 'stage': 1, 'failure_count': len(fails), 'preregistration_sha256': expected, 'no_new_full_tangent_outcome_computed': True})
