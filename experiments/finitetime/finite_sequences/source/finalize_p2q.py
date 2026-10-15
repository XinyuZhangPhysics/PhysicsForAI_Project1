# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json, hashlib, shutil, zipfile, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
O = R / 'outputs/P2Q_RANDOM_COCYCLE_COMPLETE'
A = R / 'work/PhysicsForAI_Project1_Migration_v209M'
P = Path(_p1_path('path_5e0fd824e039'))
H = Path(_p1_path('path_08e09cd8aadb'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load(n):
    return json.loads((O / n).read_text())

def write(n, x):
    (O / n).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
assert load('P2Q_FINAL_VERIFICATION.json')['status'] == 'PASS'
for line in (A / 'SHA256SUMS_v209M.txt').read_text().splitlines():
    h, n = line.split(maxsplit=1)
    assert sha(A / n) == h, n
shutil.copytree(A, O / 'authority/PhysicsForAI_Project1_Migration_v209M', dirs_exist_ok=True)
for n in ['verify_p2q.py', 'finalize_p2q.py']:
    shutil.copy2(R / 'work/p2q_source' / n, O / 'source' / n)
shutil.copy2(P / 'historical/NV0_source/source/preflight.py', O / 'source/inherited_p2p_preflight.py')
shutil.copy2(P / 'source/run_nv1.py', O / 'source/historical_run_nv1.py')
shutil.copy2(P / 'source/run_science.py', O / 'source/historical_run_science.py')
D = O / 'historical/P2H'
D.mkdir(parents=True, exist_ok=True)
for n in ['P2H_PREFLIGHT_SEAL.json', 'P2H_PREDICTIONS.json', 'P2H_DATA_BANK_SEAL.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz', 'P2H_CHECKPOINT_INDEX.json', 'P2H_PRIMARY_DECISION.json', 'P2H_TRAJECTORIES.parquet', 'SHA256SUMS_P2H.txt']:
    shutil.copy2(H / n, D / n)
    assert sha(H / n) == sha(D / n)
shutil.copy2(P / 'P2P_ENVIRONMENT_INVENTORY.json', O / 'P2Q_ENVIRONMENT_INVENTORY.json')
audit = load('P2Q_ASSET_AUDIT.json')
audit['migration_manifest_all_verified'] = True
audit['final_source_and_target_recheck'] = True
for rec in audit['P2P_verified_files']:
    assert sha(rec['path']) == rec['sha256']
for n, h in audit['historical_target_hashes'].items():
    assert sha(H / n) == h
for rec in load('P2Q_TARGET_ACCESS_AUDIT.json'):
    assert sha(rec['path']) == rec['sha256']
write('P2Q_ASSET_AUDIT.json', audit)
comp = load('P2Q_COMPLETION.json')
dec = load('P2Q_PRIMARY_DECISION.json')
rep = load('P2Q_REPLICATION.json')
bs = load('P2Q_BOOTSTRAP_SUMMARY.json')
s = bs['statistics']
rows = '\n'.join((f"| {n} | {s[n]['point']:.9g} | [{s[n]['lower95']:.9g}, {s[n]['upper95']:.9g}] |" for n in ['E_loc_RTO', 'C_loc_RTO', 'R_loc', 'E_Gamma_RTO', 'rho_Gamma_RTO', 'R_Gamma']))
report = f"# P2Q execution report\n\nStatus: COMPLETE. Primary label: `{dec['label']}`. Replication: `{rep['status']}`.\n\nThe exact historical P2H scalar trajectories and checkpoint index match the two frozen SHA-256 values. All 840 declared checkpoint files (8 seeds × 15 arms × 7 times) were byte-hash audited and checked again at finalization. Only updates 0,1,2,4,8,16,32 were opened as full checkpoint tensors. Only declared scalar columns at updates 0–32 were read from the target Parquet. Replication tensor and row access occurred after the primary JSON and its SHA-256 seal existed; P2Q_EVENT_LEDGER.json and P2Q_TARGET_ACCESS_AUDIT.json establish that order.\n\nAll four anchor captures reproduce stored P2P h, phi, g bitwise. Each anchor has zero relative s1 disagreement and zero top-four left-projector Frobenius disagreement. The complete P2P package manifest was verified in place before execution and again at finalization, without duplicating its 2.9 GB contents.\n\nBoth no-fit models use exact historical private seeds, U=0, 15 historical arms, and sealed mu/lambda/learning rates. The two right-hand sides use the previous state simultaneously. RTO applies each exact full matrix-free batch operator to all 15 arm states, with one four-sequence float32 capture per batch and float64 actions. No rank truncation is applied to RTO. F0 uses the original sealed four singular triplets cast to float64 without reorthogonalization. No full dense cross matrix or sequence SVD was formed; only the four authorized anchor reduced-Gram eigensolves were used.\n\nSequence operators: 256. Full-model capture forwards: 1024 sequence + 16 anchor. Logged predictions: 7920 rows. Checkpoint comparisons: 1680 rows. Finite-time gains: 120 rows. Paired seed-block bootstrap: 20000 draws, seed 2026100901. Execution elapsed {comp['elapsed_seconds']:.3f} seconds, excluding preparation, independent verification, and packaging.\n\n| Metric | Point | Percentile 95% CI |\n|---|---:|---:|\n{rows}\n\nLocal conjunction: {dec['local_criterion_pass']}. Window conjunction: {dec['window_criterion_pass']}. Primary seed blocks better under RTO: local {dec['primary_seeds_local_better']}/6; finite-time gain {dec['primary_seeds_gain_better']}/6. Every frozen threshold and split remains unchanged.\n\nIndependent verification recomputed every bootstrap resample and percentile from delivered tables, checked decision conjunctions and target access ordering, checked all 15 vectorized arms on all four retained anchors, reproduced every F0 amplitude through an independent exact coefficient representation that does not assume orthonormal triplets, and reproduced all 15 first-update RTO amplitudes and displacement comparisons. See P2Q_FINAL_VERIFICATION.json.\n\nNo optimizer, parameter training, token-bank construction, replacement batch, additional private seed, network access/download, fitting, horizon extension, or Q3 guidance occurred. The full model stayed frozen. Tangent-state propagation is the authorized deterministic analysis.\n\nThe source convention for the partially verbal third primary label was sealed before new operator outcomes: significant local improvement uses the frozen relative-error upper-CI <=0.75 and 5/6 seed improvement clauses. The replication MIXED versus NOT_CONSISTENT distinction also uses a pre-sealed componentwise convention. These conventions do not affect passing local/window conjunctions.\n\nThe package contains all Section 13 artifacts, sources, logs, authoritative migration package, four declared retained replay captures, exact historical scalar target and checkpoint index, and full SHA-256 manifest. The large model, token banks, private seed file (also embedded in the historical source archive), and historical high-dimensional checkpoints remain in place and are indexed by absolute path and hash. Python/library binaries are external; their recorded exact environment is included. Reproduction on this machine uses source/run_p2q.py at the original project layout; paths and working assumptions are documented in P2Q_REPRODUCE.md.\n\nPrimary decision seal: `{comp['primary_decision_sha256']}`.\n"
(O / 'P2Q_EXECUTION_REPORT.md').write_text(report)
closure = f"# P2Q scientific closure\n\nFrozen primary verdict: `{dec['label']}`.\n\n| Metric | Point | Percentile 95% CI |\n|---|---:|---:|\n{rows}\n\nThe frozen local tangent-cocycle criterion {('passes' if dec['local_criterion_pass'] else 'does not pass')}; the original P2H 8–32 finite-time-gain criterion {('passes' if dec['window_criterion_pass'] else 'does not pass')}. All component booleans and seed-level measurements are reported in P2Q_PRIMARY_DECISION.json and P2Q_BOOTSTRAP_SUMMARY.json. The RTO and F0 models were not fitted to target outcomes.\n\nReplication: `{rep['status']}`. Seed 6 and 7 median local error ratios are {rep['seed_level_R_loc'][0]:.9g} and {rep['seed_level_R_loc'][1]:.9g}; combined RTO local displacement cosine is {rep['combined_C_loc_RTO']:.9g}. Replication was opened after the primary hash seal and cannot alter it.\n\nInterpretation is restricted to the controlled SmolLM2-135M zero-based layer-20 private sector, these exact eight historical streams and 15 arms, and updates 1–32. Gamma_8:32 is a finite-time full-private log gain. This result establishes no asymptotic Lyapunov exponent, diffusion/SDE limit, universal random-matrix ensemble, long-time nonlinear closure, native OLMoE formation, improved model training, or Q3 recommendation. Historical decisions remain unchanged.\n"
(O / 'P2Q_SCIENTIFIC_CLOSURE.md').write_text(closure)
(O / 'P2Q_REPRODUCE.md').write_text(f"# Reproduce P2Q\n\nUse the exact Python executable recorded in P2Q_ASSET_AUDIT.json. Original project root: {_p1_path('path_046030bc90f5')} Execute work/p2q_source/run_p2q.py, then work/p2q_source/verify_p2q.py and work/p2q_source/finalize_p2q.py. Frozen source snapshots are supplied in source/. The script assumes this original project layout and absolute external P2P/P2H paths recorded in P2Q_SOURCE_INDEX.json; preserve the layout or adapt path configuration only before starting a fresh run. Running again overwrites analysis artifacts in outputs/P2Q_RANDOM_COCYCLE_COMPLETE; use a separate project copy when preserving the current seal.\n\nNo installation or downloads are required. External checkpoint, model and bank files must remain at the audited paths and match their hashes; missing or mismatching targets cause a non-scientific stop. The historical source ZIP includes original private seeds and banks. Target replication tensor and scalar rows are gated by primary JSON and SHA-256 seal. All operator work is CPU deterministic, 8 threads, full-model float32 capture and float64 post-capture algebra. Network connection attempts are denied and model loading is local-only.\n\nSHA256SUMS_P2Q.txt covers every package file except itself. The outer ZIP checksum is supplied separately. The external file index lists required reused dependencies without duplicating their large byte content.\n")
external = [dict(path=r['path'], sha256=r['sha256'], role='P2H checkpoint') for r in load('P2Q_TARGET_ACCESS_AUDIT.json')]
external += audit['P2P_verified_files']
external.append(dict(path=str(H.parent / 'work/p2h_private_seeds.pt'), sha256=audit['historical_seed_sha256'], role='historical private seeds'))
write('P2Q_SOURCE_INDEX.json', dict(status='COMPLETE', source_sha256={str(p.relative_to(O)): sha(p) for p in sorted((O / 'source').glob('*.py'))}, external_reused_files=external, python=audit['python'], library_inventory='P2Q_ENVIRONMENT_INVENTORY.json', authority='authority/PhysicsForAI_Project1_Migration_v209M', original_project_root=str(R), large_assets_reused_in_place=True))
mandatory = ['P2Q_ASSET_AUDIT.json', 'P2Q_P2P_ANCHOR_REPLAY.json', 'P2Q_BATCH_SEQUENCE_SEAL.json', 'P2Q_COCYCLE_TRAJECTORIES.parquet', 'P2Q_CHECKPOINT_COMPARISON.csv', 'P2Q_FINITE_TIME_GAINS.csv', 'P2Q_BOOTSTRAP_RESAMPLES.npz', 'P2Q_BOOTSTRAP_SUMMARY.json', 'P2Q_PRIMARY_DECISION.json', 'P2Q_PRIMARY_DECISION.sha256', 'P2Q_REPLICATION.json', 'P2Q_EXECUTION_REPORT.md', 'P2Q_SCIENTIFIC_CLOSURE.md', 'P2Q_SOURCE_INDEX.json']
assert all(((O / n).is_file() for n in mandatory))
manifest = O / 'SHA256SUMS_P2Q.txt'
files = [p for p in sorted(O.rglob('*')) if p.is_file() and p != manifest]
manifest.write_text(''.join((f'{sha(p)}  {p.relative_to(O)}\n' for p in files)))
for l in manifest.read_text().splitlines():
    h, n = l.split(maxsplit=1)
    assert sha(O / n) == h
z = R / 'outputs/P2Q_RANDOM_COCYCLE_COMPLETE.zip'
with zipfile.ZipFile(z, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as f:
    for p in sorted(O.rglob('*')):
        if p.is_file():
            f.write(p, str(p.relative_to(O.parent)))
with zipfile.ZipFile(z) as f:
    assert f.testzip() is None
    for l in manifest.read_text().splitlines():
        h, n = l.split(maxsplit=1)
        assert hashlib.sha256(f.read(O.name + '/' + n)).hexdigest() == h
zh = sha(z)
(z.parent / (z.name + '.sha256')).write_text(zh + '  ' + z.name + '\n')
print(json.dumps(dict(status='PACKAGED_AND_VERIFIED', zip=str(z), sha256=zh, files=len(files) + 1, size_bytes=z.stat().st_size, primary_label=dec['label'], replication=rep['status']), indent=2))
