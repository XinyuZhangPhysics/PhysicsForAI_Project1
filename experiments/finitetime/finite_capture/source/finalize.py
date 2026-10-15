# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, shutil, csv, zipfile, sys, time, importlib.metadata
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
SCI = ROOT / 'work/p2p_fresh'
NV1 = ROOT / 'work/nv1'
PKG = ROOT / 'work/PhysicsForAI_Project1_Migration_v208M'
DEST = ROOT / 'outputs/P2P_NV1_FRESH_P2P_COMPLETE'
DEST.mkdir(exist_ok=True)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

def write(n, obj):
    (DEST / n).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def manifest(root, name):
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name != name))
    (root / name).write_text(''.join((f'{sha(p)}  {p.relative_to(root)}\n' for p in fs)))
    return fs

def verify_original_assets(audit):
    n = 0
    for r in audit['model_files']:
        assert sha(r['path']) == r['sha256']
        n += 1
    for r in audit['banks'].values():
        assert sha(r['path']) == r['sha256']
        n += 1
    r = audit['F0_spectrum']
    assert sha(r['path']) == r['sha256']
    n += 1
    hist = Path(_p1_path('path_08e09cd8aadb'))
    for name, h in audit['historical_seals'].items():
        assert sha(hist / name) == h
        n += 1
    for name, h in audit['authority_sha256'].items():
        assert sha(PKG / name) == h
        n += 1
    return n

def verify_results():
    out = SCI / 'results'
    completion = json.loads((out / 'P2P_COMPLETION.json').read_text())
    assert completion['status'] == 'COMPLETE'
    events = json.loads((out / 'P2P_EVENT_LEDGER.json').read_text())
    primary = [e for e in events if e['action'] == 'SCIENTIFIC_BATCH_COMPLETE' and e['group'] == 'primary']
    fe = [e for e in events if e['action'] == 'SCIENTIFIC_BATCH_COMPLETE' and e['group'] == 'FE']
    seal = next((e for e in events if e['action'] == 'PRIMARY_DECISION_HASH_SEALED'))
    opened = next((e for e in events if e['action'] == 'FE_SCIENCE_OPENED'))
    assert len(primary) == 32 and len(fe) == 8
    assert max((e['time_ns'] for e in primary)) < seal['time_ns'] < opened['time_ns'] < min((e['time_ns'] for e in fe))
    assert sha(out / 'P2P_PRIMARY_DECISION.json') == seal['sha256'] == (out / 'P2P_PRIMARY_DECISION.sha256').read_text().split()[0]
    assert sha(SCI / 'run_science.py') == events[0]['source_sha256']
    passp = NV1 / 'results/P2P_NV1_NUMERICAL_VALIDATION_PASS.json'
    assert sha(passp) == events[0]['pass_sha256']
    a = json.loads(passp.read_text())
    assert sha(NV1 / 'run_nv1.py') == a['source_sha256']
    assert sha(NV1 / 'results/P2P_CAPTURE_REPLAY_AUDIT.json') == a['audit_sha256']
    nums = json.loads((out / 'P2P_SINGULAR_NUMERICS.json').read_text())
    assert len(nums) == 40
    assert all((all(r['capture_replay_exact']) and max((max(x) for x in r['residuals'][:4])) <= 0.0002 for r in nums))
    for name, count in [('P2P_SINGULAR_SPECTRA.csv', 23040), ('P2P_PAIRWISE_PROJECTOR_GEOMETRY.csv', 780), ('P2P_TOP8_PROJECTOR_SUMMARY.csv', 40), ('P2P_COMMUTATOR_SUMMARY.csv', 12480)]:
        with (out / name).open() as f:
            assert sum((1 for _ in csv.DictReader(f))) == count, name
    pooled = np.load(out / 'P2P_POOLED_SUBSPACE.npz', allow_pickle=False)
    qu, qm = (pooled['Q_U'], pooled['Q_M'])
    assert qu.shape == (576, 24) and qm.shape == (884736, 24)
    orthU = float(np.linalg.norm(qu.T @ qu - np.eye(24)))
    orthM = float(np.linalg.norm(qm.T @ qm - np.eye(24)))
    assert max(orthU, orthM) < 1e-09
    matrices = np.load(out / 'P2P_COMMON_OPERATOR_MATRICES.npz', allow_pickle=False)
    ks = matrices['K']
    assert ks.shape == (40, 24, 24)
    summary = json.loads((out / 'P2P_MEAN_NOISE_SUMMARY.json').read_text())
    mean = ks[:32].mean(0)
    rho = float(np.sqrt(((ks[:32] - mean) ** 2).sum((1, 2)).mean()) / np.linalg.norm(mean))
    assert abs(rho - summary['rho_Xi']) < 1e-12
    u, s, vh = np.linalg.svd(mean)
    blocks = u[:, :8].T @ ks[:32] @ vh.T[:, :8]
    diag = np.zeros_like(blocks)
    ix = np.arange(8)
    diag[:, ix, ix] = blocks[:, ix, ix]
    roff = float(np.linalg.norm(blocks - diag) / np.linalg.norm(blocks))
    assert abs(roff - summary['R_off']) < 1e-12
    zero = np.zeros((24, 24))
    A = np.block([[zero, ks[0].T], [ks[0], zero]])
    B = np.block([[zero, ks[1].T], [ks[1], zero]])
    chi = float(np.linalg.norm(A @ B - B @ A) / (2 * np.linalg.norm(A) * np.linalg.norm(B)))
    with (out / 'P2P_PAIRWISE_PROJECTOR_GEOMETRY.csv').open() as f:
        row = next(csv.DictReader(f))
    assert abs(chi - float(row['chi_C'])) < 1e-14
    bs = json.loads((out / 'P2P_BOOTSTRAP_SUMMARY.json').read_text())
    samples = np.load(out / 'P2P_BOOTSTRAP_RESAMPLES.npz', allow_pickle=False)
    for name, x in bs['statistics'].items():
        assert samples[name].shape == (20000,) and np.isfinite(samples[name]).all()
        lo, hi = np.percentile(samples[name], [2.5, 97.5])
        assert lo == x['lower95'] and hi == x['upper95']
    return dict(status='PASS', primary_before_FE_order_verified=True, decision_and_NV1_seals_verified=True, executed_source_hashes_unchanged=True, forty_batch_replay_and_residual_gates_verified=True, required_CSV_row_counts_verified=True, pooled_orthogonality_U=orthU, pooled_orthogonality_M=orthM, independent_rho_Xi=rho, independent_R_off=roff, independent_explicit_lift_first_pair_chi_C=chi, all_20000_resamples_and_percentile_CIs_verified=True)

def main():
    qa = verify_results()
    print('Independent output verification PASS', flush=True)
    audit = json.loads((SCI / 'results/P2P_ASSET_AUDIT.json').read_text())
    qa['exact_authority_assets_rehashed'] = verify_original_assets(audit)
    for p in (SCI / 'results').iterdir():
        if p.is_file():
            copy(p, DEST / p.name)
    for p in (NV1 / 'results').iterdir():
        if p.is_file():
            copy(p, DEST / 'NV1' / p.name)
    copy(NV1 / 'results/P2P_NV1_NUMERICAL_VALIDATION_PASS.json', DEST / 'P2P_NV1_NUMERICAL_VALIDATION_PASS.json')
    for p in PKG.rglob('*'):
        if p.is_file():
            copy(p, DEST / 'authority' / p.relative_to(PKG))
    copy(Path(_p1_path('path_8ea7adbae811')), DEST / 'authority/USER_EXECUTION_TICKET.md')
    for name, p in [('run_nv1.py', NV1 / 'run_nv1.py'), ('run_science.py', SCI / 'run_science.py'), ('finalize.py', Path(__file__))]:
        copy(p, DEST / 'source' / name)
    for stage, base in [('NV1', NV1), ('fresh_P2P', SCI)]:
        copy(base / 'execution.log', DEST / 'logs' / f'{stage}_execution.log')
        for p in base.glob('capture_*.pt'):
            copy(p, DEST / 'captures' / stage / p.name)
    copy(NV1 / 'frozen_probes.npz', DEST / 'inputs/frozen_probes.npz')
    for p in SCI.glob('right_*.npy'):
        copy(p, DEST / 'right_modes' / p.name)
    for p in SCI.glob('singular_*.npz'):
        copy(p, DEST / 'singular_details' / p.name)
    for p in (NV1 / 'nv0_source').rglob('*'):
        if p.is_file():
            copy(p, DEST / 'historical/NV0_source' / p.relative_to(NV1 / 'nv0_source'))
    prior = Path(_p1_path('path_aa9fd874df96'))
    for p in prior.iterdir():
        if p.is_file() and (p.name in {'P2P_STOP_REPRODUCIBILITY_SOURCE.zip', 'SHA256SUMS_P2P.txt'} or p.suffix in {'.json', '.md'}):
            copy(p, DEST / 'historical/NV0_outputs' / p.name)
    hist = Path(_p1_path('path_08e09cd8aadb'))
    for name in ['P2H_PREFLIGHT_SEAL.json', 'P2H_DATA_BANK_SEAL.json', 'P2H_ASSET_REUSE_AUDIT.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz', 'SHA256SUMS_P2H.txt']:
        copy(hist / name, DEST / 'historical/P2H' / name)
    for n, r in audit['banks'].items():
        copy(Path(r['path']), DEST / 'inputs/banks' / f'{n}.npy')
    for r in audit['model_files']:
        copy(Path(r['path']), DEST / 'inputs/model' / Path(r['path']).name)
    model = Path(audit['model_files'][0]['path']).parent
    copy(model / 'MODEL_ASSET.json', DEST / 'inputs/model/MODEL_ASSET.json')
    write('P2P_FINAL_VERIFICATION.json', qa)
    write('P2P_ENVIRONMENT_INVENTORY.json', dict(recorded_environment=audit['environment'], installed_packages=sorted([dict(name=d.metadata['Name'], version=d.version) for d in importlib.metadata.distributions()], key=lambda x: x['name'].lower())))
    completion = json.loads((DEST / 'P2P_COMPLETION.json').read_text())
    decision = json.loads((DEST / 'P2P_PRIMARY_DECISION.json').read_text())
    fe = json.loads((DEST / 'P2P_FE_REPLICATION.json').read_text())
    ci = decision['statistics']
    nums = json.loads((DEST / 'P2P_SINGULAR_NUMERICS.json').read_text())
    report = f"# P2P execution report\n\nNV1 passed on exactly B0_1 and FE_1; its hashed PASS artifact authorized a separate fresh P2P execution. No numerical-validation result entered the scientific statistics. The exact NV0 model commit, ten sealed NPY arrays, F0 triplets and environment were re-audited. All 32 primary and eight FE views match the historical SHA-256 identities.\n\nPrimary result: `{decision['primary_label']}`. FE result: `{fe['status']}`. The primary decision was written and hash sealed before FE science opened; the event ledger and independent final verification establish that order.\n\nCPU, eight threads, deterministic algorithms. Full-model forward and capture float32; post-capture operator algebra, reduced Gram, actions, right modes, pooled algebra, geometry and commutators float64. No dense full cross matrix was formed. No training, optimizer, network, download, new bank, replacement batch/probe, or threshold change occurred.\n\nFresh execution elapsed {completion['elapsed_seconds']:.6f} seconds, excluding source preparation, independent verification and packaging. Scientific batch count 40. Full-model forwards 368 (48 preflight + 320 fresh scientific capture/replay); captured sequences 336; four-sequence mixed HVP checks in each direction on each of two preflight batches. All 40 scientific captures replayed bitwise; maximum retained top-four singular residual {max((max((max(z) for z in x['residuals'][:4])) for x in nums)):.6g}.\n\nThe 20,000-resample percentile bootstrap uses only primary batch identities and seed 2026100814. Every resample includes all 496 positional pairs; duplicate identities use the exact self-value zero. The primary pooled basis remains fixed, and the mean singular basis is recomputed for each resample. These implementation conventions were in the source hashed at process start and did not change after outcomes.\n\nThe source snapshot, all exact model files, sealed banks, F0, probes, float32 captures, float64 right modes, bootstrap resamples and historical authority evidence are included. Python/library binaries are external; the exact original environment still exists and its full package inventory is recorded. Historical artifacts were read and copied without modification; their hashes were rechecked at finalization.\n\nSee P2P_FINAL_VERIFICATION.json for independent checks of seals, ordering, dimensions, row counts, pooled orthogonality, mean/noise and rotational statistics, explicit lifted commutator, and all bootstrap percentiles. SHA256SUMS_P2P.txt covers every package file except itself; the ZIP has a separate SHA-256 checksum.\n"
    (DEST / 'P2P_EXECUTION_REPORT.md').write_text(report)
    rows = '\n'.join((f"| {name} | {ci[name]['point']:.9g} | [{ci[name]['lower95']:.9g}, {ci[name]['upper95']:.9g}] |" for name in ['D_pair', 'D_U', 'D_M', 'rho_Xi', 'Gamma24', 'tau24', 'R_off', 'chi_C', 'E4']))
    closure = f"# P2P scientific closure\n\nFrozen primary verdict: `{decision['primary_label']}`.\n\n| Primary statistic | Point | Percentile 95% CI |\n|---|---:|---:|\n{rows}\n\nThe full frozen conjunction evaluation is in P2P_PRIMARY_DECISION.json. No cutoff, batch identity, retained rank (8), comparison rank (4), pooled dimension (24), bootstrap rule, or branch condition was tuned.\n\nFE replication: `{fe['status']}`. Median FE-to-primary projector dispersion {fe['FE_median_primary_dispersion']:.9g}; FE rotation in the primary mean basis {fe['FE_rotational_index_in_frozen_primary_mean_basis']:.9g}; median intrinsic FE-to-primary commutator {fe['FE_median_primary_commutator']:.9g}; median pooled capture {fe['FE_median_Gamma24']:.9g}; median miss {fe['FE_median_tau24']:.9g}. FE cannot change the sealed primary label. For an unresolved primary label, the protocol supplies no branch-specific directional replication claim; FE is reported as mixed with all component measurements.\n\nNV0/NV1 are numerical provenance only. This closure describes the local batch-conditioned operator at the exactly symmetric state. It establishes no full-training stochastic Lyapunov exponent, diffusion limit, long-time nonlinear closure, native OLMoE formation mechanism, improved training utility, or Q3 recommendation. Existing historical scientific decisions remain unchanged.\n"
    (DEST / 'P2P_SCIENTIFIC_CLOSURE.md').write_text(closure)
    index = dict(status='COMPLETE', executed_source_sha256={n: sha(DEST / 'source' / n) for n in ['run_nv1.py', 'run_science.py']}, source_extraction='Only PrivateMoE, capture, CrossOperator from exact P2H preflight; historical bank-generation/training code never executed', captured_token_files=[dict(path=str(p.relative_to(DEST)), sha256=sha(p)) for p in sorted((DEST / 'captures').rglob('*.pt'))], right_vector_files=[dict(path=str(p.relative_to(DEST)), sha256=sha(p)) for p in sorted((DEST / 'right_modes').glob('*.npy'))], all_exact_local_model_files_included=True, all_sealed_banks_included=True, all_bootstrap_resamples_included=True, original_environment=audit['environment']['python'], primary_decision_sha256=sha(DEST / 'P2P_PRIMARY_DECISION.json'), NV1_pass_sha256=sha(DEST / 'P2P_NV1_NUMERICAL_VALIDATION_PASS.json'))
    write('P2P_SOURCE_INDEX.json', index)
    (DEST / 'README.md').write_text('# Complete NV1 + fresh P2P package\n\nStart with P2P_EXECUTION_REPORT.md, P2P_SCIENTIFIC_CLOSURE.md, P2P_PRIMARY_DECISION.json and P2P_FE_REPLICATION.json. All Section-13 deliverables are at the root. NV1 validation evidence is under NV1/.\n\nThe package includes exact source snapshots, authority, historical evidence, model snapshot, banks, F0, probes, captures, right modes and bootstrap resamples. SHA256SUMS_P2P.txt covers the full directory. Validate from this directory with `shasum -a 256 -c SHA256SUMS_P2P.txt`.\n\nThe executed sources preserve the original machine paths for forensic fidelity. On the originating machine, reproduce in two fresh directories using the indexed original interpreter and those exact sources. For relocation, use inputs/model, inputs/banks, historical/P2H and historical/NV0_source as the corresponding path replacements, preserving every sealed hash and the original versions. Source-only path adaptation is necessary; no downloads or reconstructed inputs are needed. The original environment and installed package versions are in P2P_ENVIRONMENT_INVENTORY.json.\n\nThe additional scientific captures and right vectors are included rather than kept in an external worktree. No historical artifact was modified.\n')
    fs = manifest(DEST, 'SHA256SUMS_P2P.txt')
    for line in (DEST / 'SHA256SUMS_P2P.txt').read_text().splitlines():
        h, n = line.split(maxsplit=1)
        assert sha(DEST / n) == h, n
    print(f'Directory manifest verified: {len(fs)} files', flush=True)
    archive = DEST.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():
                z.write(p, str(Path(DEST.name) / p.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        error = z.testzip()
        assert error is None, error
    checksum = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum + '  ' + archive.name + '\n')
    print(json.dumps(dict(directory=str(DEST), archive=str(archive), zip_sha256=checksum, files=len(fs) + 1, archive_bytes=archive.stat().st_size, archive_CRC_verified=True)), flush=True)
if __name__ == '__main__':
    main()
