# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Deterministic artifact validation and final immutable SHA-256 manifest."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, csv, hashlib, zipfile
import numpy as np
import pyarrow.parquet as pq
from audit_assets import ROOT, OUT, PKG, sha

def main():
    checks = []

    def check(condition, name):
        assert condition, name
        checks.append(name)
    required = 'P2H_ASSET_REUSE_AUDIT.json P2H_DATA_BANK_SEAL.json P2H_CROSS_HESSIAN_SPECTRUM.npz P2H_LINEARIZATION_REPORT.md P2H_PREDICTIONS.json P2H_PREFLIGHT_SEAL.json P2H_TRAJECTORIES.parquet P2H_CHECKPOINT_INDEX.json P2H_GROWTH_RATES.csv P2H_STATIC_THRESHOLD.csv P2H_KINETIC_PREDICTION.csv P2H_ONSET_SCALING.csv P2H_MODE_LOCKING.csv P2H_BOOTSTRAP_DRAWS.npz P2H_PRIMARY_DECISION.json P2H_PRIMARY_REPORT.md P2H_LOCAL_RESPONSE_GEOMETRY.csv P2H_LOCAL_RESPONSE_GEOMETRY.npz P2H_SECONDARY_REPORT.md'.split()
    check(all(((OUT / n).is_file() for n in required)), 'all frozen stage deliverables present')
    for name, field in [('P2H_PREFLIGHT_SEAL.json', 'artifact_sha256'), ('P2H_PRIMARY_DECISION.json', 'primary_artifact_sha256'), ('P2H_SECONDARY_SEAL.json', 'artifact_sha256')]:
        rec = json.loads((OUT / name).read_text())
        check(all((sha(OUT / n) == h for n, h in rec[field].items())), name + ' referenced hashes match')
    exe = json.loads((OUT / 'P2H_EXECUTION_CODE_FREEZE.json').read_text())
    check(all((sha(n) == h for n, h in exe['source_sha256'].items())), 'execution source freeze matches')
    pre = json.loads((OUT / 'P2H_PREFLIGHT_SEAL.json').read_text())
    check(pre['training_updates_before_seal'] == 0 and pre['status'] == 'PASS', 'preflight fail-closed order')
    rows = pq.read_table(OUT / 'P2H_TRAJECTORIES.parquet').to_pylist()
    check(len(rows) == 7800, '120 arms x 65 logged states')
    ids = {(r['seed'], r['arm'], r['update']) for r in rows}
    check(ids == {(j, k, n) for j in range(8) for k in range(15) for n in range(65)}, 'exact grid and horizon with no additions')
    check(all((all((np.isfinite(r[n]) for n in ['M_norm', 'U_norm', 'm1', 'u1', 'A1', 'F_lead', 'C_plus'])) for r in rows)), 'finite logged private observables')
    index = json.loads((OUT / 'P2H_CHECKPOINT_INDEX.json').read_text())
    check(index['status'] == 'COMPLETE' and len(index['checkpoints']) == 1080, 'all 1080 checkpoint records')
    for r in index['checkpoints']:
        check(sha(r['path']) == r['sha256'], f"checkpoint {r['seed']}/{r['arm']}/{r['update']} hash")
        with np.load(r['path']) as z:
            assert z['M'].shape == (576, 1536) and z['U'].shape == (576,) and (z['M'].dtype == np.float32) and (z['U'].dtype == np.float32)
    check(True, 'all checkpoints private-only float32 shape verified')
    banks = json.loads((OUT / 'P2H_DATA_BANK_SEAL.json').read_text())['banks']
    hashes = []
    for name, rec in banks.items():
        check(sha(rec['path']) == rec['file_sha256'], 'bank ' + name + ' file hash')
        hashes.extend((x['token_array_sha256'] for x in rec['sequences']))
    check(len(hashes) == 2112 and len(set(hashes)) == 2112, 'all banks disjoint by sequence token array hash')
    spec = np.load(OUT / 'P2H_CROSS_HESSIAN_SPECTRUM.npz')
    check(len(spec['s']) >= 4 and spec['relative_residuals'][0] <= 0.0001 and (int(spec['iterations']) >= 24), 'cross Hessian frozen numerical gates')
    pred = json.loads((OUT / 'P2H_PREDICTIONS.json').read_text())
    check(len(pred['predictions']) == 15 and pred['primary_runs'] == 120, 'prediction grid frozen')
    for p in pred['predictions']:
        a = np.array(p['A'])
        e = np.array(p['e_plus'])
        check(np.linalg.norm(a @ e - p['sigma_plus'] * e) <= 1e-12, f"predicted eigenpair {p['epsilon']}/{p['r_eta']}")
    decision = json.loads((OUT / 'P2H_PRIMARY_DECISION.json').read_text())
    check(decision['objective_susceptibility_computed'] is False and decision['Q3_recommendation'] is False, 'scientific interpretation prohibitions respected')
    geom = np.load(OUT / 'P2H_LOCAL_RESPONSE_GEOMETRY.npz')
    check(geom['grams'].shape == (96, 2, 2), 'H3 exact 96 saved checkpoints')
    check(np.all(geom['eigenvalues'] >= -1e-20), 'local response Gram positivity')
    audit = json.loads((OUT / 'P2H_ASSET_REUSE_AUDIT.json').read_text())
    train = json.loads((OUT / 'P2H_TRAINING_COMPLETION.json').read_text())
    check(train['training_tokens'] == 7864320, 'training token budget respected')
    report = '# P2H execution report\n\n'
    report += 'Status: **COMPLETE**. H0 preflight passed, H1 ran exactly 120 arms, H2 sealed the componentwise primary decision, and H3 measured 96 saved-checkpoint local response Grams after the primary seal.\n\n'
    report += 'Primary verdict: **' + decision['combined'] + '**.\n\n' + '\n'.join((f'- {k}: `{v}`' for k, v in decision['components'].items())) + '\n\n'
    report += 'Frozen authority: user-supplied Project1_P2H_FormationBridge_Codex_Package_v1, protocol SHA-256 ff252991f9373763472c151757edced32e9d0336a67f56eb66f70dbbcc8f6ca1. Later migration bundles discovered during the local asset audit were not adopted as this run’s scientific authority.\n\n'
    report += 'Model: SmolLM2-135M commit 28e66ca6931668447a3bac213f23d990ad3b0e2b; dataset: DKYoon/SlimPajama-6B revision b5f90f419b7489cdba26fdbc8c022fcb5562f968. Canonical model files and one exact cached training shard were hash verified and reused in place. Other cached shards were inventoried by pinned tree and size but were not consumed.\n\n'
    report += '| Reuse counter | Count |\n|---|---:|\n' + ''.join((f'| {k} | {audit[k]} |\n' for k in ['MODEL_FILES_REUSED', 'MODEL_FILES_NEWLY_DOWNLOADED', 'DATA_FILES_REUSED', 'DATA_FILES_NEWLY_DOWNLOADED', 'BYTES_NEWLY_DOWNLOADED'])) + '\n'
    report += 'Implementation follows the historical g=1 SwiGLU full-copy upcycling equations with shared frozen common storage and exactly two private blocks, M and U. Historical J1 source actually addressed zero-based layer 19; this protocol explicitly requires 20, so this run uses 20. No historical S2/J1 files were modified. No model-family substitution or network download occurred.\n\n'
    report += 'Deterministic CPU float32 derivatives, float64 scalar reductions, no autocast, plain SGD, zero momentum/weight decay, and no clipping. Frozen prefix computation was reused across arms only after full-model output and gradient equivalence (all relative errors exactly zero). Small-step calibration was performed on F0, the bank defining C, with the specified cell and seed; the original LR factor passed. Exact preflight gradient replay was bitwise equal. An initial predictions JSON serialization error from zero-imaginary complex eigensolver outputs was repaired before any training, using the protocol’s analytic eigenvalue formula and its real eigenvector; H0 was rerun and passed.\n\n'
    report += f"120 runs x 64 updates x 1024 tokens = 7,864,320 training tokens. 7800 logged states and 1080 immutable private checkpoints. Training elapsed time: {train['elapsed_seconds']:.1f} s. The next-token loss averages the 255 valid next-token targets of each 256-token sequence; the protocol’s input-token accounting remains 1024/update.\n\n"
    report += 'Analysis conventions fixed before science outcomes were inspected: seed fits use updates 8–32 and the declared R² gate; arm median and mean summaries use all eight seeds, with arm resolution requiring at least six resolved fits. Static zero crossings retain extrapolations rather than clamping to the grid. The secondary probe is the target MLP Euclidean output response in raw frozen (M,U) coordinates. No objective susceptibility was evaluated.\n\n'
    report += 'Interpretation ceiling: controlled real-model bridge only. Negative/partial/unresolved components remain as measured. No native OLMoE onset, architecture universality, utility claim, or Q3 recommendation follows.\n\n'
    report += 'Reproducibility sources and original authority are in P2H_REPRODUCIBILITY_SOURCE.zip. Original large model/data bytes are referenced in the audit and not duplicated. Data token banks and initial private seeds are included as reproducibility inputs in the source archive. Checkpoints remain in outputs/checkpoints and are indexed and hashed.\n'
    (OUT / 'P2H_EXECUTION_REPORT.md').write_text(report)
    validation = dict(status='PASS', checks_passed=len(checks), summary_checks=[c for c in checks if not c.startswith('checkpoint ')], required_deliverables=required, trajectory_rows=7800, checkpoints_verified=1080, secondary_grams=96)
    (OUT / 'P2H_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
    bundle = OUT / 'P2H_REPRODUCIBILITY_SOURCE.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / 'work/p2h_source').glob('*.py')):
            z.write(p, 'source/' + p.name)
        for p in sorted(PKG.iterdir()):
            if p.is_file():
                z.write(p, 'authority/' + p.name)
        for p in sorted((ROOT / 'work/p2h_banks').glob('*.npy')):
            z.write(p, 'inputs/banks/' + p.name)
        z.write(ROOT / 'work/p2h_private_seeds.pt', 'inputs/p2h_private_seeds.pt')
        for n in ['p2h_preflight.log', 'p2h_benchmark.log', 'p2h_primary.log', 'p2h_analysis.log', 'p2h_secondary.log', 'p2h_preflight_serialization_repair.json']:
            p = ROOT / 'work' / n
            if p.exists():
                z.write(p, 'logs/' + n)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS_P2H.txt']
    manifest = OUT / 'SHA256SUMS_P2H.txt'
    manifest.write_text(''.join((f'{sha(p)}  {p.relative_to(OUT)}\n' for p in sorted(files))))
    for p in files + [manifest]:
        p.chmod(292)
    for line in manifest.read_text().splitlines():
        h, n = line.split(maxsplit=1)
        assert sha(OUT / n) == h, n
    print('FINAL PASS:', len(checks), 'checks;', len(files), 'immutable files hashed; all final manifest entries reverified', flush=True)
if __name__ == '__main__':
    main()
