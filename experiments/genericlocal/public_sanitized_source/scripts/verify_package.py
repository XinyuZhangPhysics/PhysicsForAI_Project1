from pathlib import Path
import csv, hashlib, json, math, sys
import numpy as np
from generic_core import sha256_file, array_sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
MANIFEST = ROOT / 'PACKAGE_MANIFEST.json'
MANIFEST_SHA = ROOT / 'PACKAGE_MANIFEST.sha256'
SEAL = ROOT / 'SCIENTIFIC_SEAL.json'
PROTOCOL = ROOT / 'protocol.json'


def fail(msg: str):
    raise SystemExit(msg)


def verify_package_manifest():
    expected_manifest_sha = MANIFEST_SHA.read_text().strip().split()[0]
    actual_manifest_sha = sha256_file(MANIFEST)
    if actual_manifest_sha != expected_manifest_sha:
        fail(f'PACKAGE_MANIFEST hash mismatch: {actual_manifest_sha} != {expected_manifest_sha}')
    m = json.loads(MANIFEST.read_text())
    listed = {row['path']: row for row in m['files']}
    for rel, row in listed.items():
        path = ROOT / rel
        if not path.is_file():
            fail(f'PACKAGE_MANIFEST missing file: {rel}')
        actual = sha256_file(path)
        if actual != row['sha256'] or path.stat().st_size != int(row['bytes']):
            fail(f'PACKAGE_MANIFEST mismatch: {rel}')
    actual_files = set()
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith('results/') or rel in {'PACKAGE_MANIFEST.json', 'PACKAGE_MANIFEST.sha256'}:
            continue
        actual_files.add(rel)
    if actual_files != set(listed):
        fail(f'PACKAGE_MANIFEST file-set mismatch: unlisted={sorted(actual_files-set(listed))}, missing={sorted(set(listed)-actual_files)}')
    if any('__pycache__' in rel or rel.endswith('.pyc') for rel in actual_files):
        fail('Generated Python cache files are prohibited in the package')
    return actual_manifest_sha


def verify_scientific_seal():
    seal = json.loads(SEAL.read_text())
    if seal.get('status') != 'SEALED' or not seal.get('scientific_protocol_unchanged_from_v1'):
        fail('SCIENTIFIC_SEAL is not authoritative')
    for rel, expected in seal['immutable_scientific_file_sha256'].items():
        path = ROOT / rel
        if not path.is_file():
            fail(f'SCIENTIFIC_SEAL missing file: {rel}')
        if sha256_file(path) != expected:
            fail(f'SCIENTIFIC_SEAL hash mismatch: {rel}')
    if sha256_file(PROTOCOL) != seal['protocol_sha256']:
        fail('protocol.json does not match SCIENTIFIC_SEAL')
    return seal


def verify_historical_authority(p, seal):
    frozen_path = ROOT / 'inputs/P2J_FROZEN_CELL_PREDICTIONS.csv'
    completion_path = ROOT / 'inputs/P2J_TRAINING_COMPLETION.json'
    gate_path = ROOT / 'inputs/P2J_AMPLITUDE_GATE.csv'
    frozen = list(csv.DictReader(frozen_path.open()))
    target = {}
    for row in frozen:
        eps = float(row['epsilon']); r = float(row['r_eta'])
        if abs(abs(eps)-0.08) < 1e-12 and abs(r-1.0) < 1e-12:
            target[round(eps, 2)] = row
    if set(target) != {-0.08, 0.08}:
        fail('Historical P2J frozen predictions do not contain both epsilon=+-0.08, r_eta=1 cells')
    for eps, row in target.items():
        if abs(float(row['a0']) - float(p['a0_aligned_raw'])) > 1e-15:
            fail(f'Historical a0 mismatch at epsilon={eps}')
        if abs(float(row['mu']) - float(p['mu'])) > 1e-15:
            fail(f'Historical mu mismatch at epsilon={eps}')
        if abs(float(row['eta_M']) - float(p['eta_X'])) > 1e-12 or abs(float(row['eta_U']) - float(p['eta_X'])) > 1e-12:
            fail(f'Historical learning-rate mismatch at epsilon={eps}')
        if int(row['N_cell']) < int(p['horizon']):
            fail(f'Frozen horizon is not locality-safe at epsilon={eps}')
    if int(target[0.08]['N_cell']) != int(p['horizon']) or int(target[-0.08]['N_cell']) != 32:
        fail('Historical horizon provenance mismatch: expected stable=32, unstable=28')

    completion = json.loads(completion_path.read_text())
    found = {}
    for row in completion['cell_status']:
        eps=float(row['epsilon']); r=float(row['r_eta'])
        if abs(abs(eps)-0.08)<1e-12 and abs(r-1.0)<1e-12:
            found[round(eps,2)] = row
    if set(found) != {-0.08,0.08}:
        fail('Historical completion authority missing target cells')
    for eps,row in found.items():
        if row['status'] != 'HORIZON_COMPLETE' or row.get('violation') is not None:
            fail(f'Historical locality completion failed at epsilon={eps}')
        if int(row['completed_update']) < int(p['horizon']):
            fail(f'Historical completed horizon too short at epsilon={eps}')

    gates = [r for r in csv.DictReader(gate_path.open()) if abs(abs(float(r['epsilon']))-0.08)<1e-12 and abs(float(r['r_eta'])-1.0)<1e-12]
    if len(gates) != 4:
        fail(f'Expected four sign-resolved historical amplitude-gate rows, found {len(gates)}')
    for row in gates:
        if abs(float(row['a0']) - float(p['a0_aligned_raw'])) > 1e-15 or row['passed'].strip().lower() != 'true':
            fail('Historical amplitude gate does not authorize frozen a0')
    return {
        'a0': float(p['a0_aligned_raw']),
        'stable_historical_horizon': int(target[-0.08]['N_cell']),
        'unstable_historical_horizon': int(target[0.08]['N_cell']),
        'common_frozen_horizon': int(p['horizon']),
        'amplitude_gate_rows_passed': len(gates),
    }


def main():
    manifest_sha = verify_package_manifest()
    seal = verify_scientific_seal()
    p = json.loads(PROTOCOL.read_text())
    authority = verify_historical_authority(p, seal)

    checks = {
        ROOT/'inputs/F0.npy': p['F0_file_sha256'],
        ROOT/'inputs/P2H_CROSS_HESSIAN_SPECTRUM.npz': p['spectrum_file_sha256'],
    }
    for path, expected in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            fail(f'HASH FAIL {path.name}: {actual} != {expected}')
    F0 = np.load(ROOT/'inputs/F0.npy', allow_pickle=False)
    if F0.shape != (32,256) or str(F0.dtype) != 'int64':
        fail(f'F0 shape/dtype mismatch: {F0.shape} {F0.dtype}')
    if array_sha256(F0) != p['F0_array_sha256']:
        fail('F0 array SHA256 mismatch')
    z = np.load(ROOT/'inputs/P2H_CROSS_HESSIAN_SPECTRUM.npz', allow_pickle=False)
    if z['u'].shape != (576,4) or z['v'].shape != (576,1536,4):
        fail('Archived singular-vector shapes mismatch')
    if abs(float(z['s'][0])-p['mu']) > 1e-12:
        fail('mu/s1 mismatch')
    if abs(p['eta_X']*p['mu']-0.2) > 1e-12:
        fail('eta_X*mu is not 0.2')
    print('PACKAGE_VERIFY PASS')
    print(json.dumps({
        'package_manifest_sha256': manifest_sha,
        'scientific_seal_sha256': sha256_file(SEAL),
        'F0_shape': list(F0.shape),
        's1': float(z['s'][0]),
        'eta_X_mu': p['eta_X']*p['mu'],
        'historical_authority': authority,
    }, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
