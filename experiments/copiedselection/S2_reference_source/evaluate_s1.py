# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
SEEDS = tuple(range(2026100101, 2026100109))
ARMS = ('S2', 'S4', 'S8')
G = np.array([2.0, 4.0, 8.0])
X = np.log(G)
FLOOR = 1e-20
BOOT_SEED = 2026100199
BOOT_N = 20000

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def load_response(arm, seed, t):
    return json.loads((OUT / 'audit_responses' / f'{arm}_seed_{seed}_t{t}.json').read_text())

def seed_components(values):
    num = np.zeros(8, dtype=np.float64)
    den = np.zeros(8, dtype=np.float64)
    for i in range(8):
        row = values[i]
        valid = np.isfinite(row) & (row > 0)
        if np.sum(valid) < 2:
            continue
        xx = X[valid]
        yy = np.log(row[valid])
        num[i] = float(np.dot(xx - xx.mean(), yy - yy.mean()))
        den[i] = float(np.square(xx - xx.mean()).sum())
    return (num, den)

def slope(values):
    num, den = seed_components(values)
    if den.sum() <= 0:
        return None
    return float(num.sum() / den.sum())

def bootstrap(values, indices):
    num, den = seed_components(values)
    bnum = num[indices].sum(axis=1)
    bden = den[indices].sum(axis=1)
    valid = bden > 0
    draws = np.full(len(indices), np.nan, dtype=np.float64)
    draws[valid] = bnum[valid] / bden[valid]
    clean = draws[np.isfinite(draws)]
    if len(clean) != BOOT_N:
        return {'resolved': False, 'valid_replicates': len(clean), 'ci95': None}
    low, high = np.quantile(clean, [0.025, 0.975], method='linear')
    return {'resolved': True, 'valid_replicates': len(clean), 'ci95': [float(low), float(high)], 'bootstrap_mean': float(clean.mean())}

def primary():
    if not (OUT / 'AUDIT_OPENING_RECORD.json').exists():
        raise RuntimeError('AUDIT_NOT_OPENED')
    integrity = json.loads((OUT / 'TRAINING_INTEGRITY.json').read_text())
    if integrity['status'] != 'PASS':
        raise RuntimeError('TRAINING_INTEGRITY_FAILED')
    if (OUT / 'FORMAL_LABEL.json').exists():
        raise RuntimeError('PRIMARY_ALREADY_FROZEN')
    grid = {name: np.full((8, 3), np.nan, dtype=np.float64) for name in ('Q_C', 'K_C_0', 'P_B', 'Theta_R')}
    runs = []
    for i, seed in enumerate(SEEDS):
        for j, arm in enumerate(ARMS):
            r = load_response(arm, seed, 64000)
            assert r['horizon_nonpadding_tokens'] == 64000 and r['seed'] == seed and (r['arm'] == arm)
            for name in grid:
                value = r[name]
                if value is not None and math.isfinite(value) and (value > FLOOR):
                    grid[name][i, j] = value
            runs.append({'seed': seed, 'arm': arm, 'g': int(G[j]), 'Q_C': r['Q_C'], 'K_C_0': r['K_C_0'], 'P_B': r['P_B'], 'Theta_R': r['Theta_R'], 'resolved_positive': r['resolved_positive'], 'response_projector_max_abs_error': r['response_projector_max_abs_error'], 'factorization_abs_error': r['factorization_abs_error'], 'response_file_sha256': sha(OUT / 'audit_responses' / f'{arm}_seed_{seed}_t64000.json')})
    resolved = np.isfinite(grid['Q_C']) & np.isfinite(grid['K_C_0']) & np.isfinite(grid['P_B'])
    counts = {arm: int(resolved[:, j].sum()) for j, arm in enumerate(ARMS)}
    resolved_pass = all((n >= 7 for n in counts.values()))
    h1_rows = []
    for seed in SEEDS:
        for t in (0, 16000, 64000, 256000):
            r = load_response('S1', seed, t)
            powers = r['B_coordinate_powers']
            assert len(powers) == 12
            h1_rows.append({'seed': seed, 'horizon': t, 'max_B_coordinate_power': max(powers), 'all_12_below_floor': all((math.isfinite(v) and v <= FLOOR for v in powers)), 'response_file_sha256': sha(OUT / 'audit_responses' / f'S1_seed_{seed}_t{t}.json')})
    h1_pass = all((x['all_12_below_floor'] for x in h1_rows))
    primary_data = {'status': 'FROZEN', 'primary_horizon_nonpadding_tokens': 64000, 'resolution_power_floor': FLOOR, 'runs': runs, 'resolved_count_by_arm': counts, 'resolved_requirement_at_least_7_of_8': resolved_pass, 'S1_H1_rows': h1_rows, 'S1_H1_pass': h1_pass}
    write('AUDIT_PRIMARY_64K.json', primary_data)
    central = {name: slope(grid[name]) for name in ('Q_C', 'Theta_R', 'P_B')}
    geom = json.loads((OUT / 'AUDIT_GEOMETRY.json').read_text())
    kval = np.array([geom['arms'][arm]['K_C_0'] for arm in ARMS], dtype=np.float64)
    assert np.all(kval > FLOOR)
    beta_k = float(np.polyfit(X, np.log(kval), 1)[0])
    regressions = {'status': 'FROZEN', 'method': 'seed fixed-effect OLS on log values; within each paired seed center log(g) and log(response) over resolved arms; unresolved observations omitted without regularization', 'primary_horizon_nonpadding_tokens': 64000, 'central_slopes': {'beta_Q': central['Q_C'], 'beta_Theta': central['Theta_R'], 'beta_B': central['P_B']}, 'descriptive_K_C_0_by_arm': {arm: float(kval[j]) for j, arm in enumerate(ARMS)}, 'descriptive_beta_K': beta_k, 'resolved_count_by_arm': counts}
    write('PRIMARY_REGRESSIONS.json', regressions)
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, 8, size=(BOOT_N, 8), dtype=np.int16)
    index_path = OUT / 'BOOTSTRAP_INDICES.npy'
    np.save(index_path, indices)
    bo = {name: bootstrap(grid[name], indices) for name in ('Q_C', 'Theta_R', 'P_B')}
    boot = {'status': 'FROZEN', 'replicates': BOOT_N, 'seed': BOOT_SEED, 'resampling_unit': 'whole paired seed triplet; eight seed indices drawn with replacement per replicate', 'indices_sha256': sha(index_path), 'percentile_method': 'numpy quantile linear 2.5%, 97.5%', 'beta_Q': bo['Q_C'], 'beta_Theta': bo['Theta_R'], 'beta_B': bo['P_B']}
    write('BOOTSTRAP.json', boot)
    h2 = bool(bo['Theta_R']['resolved'] and bo['Theta_R']['ci95'][1] < 0)
    h3 = bool(bo['Q_C']['resolved'] and bo['Q_C']['ci95'][0] > 0)
    h4 = bool(bo['P_B']['resolved'] and bo['P_B']['ci95'][0] > 0)
    hypotheses = {'H1': 'PASS' if h1_pass else 'FAIL', 'H2': 'PASS' if h2 else 'FAIL', 'H3': 'PASS' if h3 else 'FAIL', 'H4': 'PASS' if h4 else 'FAIL', 'resolved_requirement': 'PASS' if resolved_pass else 'FAIL', 'H1_frozen_zero_floor': FLOOR, 'H2_rule': 'upper 95% CI beta_Theta < 0', 'H3_rule': 'lower 95% CI beta_Q > 0', 'H4_rule': 'lower 95% CI beta_B > 0'}
    write('FORMAL_HYPOTHESES.json', hypotheses)
    all_pass = integrity['status'] == 'PASS' and all((v == 'PASS' for k, v in hypotheses.items() if k in ('H1', 'H2', 'H3', 'H4', 'resolved_requirement')))
    label = 'EXP_S1_SYMMETRY_UNLOCKING_PASS' if all_pass else 'EXP_S1_SYMMETRY_UNLOCKING_FAIL'
    formal = {'label': label, 'final': True, 'primary_horizon_nonpadding_tokens': 64000, 'training_integrity': 'PASS', 'hypotheses': {k: v for k, v in hypotheses.items() if k.startswith('H')}, 'resolved_requirement': hypotheses['resolved_requirement'], 'central_slopes': regressions['central_slopes'], 'ci95': {'beta_Q': bo['Q_C']['ci95'], 'beta_Theta': bo['Theta_R']['ci95'], 'beta_B': bo['P_B']['ci95']}}
    write('FORMAL_LABEL.json', formal)
    frozen_files = ('AUDIT_PRIMARY_64K.json', 'PRIMARY_REGRESSIONS.json', 'BOOTSTRAP.json', 'BOOTSTRAP_INDICES.npy', 'FORMAL_HYPOTHESES.json', 'FORMAL_LABEL.json')
    freeze = {'status': 'PRIMARY_FROZEN_BEFORE_SECONDARY', 'timestamp_asia_shanghai': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'files_sha256': {n: sha(OUT / n) for n in frozen_files}, 'evaluator_code_sha256': sha(Path(__file__)), 'central_label': label}
    write('PRIMARY_FREEZE.json', freeze)
    print(json.dumps(formal, indent=2), flush=True)

def secondary():
    freeze = json.loads((OUT / 'PRIMARY_FREEZE.json').read_text())
    for name, expected in freeze['files_sha256'].items():
        assert sha(OUT / name) == expected
    assert sha(Path(__file__)) == freeze['evaluator_code_sha256']
    result = {'status': 'DESCRIPTIVE_ONLY', 'primary_label_unchanged': freeze['central_label'], 'primary_freeze_sha256': sha(OUT / 'PRIMARY_FREEZE.json'), 'horizons': {}}
    for t in (16000, 256000):
        values = {name: np.full((8, 3), np.nan, dtype=np.float64) for name in ('Q_C', 'Theta_R', 'P_B')}
        for i, seed in enumerate(SEEDS):
            for j, arm in enumerate(ARMS):
                r = load_response(arm, seed, t)
                for name in values:
                    v = r[name]
                    if v is not None and math.isfinite(v) and (v > FLOOR):
                        values[name][i, j] = v
        result['horizons'][str(t)] = {'beta_Q': slope(values['Q_C']), 'beta_Theta': slope(values['Theta_R']), 'beta_B': slope(values['P_B']), 'resolved_counts': {name: int(np.isfinite(v).sum()) for name, v in values.items()}}
    write('SECONDARY_TEMPORAL.json', result)
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('primary', 'secondary'))
    a = p.parse_args()
    if a.stage == 'primary':
        primary()
    else:
        secondary()
