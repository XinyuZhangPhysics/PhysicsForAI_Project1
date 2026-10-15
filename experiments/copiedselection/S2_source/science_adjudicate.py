# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""EXP-S2 preregistered gates, primary adjudication, and secondary analysis."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from science_response import RESPONSE_KEYS
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/science_exp_s2'
BANK = ROOT / 'outputs/reports/exp_s2/preflight/banks'
FREEZE = ROOT / 'outputs/reports/exp_s2/preregistration_freeze'
S1_EVALUATOR = ROOT.parent / 'EXP_S2_SCIENCE_EXECUTION_PACKAGE_v1/s1_frozen_sources/evaluate_s1.py'
SEEDS = tuple(range(2026110201, 2026110209))
ARMS = ('S2', 'S4', 'S8')
HORIZONS = (0, 16000, 64000, 256000)
X = np.log(np.asarray([2.0, 4.0, 8.0]))
FLOOR = 1e-20
BOOT_SEED = 2026100199
BOOT_N = 20000
CODE_FILES = (ROOT / 'src/s2_impl.py', ROOT / 'src/science_train.py', ROOT / 'src/science_response.py', Path(__file__), S1_EVALUATOR)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(name: str, data: dict) -> None:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f'REFUSING_TO_REPLACE {path}')
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def timestamp() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def code_hashes() -> dict:
    return {p.name: sha(p) for p in CODE_FILES}

def response_path(membership: str, arm: str, seed: int, t: int) -> Path:
    return OUT / f'{membership}_responses/{arm}_seed_{seed}_t{t}.json'

def response(membership: str, arm: str, seed: int, t: int) -> dict:
    path = response_path(membership, arm, seed, t)
    value = json.loads(path.read_text())
    if tuple(value) != RESPONSE_KEYS or value['membership'] != membership or value['arm'] != arm or (value['seed'] != seed) or (value['horizon_nonpadding_tokens'] != t):
        raise RuntimeError(f'INVALID_RESPONSE_SCHEMA {path}')
    for key in ('Pq_normalizer', 'Q_C', 'K_C_0', 'P_B', 'direct_response_power', 'response_projector_max_abs_error'):
        if not math.isfinite(value[key]):
            raise RuntimeError(f'NONFINITE_RESPONSE {path} {key}')
    if len(value['B_coordinate_powers']) != 12 or not all((math.isfinite(x) for x in value['B_coordinate_powers'])):
        raise RuntimeError(f'INVALID_B_COORDINATES {path}')
    return value

def verify_training() -> dict:
    data = json.loads((OUT / 'TRAINING_INTEGRITY.json').read_text())
    assert data['status'] == 'PASS' and data['runs'] == 32
    assert data['total_nonpadding_tokens'] == 8192000
    assert data['optimizer_steps_per_run'] == 1001
    for seed in SEEDS:
        for arm in ('S1',) + ARMS:
            path = OUT / f'checkpoints/{arm}_seed_{seed}/RUN_RESULT.json'
            assert sha(path) == data['run_result_sha256'][f'{arm}_{seed}']
            run = json.loads(path.read_text())
            assert run['status'] == 'PASS' and run['nonpadding_tokens_consumed'] == 256000
            assert run['optimizer_steps'] == 1001 and sorted(map(int, run['checkpoints'])) == list(HORIZONS)
            for t in HORIZONS:
                ckpt = run['checkpoints'][str(t)]
                assert ckpt['finite'] and sha(ROOT / ckpt['path']) == ckpt['sha256']
                assert ckpt['frozen_tensor_hashes'] == run['frozen_tensor_hashes']
            assert sha(path.parent / 'training_trace.jsonl') == run['trace_sha256']
    return data

def freeze_calibration() -> dict:
    training = verify_training()
    files = {}
    for seed in SEEDS:
        for arm in ('S1',) + ARMS:
            for t in HORIZONS:
                response('calibration', arm, seed, t)
                path = response_path('calibration', arm, seed, t)
                files[str(path.relative_to(OUT))] = sha(path)
    assert len(files) == 128
    geometry = OUT / 'CALIBRATION_GEOMETRY.json'
    rep = OUT / 'CALIBRATION_REPRESENTATION.json'
    bank_tokens = OUT / 'CALIBRATION_BANK_TOKENS.json'
    for p in (geometry, rep, bank_tokens):
        assert p.exists()
        files[p.name] = sha(p)
    schema = hashlib.sha256(json.dumps(RESPONSE_KEYS, separators=(',', ':')).encode()).hexdigest()
    result = {'status': 'CALIBRATION_FROZEN', 'timestamp_asia_shanghai': timestamp(), 'training_integrity_sha256': sha(OUT / 'TRAINING_INTEGRITY.json'), 'response_count': 128, 'response_schema_keys': list(RESPONSE_KEYS), 'response_schema_sha256': schema, 'code_sha256': code_hashes(), 'frozen_preregistration_sha256': sha(FREEZE / 'PhysicsForAI_Project1_EXP_S2_PREREGISTRATION_FROZEN_v1.md'), 'files_sha256': files}
    write('CALIBRATION_FREEZE.json', result)
    return result

def open_audit() -> dict:
    if (OUT / 'AUDIT_OPENING_RECORD.json').exists():
        raise RuntimeError('AUDIT_ALREADY_OPENED')
    training = verify_training()
    cal_path = OUT / 'CALIBRATION_FREEZE.json'
    cal = json.loads(cal_path.read_text())
    assert cal['status'] == 'CALIBRATION_FROZEN' and cal['code_sha256'] == code_hashes()
    assert cal['training_integrity_sha256'] == sha(OUT / 'TRAINING_INTEGRITY.json')
    assert cal['response_count'] == 128 and training['runs'] == 32
    for name, expected in cal['files_sha256'].items():
        assert sha(OUT / name) == expected
    frozen = json.loads((FREEZE / 'EXP_S2_PREREGISTRATION_FREEZE.json').read_text())
    index = json.loads((BANK / 'BANKS_INDEX.json').read_text())
    result = {'status': 'AUDIT_OPENED_ONCE', 'timestamp_asia_shanghai': timestamp(), 'frozen_commit': '57b6e98ecb48640746de53c30715092b62598460', 'frozen_preregistration_sha256': cal['frozen_preregistration_sha256'], 'freeze_record_sha256': sha(FREEZE / 'EXP_S2_PREREGISTRATION_FREEZE.json'), 'training_integrity_sha256': sha(OUT / 'TRAINING_INTEGRITY.json'), 'calibration_freeze_sha256': sha(cal_path), 'code_sha256': code_hashes(), 'response_schema_sha256': cal['response_schema_sha256'], 'audit_document_manifest_sha256': sha(BANK / 'future_audit_document_ids.jsonl'), 'audit_pair_manifest_sha256': sha(BANK / 'future_audit_pair_ids.jsonl'), 'model_asset_sha256': frozen['model']['asset_sha256'], 'dataset_revision': frozen['dataset']['revision'], 'audit_bank_index_sha256': index['manifest_sha256']['future_audit']}
    assert result['audit_document_manifest_sha256'] == index['manifest_sha256']['future_audit']
    assert result['audit_pair_manifest_sha256'] == index['manifest_sha256']['future_audit_pairs']
    write('AUDIT_OPENING_RECORD.json', result)
    return result

def s1_slope_functions():
    spec = importlib.util.spec_from_file_location('frozen_s1_evaluator', S1_EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BOOT_N == BOOT_N and module.BOOT_SEED == BOOT_SEED and (module.FLOOR == FLOOR)
    return (module.slope, module.bootstrap)

def primary() -> dict:
    assert (OUT / 'AUDIT_OPENING_RECORD.json').exists()
    assert not (OUT / 'PRIMARY_FREEZE.json').exists()
    verify_training()
    slope, bootstrap = s1_slope_functions()
    grid = {name: np.full((8, 3), np.nan, dtype=np.float64) for name in ('Q_C', 'K_C_0', 'P_B', 'Theta_R')}
    rows = []
    for i, seed in enumerate(SEEDS):
        for j, arm in enumerate(ARMS):
            r = response('audit', arm, seed, 64000)
            for name in grid:
                v = r[name]
                if v is not None and math.isfinite(v) and (v > FLOOR):
                    grid[name][i, j] = v
            rows.append({'seed': seed, 'arm': arm, 'g': 2 ** (j + 1), 'Q_C': r['Q_C'], 'K_C_0': r['K_C_0'], 'P_B': r['P_B'], 'Theta_R': r['Theta_R'], 'resolved_positive': r['resolved_positive'], 'response_file_sha256': sha(response_path('audit', arm, seed, 64000))})
    resolved = np.isfinite(grid['Q_C']) & np.isfinite(grid['K_C_0']) & np.isfinite(grid['P_B'])
    counts = {arm: int(resolved[:, j].sum()) for j, arm in enumerate(ARMS)}
    resolved_pass = all((x >= 7 for x in counts.values()))
    h1_rows = []
    for seed in SEEDS:
        for t in HORIZONS:
            r = response('audit', 'S1', seed, t)
            powers = r['B_coordinate_powers']
            h1_rows.append({'seed': seed, 'horizon': t, 'max_B_coordinate_power': max(powers), 'all_12_below_floor': all((math.isfinite(x) and x <= FLOOR for x in powers)), 'response_file_sha256': sha(response_path('audit', 'S1', seed, t))})
    h1 = all((x['all_12_below_floor'] for x in h1_rows))
    write('AUDIT_PRIMARY_64K.json', {'status': 'FROZEN', 'primary_horizon_nonpadding_tokens': 64000, 'resolution_power_floor': FLOOR, 'runs': rows, 'resolved_count_by_arm': counts, 'resolved_requirement_at_least_7_of_8': resolved_pass, 'S2_H1_rows': h1_rows, 'S2_H1_pass': h1})
    central = {name: slope(grid[name]) for name in ('Q_C', 'Theta_R', 'P_B')}
    geom = json.loads((OUT / 'AUDIT_GEOMETRY.json').read_text())
    kval = np.asarray([geom['arms'][arm]['K_C_0'] for arm in ARMS], dtype=np.float64)
    assert np.all(kval > FLOOR)
    beta_k = float(np.polyfit(X, np.log(kval), 1)[0])
    regressions = {'status': 'FROZEN', 'method': 'seed fixed-effect OLS on log positive resolved values; within-seed center log(g) and log(response); omit unresolved without regularization', 'primary_horizon_nonpadding_tokens': 64000, 'central_slopes': {'beta_Q': central['Q_C'], 'beta_Theta': central['Theta_R'], 'beta_B': central['P_B']}, 'descriptive_K_C_0_by_arm': {arm: float(kval[j]) for j, arm in enumerate(ARMS)}, 'descriptive_beta_K': beta_k, 'resolved_count_by_arm': counts}
    write('PRIMARY_REGRESSIONS.json', regressions)
    indices = np.random.default_rng(BOOT_SEED).integers(0, 8, size=(BOOT_N, 8), dtype=np.int16)
    idx_path = OUT / 'BOOTSTRAP_INDICES.npy'
    np.save(idx_path, indices)
    bo = {name: bootstrap(grid[name], indices) for name in ('Q_C', 'Theta_R', 'P_B')}
    write('BOOTSTRAP.json', {'status': 'FROZEN', 'replicates': BOOT_N, 'seed': BOOT_SEED, 'resampling_unit': 'whole paired seed triplet; eight indices with replacement per replicate', 'indices_sha256': sha(idx_path), 'percentile_method': 'numpy quantile linear 2.5%, 97.5%', 'beta_Q': bo['Q_C'], 'beta_Theta': bo['Theta_R'], 'beta_B': bo['P_B']})
    h2 = bool(bo['Theta_R']['resolved'] and bo['Theta_R']['ci95'][1] < 0)
    h3 = bool(bo['Q_C']['resolved'] and bo['Q_C']['ci95'][0] > 0)
    h4 = bool(bo['P_B']['resolved'] and bo['P_B']['ci95'][0] > 0)
    hypotheses = {'H1': 'PASS' if h1 else 'FAIL', 'H2': 'PASS' if h2 else 'FAIL', 'H3': 'PASS' if h3 else 'FAIL', 'H4': 'PASS' if h4 else 'FAIL', 'resolved_requirement': 'PASS' if resolved_pass else 'FAIL', 'H1_frozen_zero_floor': FLOOR, 'H2_rule': 'upper 95% CI beta_Theta < 0', 'H3_rule': 'lower 95% CI beta_Q > 0', 'H4_rule': 'lower 95% CI beta_B > 0'}
    write('FORMAL_HYPOTHESES.json', hypotheses)
    all_pass = all((hypotheses[k] == 'PASS' for k in ('H1', 'H2', 'H3', 'H4', 'resolved_requirement')))
    label = 'EXP_S2_SCIENCE_PASS' if all_pass else 'EXP_S2_SCIENCE_FORMAL_FAIL'
    formal = {'label': label, 'final': True, 'primary_horizon_nonpadding_tokens': 64000, 'training_integrity': 'PASS', 'hypotheses': {k: hypotheses[k] for k in ('H1', 'H2', 'H3', 'H4')}, 'resolved_requirement': hypotheses['resolved_requirement'], 'central_slopes': regressions['central_slopes'], 'ci95': {'beta_Q': bo['Q_C']['ci95'], 'beta_Theta': bo['Theta_R']['ci95'], 'beta_B': bo['P_B']['ci95']}}
    write('FORMAL_LABEL.json', formal)
    names = ('AUDIT_PRIMARY_64K.json', 'PRIMARY_REGRESSIONS.json', 'BOOTSTRAP.json', 'BOOTSTRAP_INDICES.npy', 'FORMAL_HYPOTHESES.json', 'FORMAL_LABEL.json')
    write('PRIMARY_FREEZE.json', {'status': 'PRIMARY_FROZEN_BEFORE_SECONDARY', 'timestamp_asia_shanghai': timestamp(), 'files_sha256': {n: sha(OUT / n) for n in names}, 'code_sha256': code_hashes(), 'central_label': label})
    return formal

def secondary() -> dict:
    freeze = json.loads((OUT / 'PRIMARY_FREEZE.json').read_text())
    assert freeze['code_sha256'] == code_hashes()
    for name, expected in freeze['files_sha256'].items():
        assert sha(OUT / name) == expected
    slope, _ = s1_slope_functions()
    result = {'status': 'DESCRIPTIVE_ONLY', 'primary_label_unchanged': freeze['central_label'], 'primary_freeze_sha256': sha(OUT / 'PRIMARY_FREEZE.json'), 'horizons': {}}
    for t in (16000, 256000):
        values = {name: np.full((8, 3), np.nan, dtype=np.float64) for name in ('Q_C', 'Theta_R', 'P_B')}
        for i, seed in enumerate(SEEDS):
            for j, arm in enumerate(ARMS):
                r = response('audit', arm, seed, t)
                for name in values:
                    v = r[name]
                    if v is not None and math.isfinite(v) and (v > FLOOR):
                        values[name][i, j] = v
        result['horizons'][str(t)] = {'beta_Q': slope(values['Q_C']), 'beta_Theta': slope(values['Theta_R']), 'beta_B': slope(values['P_B']), 'resolved_counts': {name: int(np.isfinite(v).sum()) for name, v in values.items()}}
    write('SECONDARY_TEMPORAL.json', result)
    return result
if __name__ == '__main__':
    if sys.argv[1:] == ['freeze-calibration']:
        print(json.dumps(freeze_calibration(), indent=2))
    elif sys.argv[1:] == ['open-audit']:
        print(json.dumps(open_audit(), indent=2))
    elif sys.argv[1:] == ['primary']:
        print(json.dumps(primary(), indent=2))
    elif sys.argv[1:] == ['secondary']:
        print(json.dumps(secondary(), indent=2))
    else:
        raise SystemExit('usage: science_adjudicate.py freeze-calibration|open-audit|primary|secondary')
