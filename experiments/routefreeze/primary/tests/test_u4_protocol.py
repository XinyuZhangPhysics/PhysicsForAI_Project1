# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Independent U4 artifact and frozen-protocol checks."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pytest
from transformers import AutoTokenizer
BASE = Path(__file__).parent
OUT = BASE / 'u4_run'
PKG = BASE / 'u4_codex_pkg'
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']

def read_json(path):
    return json.loads(path.read_text())

def test_permutation_identity_and_uniqueness():
    p = read_json(PKG / 'task/u4_reynolds_permutations.json')['permutations']
    assert len(p) == 16 and len({tuple(x) for x in p}) == 16
    assert all((sorted(x) == list(range(64)) for x in p))
    digest = hashlib.sha256(json.dumps(p, separators=(',', ':')).encode()).hexdigest()
    assert digest == '238a09d483374cfcf45863c0b4ffa93cf70baa8b1d1c14ae2f47650ca3c840fb'

def test_cross_distribution_exact_reynolds_identity():
    rng = np.random.default_rng(7)
    H = 5
    J = rng.normal(size=(H, 3, 4))
    K = np.array([J[j] @ J[j].T for j in range(H)])
    left = rng.random(H)
    right = rng.random(H)
    x = rng.normal(size=3)
    y = rng.normal(size=3)
    exact = np.mean([sum((left[i] * right[i] * (x @ K[p[i]] @ y) for i in range(H))) for p in itertools.permutations(range(H))])
    analytic = float(left @ right * (x @ K.mean(axis=0) @ y))
    assert abs(exact - analytic) / max(abs(analytic), 1e-30) < 1e-10

def test_exact_dataset_file_hashes():
    root = Path(_p1_path('path_f6b2b72181c3'))
    expected = {'validation-00000-of-00001-4fb685c22a3f91ef.parquet': 'ee4de07109f4c96251d1425e7da0ec5cb77a37bea738fc63d9712a5ec2d1f19e', 'test-00000-of-00001-9f769cf7ce219017.parquet': 'c474e86322ef235d0519484c2e37aafe52ab7f83ea3c4b707f4982e7c5f9de04'}
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest

def test_cross_split_duplicates_excluded():
    rows = [json.loads(x) for x in (BASE / 'u4_candidates.jsonl').read_text().splitlines()]
    validation = {r['text_sha256'] for r in rows if r['split'] == 'validation'}
    test = {r['text_sha256'] for r in rows if r['split'] == 'test'}
    assert validation.isdisjoint(test)

def test_selected_sequence_identity_and_lengths():
    selected = read_json(BASE / 'u4_selected_sequences.json')
    assert len(selected) == 128
    groups = defaultdict(list)
    for row in selected:
        assert len(row['token_ids']) == 128
        assert row['split'] in {'validation', 'test'} and row['domain'] in DOMAINS
        groups[row['split'], row['domain']].append(row)
    assert len(groups) == 8 and all((len(v) == 16 for v in groups.values()))
    assert all(([r['selection_key'] for r in v] == sorted((r['selection_key'] for r in v)) for v in groups.values()))

def test_deterministic_selection_replay():
    candidates = defaultdict(list)
    for line in (BASE / 'u4_candidates.jsonl').open():
        r = json.loads(line)
        candidates[r['split'], r['domain']].append(r)
    selected = read_json(BASE / 'u4_selected_sequences.json')
    tokenizer = AutoTokenizer.from_pretrained(_p1_path('path_f2699b70cfbf'), local_files_only=True)
    for group, rows in candidates.items():
        rows.sort(key=lambda r: r['selection_key'])
        expected = []
        for row in rows:
            key = hashlib.sha256('\x00'.join([row['split'], row['domain'], str(row['index']), row['text']]).encode()).hexdigest()
            assert key == row['selection_key']
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['text_sha256']
            ids = tokenizer.encode(row['text'], add_special_tokens=False)
            if len(ids) >= 128:
                expected.append((row['selection_key'], ids[:128]))
            if len(expected) == 16:
                break
        observed = [(r['selection_key'], r['token_ids']) for r in selected if (r['split'], r['domain']) == group]
        assert observed == expected

def test_preflight_numeric_gates():
    e = read_json(BASE / 'u4_preflight_evidence.json')
    assert e['status'] == 'PASS' and e['precision'] == 'CPU float32'
    assert e['synthetic_relative_error'] < 1e-10
    assert e['native_reconstruction_relative_error'] < 1e-06
    assert e['cached_gradient_relative_error'] < 1e-05
    assert e['common_expert_null_ratio'] < 1e-05
    assert e['router_max_logit_error'] < 1e-07 and e['router_max_weight_error'] < 1e-07
    assert e['expert_restoration_exact']

def test_scientific_validity_audits():
    e = read_json(OUT / 'u4_execution_audit.json')
    assert e['status'] == 'PASS'
    assert e['max_native_reconstruction_error'] < 1e-06
    assert e['cached_T_relative_error'] < 1e-05
    assert e['max_cached_gradient_relative_error'] < 1e-05
    assert e['router_stability'] == {'indices_mismatches': 0, 'max_logit_abs': 0.0, 'max_weight_abs': 0.0}
    assert e['expert_restoration_exact']
    assert e['replay_gradient_relative_error'] < 1e-06 and e['replay_probe_relative_error'] < 1e-06

def test_transfer_decomposition_and_permutation_mean():
    a = np.load(OUT / 'u4_scientific_arrays.npz')
    assert np.allclose(a['T_full'], a['T_route'] + a['B'], rtol=0, atol=1e-12)
    assert np.allclose(a['T_route'], a['T_permutation'].mean(axis=0), rtol=0, atol=1e-12)
    assert a['T_permutation'].shape == (16, 4, 4)

def test_symmetric_finite_differences_exact():
    a = np.load(OUT / 'u4_scientific_arrays.npz')
    rows = list(csv.DictReader((OUT / 'u4_probe_raw.csv').open()))
    assert len(rows) == 48
    for r in rows:
        source = DOMAINS.index(r['source'])
        target = DOMAINS.index(r['target'])
        rho = [5e-06, 1e-05, 2e-05].index(float(r['rho']))
        computed = (float(r['loss_plus']) - float(r['loss_minus'])) / (2 * float(r['epsilon']))
        assert computed == float(r['D_probe']) == a['probe'][rho, target, source]

def test_bootstrap_frozen_seed_replay():
    boot = np.load(OUT / 'u4_bootstrap_arrays.npz')
    a = np.load(OUT / 'u4_scientific_arrays.npz')
    rng = np.random.Generator(np.random.PCG64DXSM(202609270099))
    indices = rng.integers(0, 16, size=(10000, 16), dtype=np.int64)
    assert np.array_equal(indices, boot['bootstrap_indices'])
    assert np.allclose(a['T_permutation'][indices].mean(axis=1), boot['T_route_boot'], rtol=0, atol=0)

def test_frozen_decision_recomputed():
    m = read_json(OUT / 'u4_metrics.json')
    c = {'C1': m['R_full'] <= 0.05, 'C2': m['I_MAE'] >= 0.5 and m['I_MAE_ci95'][0] > 0.25, 'C3': m['Q_B'] >= 3 and m['S_B_ci95'][0] > 0, 'C4': m['E_MC_over_S_B'] <= 0.25, 'C5': m['locality_median_delta_scale'] <= 0.05 and m['locality_max_delta_scale'] <= 0.2}
    assert c == m['criteria']
    decision = 'INCONCLUSIVE_MC' if not c['C4'] else 'CONFIRMED' if all(c.values()) else 'NOT_CONFIRMED'
    assert decision == m['decision'] == read_json(OUT / 'u4_decision.json')['decision']

def test_routing_diagnostics_complete():
    rows = list(csv.DictReader((OUT / 'u4_routing_diagnostics.csv').open()))
    overlap = list(csv.DictReader((OUT / 'u4_routing_overlap.csv').open()))
    assert len(rows) == 8 * 64 and len(overlap) == 8 * 8
    for row in rows:
        assert 0 <= float(row['selection_frequency_per_token']) <= 1
        assert 0 <= float(row['mean_weight_when_selected']) <= 1
        assert int(row['token_positions']) == 2048

def test_required_six_figures():
    assert len(list((OUT / 'figures').glob('*.png'))) == 6
