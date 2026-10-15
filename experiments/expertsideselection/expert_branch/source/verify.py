# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
from collections import defaultdict
from check import verify_freeze
fr = verify_freeze()
binds = read(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json')
result = read(OUT / 'EXPERT_SIDE_SELECTION_RESULTS_v284M.json')
checks = {}
with (OUT / 'EXPERT_SIDE_SELECTION_COORDINATES_v284M.csv').open() as stream:
    df = list(csv.DictReader(stream))
with (OUT / 'EXPERT_SIDE_SELECTION_CELL_SEED_SUMMARY_v284M.csv').open() as stream:
    ss = list(csv.DictReader(stream))
rowmap = defaultdict(list)
seedsets = defaultdict(set)
unique = set()
for x in df:
    key = (x['family'], int(x['g']), int(x['seed']), int(x['direction']), x['class'], int(x['s']))
    rowmap[key].append(x)
    unique.add(key + (int(x['coordinate']),))
    seedsets[key[:2]].add(key[2])
checks['coordinate_rows_10752'] = len(df) == 10752
checks['seed_rows_224'] = len(ss) == 224
checks['complete_family_g_seed_direction_class_background_coordinates'] = len(unique) == len(df) and all((len(v) == 12 for v in rowmap.values()))
checks['exactly_8_seed_identities_per_family_g'] = len(seedsets) == 8 and all((len(v) == 8 for v in seedsets.values()))
checks['no_child_class_at_g1'] = all((not (int(x['g']) == 1 and x['class'] == 'child') for x in df))
checks['all_numeric_values_finite'] = all((math.isfinite(float(x[k])) for x in df for k in ['dot_B_actual', 'dot_B_prediction_A', 'residual', 'dot_B_secondary_restricted', 'P_ref']))
for x in binds['entries']:
    assert sha(x['path']) == x['sha256'], x['path']
checks['all_bound_external_source_hashes_unchanged'] = True
old = read(OUT / 'HISTORICAL_SEAL_BEFORE_v284M.json')
checks['historical_seal_unchanged'] = all((sha(p) == h for p, h in old.items())) and {str(p) for p in OLD.rglob('*') if p.is_file()} == set(old)
max_actual = 0.0
max_pred = 0.0
max_projection = 0.0
max_qC = 0.0
max_qM = 0.0
for f in ['S1', 'S2']:
    bank = next((x for x in binds['banks'] if x['family'] == f and x['phase'] == 'audit'))
    z = np.load(bank['path'])
    h = torch.from_numpy(z['h']).double()
    src, tgt, group = (z['src'], z['tgt'], z['group'])
    for g in [1, 2, 4, 8]:
        j4 = np.ones((4, 4)) / 4
        jg = np.ones((g, g)) / g
        PA = np.kron(np.eye(4) - j4, jg)
        PC = np.kron(j4, np.eye(g) - jg)
        PM = np.kron(np.eye(4) - j4, np.eye(g) - jg)
        for point in [x for x in binds['copied_points'] if x['family'] == f and x['g'] == g]:
            seed = point['seed']
            W = load_file(point['path'])['router.weight'].double()
            logits = h @ W.T
            u1 = (logits.softmax(1)[src] * logits.softmax(1)[tgt]).numpy()
            center = u1 - u1.mean(1, keepdims=True)
            pref = np.mean((center * center).sum(1) / (4 * g - 1))
            root = math.sqrt(pref)
            for bg in [0, 1]:
                probability = (logits * bg).softmax(1)
                u = (probability[src] * probability[tgt]).numpy()
                q = (u - u.mean(1, keepdims=True)) / root
                qa = q @ PA
                max_qC = max(max_qC, float(np.max(abs(q @ PC))))
                max_qM = max(max_qM, float(np.max(abs(q @ PM))))
                for r in range(4):
                    v = direction(f, g, seed, r)
                    normS1 = np.sum(v['a'] ** 2) * np.sum(v['b'] ** 2) + np.sum(v['bias'] ** 2) + np.sum(v['a'] ** 2) * np.sum(v['e'] ** 2)
                    normS2 = np.sum(v['a'] ** 2) * (np.sum(v['b'] ** 2) + np.sum(v['b2'] ** 2) + np.sum(v['e'] ** 2))
                    assert abs((normS1 if f == 'S1' else normS2) - 1) < 1e-12
                    assert abs(v['parent'].sum()) < 1e-12 and abs(np.linalg.norm(v['parent']) - 1) < 1e-12
                    for cls in ['parent'] + (['child'] if g > 1 else []):
                        fp = OUT / 'fields' / f'{f}_g{g}_seed{seed}_r{r}_{cls}.npz'
                        field = np.load(fp)
                        dk = field['dot_kappa']
                        a = dk @ PA
                        max_projection = max(max_projection, float(np.max(abs(a - field['dot_kappa_A']))), float(np.max(abs(dk @ PC - field['dot_kappa_C']))), float(np.max(abs(dk @ PM - field['dot_kappa_M']))))
                        actualb = aggregate(np.sum(q * (dk - dk.mean(1, keepdims=True)), axis=1), group)
                        pred = aggregate(np.sum(qa * a, axis=1), group)
                        rows = sorted(rowmap[f, g, seed, r, cls, bg], key=lambda x: int(x['coordinate']))
                        max_actual = max(max_actual, float(np.max(abs(actualb - np.array([float(x['dot_B_actual']) for x in rows])))))
                        max_pred = max(max_pred, float(np.max(abs(pred - np.array([float(x['dot_B_prediction_A']) for x in rows])))))
checks['independent_torch_occupancy_actual_coordinate_reconstruction'] = max_actual < 1e-10
checks['independent_dense_A_prediction_reconstruction'] = max_pred < 1e-10
checks['dense_projectors_agree_with_saved_fields'] = max_projection < 1e-10
checks['copied_q_has_only_A_sector'] = max(max_qC, max_qM) < 1e-10
checks['all_calibration_and_audit_independent_checks_pass'] = read(OUT / 'CALIBRATION_IMPLEMENTATION_CHECK_v284M.json')['pass'] and read(OUT / 'AUDIT_IMPLEMENTATION_CHECK_v284M.json')['pass']
H1 = all((c['hypotheses']['H1'] for c in result['cells']))
H2 = all((c['hypotheses']['H2'] for c in result['cells'] if c['g'] > 1))
H3 = all((c['hypotheses']['H3'] for c in result['cells'] if c['g'] == 1))
H4 = all((c['hypotheses']['H4'] for c in result['cells']))
H5 = all((c['hypotheses']['H5'] for c in result['cells']))
checks['classification_matches_preregistered_all_H1_H5_rule'] = (result['label'] == 'EXPERT_SIDE_SELECTION_CONFIRMED') == (H1 and H2 and H3 and H4 and H5)
checkpass = all((bool(x) for x in checks.values()))
write('FINAL_VERIFICATION_v284M.json', {'timestamp': stamp(), 'pass': checkpass, 'checks': {k: bool(v) for k, v in checks.items()}, 'independent_actual_max_abs_error': max_actual, 'independent_prediction_max_abs_error': max_pred, 'dense_projector_max_abs_error': max_projection, 'q_C_max_abs': max_qC, 'q_M_max_abs': max_qM, 'historical_files_verified': len(old), 'sources_verified': len(binds['entries'])})
assert checkpass, checks
print('FINAL VERIFICATION PASS', len(checks), 'checks', flush=True)
