# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
from check import verify_freeze, implementation
import time
start = time.monotonic()
fr = verify_freeze()
assert read(OUT / 'CALIBRATION_IMPLEMENTATION_CHECK_v284M.json')['pass']
binding = read(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json')
records = []
summaries = []
projchecks = []
refs = []
write('AUDIT_OPENING_v284M.json', {'timestamp': stamp(), 'preregistration_sha256': fr['preregistration_sha256'], 'calibration_pass': True, 'audit_derivatives_previously_computed': False})
implementation('audit')
for f in ['S1', 'S2']:
    bank = next((x for x in binding['banks'] if x['family'] == f and x['phase'] == 'audit'))
    z = np.load(bank['path'])
    h = z['h'].astype(np.float64)
    src, tgt, group = (z['src'], z['tgt'], z['group'])
    s = state(f)
    p = prep(h, s, f)
    d, D = DIM[f]
    for g in [1, 2, 4, 8]:
        j4 = np.ones((4, 4)) / 4
        jg = np.ones((g, g)) / g
        P = {'A': np.kron(np.eye(4) - j4, jg), 'C': np.kron(j4, np.eye(g) - jg), 'M': np.kron(np.eye(4) - j4, np.eye(g) - jg)}
        for point in [x for x in binding['copied_points'] if x['family'] == f and x['g'] == g]:
            seed = point['seed']
            assert sha(point['path']) == point['sha256']
            w = load_file(point['path'])['router.weight'].double().numpy()
            qs = {bg: qfield(h, w, g, bg, src, tgt) for bg in [1, 0]}
            refs.append({'family': f, 'g': g, 'seed': seed, 'P_ref': qs[1][-1]})
            for r in range(4):
                v = direction(f, g, seed, r)
                dchild, drestricted = derivative(p, f, g, v, src, tgt)
                for cls in ['parent'] + (['child'] if g > 1 else []):
                    dk = field_for(dchild, f, g, v, cls)
                    ka, kc, km = project(dk, g)
                    dr = field_for(drestricted, f, g, v, cls)
                    dra = project(dr, g)[0]
                    perr = {name: float(np.max(abs(dk @ P[name] - component))) for name, component in [('A', ka), ('C', kc), ('M', km)]}
                    projchecks.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'class': cls, 'max_abs_error': max(perr.values())})
                    fieldpath = OUT / 'fields' / f'{f}_g{g}_seed{seed}_r{r}_{cls}.npz'
                    fieldpath.parent.mkdir(exist_ok=True)
                    np.savez_compressed(fieldpath, dot_kappa=dk, dot_kappa_A=ka, dot_kappa_C=kc, dot_kappa_M=km)
                    for bg, (q, u, root, pref) in qs.items():
                        qa, qc, qm = project(q, g)
                        actualb = actual(dk, u, root, group)
                        pred = coords(qa, ka, group)
                        restrict_actual = actual(dr, u, root, group)
                        res = actualb - pred
                        tol = fr['closure_atol'] + fr['closure_rtol'] * float(np.max(abs(pred)))
                        fieldA = float(np.max(abs(ka)))
                        null = cls == 'child'
                        dark = bg == 0 or null
                        qaerr = float(np.max(abs(q @ P['A'] - qa)))
                        assert qaerr < 1e-10
                        common = {'family': f, 'g': g, 'seed': seed, 'seed_index': seed - (2026100101 if f == 'S1' else 2026110201), 'direction': r, 'class': cls, 's': bg, 'P_ref': pref}
                        summary = common | {'actual_RMS12': float(np.sqrt(np.mean(actualb ** 2))), 'prediction_RMS12': float(np.sqrt(np.mean(pred ** 2))), 'max_abs_residual': float(np.max(abs(res))), 'relative_L2_residual': float(np.linalg.norm(res) / max(np.linalg.norm(pred), fr['structural_floor'])), 'closure_tolerance': tol, 'closure_pass': bool(np.max(abs(res)) <= tol), 'dark_pass': bool(np.max(abs(actualb)) <= fr['structural_floor']) if dark else None, 'field_A_max_abs': fieldA, 'field_C_max_abs': float(np.max(abs(kc))), 'field_M_max_abs': float(np.max(abs(km))), 'q_A_max_abs': float(np.max(abs(qa))), 'q_C_max_abs': float(np.max(abs(qc))), 'q_M_max_abs': float(np.max(abs(qm))), 'A_null_field_pass': fieldA <= fr['structural_floor'] if null else None, 'q_A_symmetric_pass': np.max(abs(qa)) <= 1e-12 if bg == 0 else None, 'projector_pass': max(perr.values()) <= fr['structural_floor'] and qaerr <= fr['structural_floor'], 'field_path': str(fieldpath.relative_to(OUT)), 'field_sha256': sha(fieldpath)}
                        summaries.append(summary)
                        for i in range(12):
                            records.append(common | {'coordinate': i, 'domain_pair': f'{DOMAIN[i // 3]}->{[x for x in DOMAIN if x != DOMAIN[i // 3]][i % 3]}', 'dot_B_actual': float(actualb[i]), 'dot_B_prediction_A': float(pred[i]), 'residual': float(res[i]), 'dot_B_secondary_restricted': float(restrict_actual[i]), 'closure_tolerance': tol, 'structural_floor': fr['structural_floor'], 'pass_closure': abs(res[i]) <= tol})
            print('PRIMARY', f, g, seed, 'complete', flush=True)
        csvwrite('EXPERT_SIDE_SELECTION_COORDINATES_v284M.csv', records)
        csvwrite('EXPERT_SIDE_SELECTION_DIRECTION_SUMMARY_v284M.csv', summaries)
write('REFERENCE_SCALES_v284M.json', refs)
write('PROJECTOR_CHECKS_v284M.json', projchecks)
seedrows = []
for f in ['S1', 'S2']:
    for g in [1, 2, 4, 8]:
        for seed in sorted({x['seed'] for x in summaries if x['family'] == f and x['g'] == g}):
            for cls in ['parent'] + (['child'] if g > 1 else []):
                for bg in [1, 0]:
                    ss = [x for x in summaries if (x['family'], x['g'], x['seed'], x['class'], x['s']) == (f, g, seed, cls, bg)]
                    seedrows.append({'family': f, 'g': g, 'seed': seed, 'class': cls, 's': bg, 'n_directions': len(ss), 'mean_direction_RMS12': float(np.mean([x['actual_RMS12'] for x in ss])), 'max_abs_residual': max((x['max_abs_residual'] for x in ss)), 'all_closure_pass': all((x['closure_pass'] for x in ss)), 'all_dark_pass': all((x['dark_pass'] for x in ss)) if cls == 'child' or bg == 0 else None})
csvwrite('EXPERT_SIDE_SELECTION_CELL_SEED_SUMMARY_v284M.csv', seedrows)
idx = np.random.default_rng(fr['bootstrap_seed']).integers(0, 8, (fr['bootstrap_draws'], 8))
np.save(OUT / 'BOOTSTRAP_SEED_INDICES_v284M.npy', idx)
cells = []
dists = {}
for f in ['S1', 'S2']:
    for g in [1, 2, 4, 8]:
        vals = np.array([x['mean_direction_RMS12'] for x in seedrows if (x['family'], x['g'], x['class'], x['s']) == (f, g, 'parent', 1)])
        boot = vals[idx].mean(1)
        dists[f'{f}_g{g}'] = boot
        ci = np.quantile(boot, [0.025, 0.975])
        ss = [x for x in summaries if x['family'] == f and x['g'] == g]
        H = {'H1': all((x['closure_pass'] for x in ss if x['class'] == 'parent' and x['s'] == 1)), 'H2': all((x['A_null_field_pass'] and x['dark_pass'] for x in ss if x['class'] == 'child' and x['s'] == 1)) if g > 1 else None, 'H3': bool(vals.min() > fr['structural_floor'] and ci[0] > fr['structural_floor']) if g == 1 else None, 'H4': all((x['dark_pass'] and x['q_A_symmetric_pass'] for x in ss if x['s'] == 0)), 'H5': all((x['projector_pass'] for x in ss)) and read(OUT / 'AUDIT_IMPLEMENTATION_CHECK_v284M.json')['pass'] and read(OUT / 'CALIBRATION_IMPLEMENTATION_CHECK_v284M.json')['pass']}
        cells.append({'family': f, 'g': g, 'n_seeds': 8, 'mean_direction_RMS12': float(vals.mean()), 'bootstrap_CI95': ci.tolist(), 'min_seed_mean_direction_RMS12': float(vals.min()), 'max_abs_residual': max((x['max_abs_residual'] for x in ss)), 'max_dark_response_RMS12': max((x['actual_RMS12'] for x in ss if x['s'] == 0 or x['class'] == 'child')), 'hypotheses': H, 'all_applicable_pass': all((x for x in H.values() if x is not None))})
np.savez_compressed(OUT / 'BOOTSTRAP_DISTRIBUTIONS_v284M.npz', **dists)
csvwrite('EXPERT_SIDE_SELECTION_CELL_SUMMARY_v284M.csv', [{k: v for k, v in c.items() if k not in ['hypotheses', 'bootstrap_CI95']} | {'CI95_lower': c['bootstrap_CI95'][0], 'CI95_upper': c['bootstrap_CI95'][1], **c['hypotheses']} for c in cells])
technical = all((c['hypotheses']['H5'] for c in cells))
structural = all((all((c['hypotheses'][h] for h in ['H1', 'H4'])) and c['hypotheses']['H2'] is not False for c in cells))
bright = [c['hypotheses']['H3'] for c in cells if c['g'] == 1]
label = 'TECHNICAL_PROTOCOL_FAILURE' if not technical else 'SCIENTIFIC_CONFLICT' if not structural else 'EXPERT_SIDE_SELECTION_CONFIRMED' if all(bright) else 'PARTIAL_EXPERT_SIDE_SELECTION' if any(bright) else 'EXPERT_SIDE_SELECTION_NOT_VALIDATED'
write('EXPERT_SIDE_SELECTION_RESULTS_v284M.json', {'timestamp': stamp(), 'label': label, 'preregistration_sha256': fr['preregistration_sha256'], 'n_cells': 8, 'n_copied_points': 64, 'n_coordinate_records': len(records), 'n_direction_background_records': len(summaries), 'bootstrap_unit': 'eight whole paired seed indices; four nuisance directions averaged inside seed', 'cells': cells, 'max_primary_abs_closure_residual': max((x['max_abs_residual'] for x in summaries)), 'runtime_seconds': time.monotonic() - start, 'secondary_restricted_not_used_for_classification': True, 'historical_full_tangent_classification_unchanged': 'PARTIAL_CHANNEL_AGREEMENT'})
print('FINAL LABEL', label, flush=True)
