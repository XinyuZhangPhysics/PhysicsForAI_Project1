# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
from inference import *
from kernels import *
assert read(OUT / 'RUN_STATUS.json')['status'] == 'PRIMARY_COORDINATES_COMPLETE'
with (OUT / 'FULL_TANGENT_COORDINATES_v283M.csv').open() as f:
    rows = list(csv.DictReader(f))
for r in rows:
    for k in ['g', 'seed', 'coordinate_index']:
        r[k] = int(r[k])
    for k in ['B_restricted', 'B_full', 'P_B_restricted', 'P_B_full', 'epsilon_B', 'tau_P', 'restricted_sign_floor', 'full_sign_floor']:
        r[k] = float(r[k])
floor = read(OUT / 'NUMERICAL_FLOORS_FREEZE_v283M.json')['cells']
geo = read(OUT / 'FULL_TANGENT_GEOMETRY_AUDIT_v283M.json')['cells']
assert np.array_equal(rank_rows(np.array([[3.0, 1.0, 3.0, 9.0]])), [[2.5, 1.0, 2.5, 4.0]])
test = np.array([[1.0, 1.0, 2.0, 2.0, 3.0]])
assert np.array_equal(rank_rows(test), [[1.5, 1.5, 3.5, 3.5, 5.0]])
rng = np.random.Generator(np.random.PCG64(2026101201))
indices = rng.integers(0, 8, size=(20000, 8))
np.save(OUT / 'BOOTSTRAP_SEED_INDICES_v283M.npy', indices)
families = {}
cells = []
h0 = {}
h1 = {}
for fam in ['S1', 'S2']:
    seeds = sorted({r['seed'] for r in rows if r['family'] == fam})
    assert len(seeds) == 8
    by = {(r['seed'], r['g'], r['coordinate_index']): r for r in rows if r['family'] == fam}
    x = np.array([[[by[s, g, i]['B_restricted'] for i in range(12)] for g in [2, 4, 8]] for s in seeds])
    y = np.array([[[by[s, g, i]['B_full'] for i in range(12)] for g in [2, 4, 8]] for s in seeds])
    fr = np.array([next((z['restricted_sign_floor'] for z in floor if z['family'] == fam and z['g'] == g)) for g in [2, 4, 8]])[:, None]
    ff = np.array([next((z['full_sign_floor'] for z in floor if z['family'] == fam and z['g'] == g)) for g in [2, 4, 8]])[:, None]
    assert x.shape == y.shape == (8, 3, 12) and np.isfinite(x).all() and np.isfinite(y).all()
    point = all_metrics(x, y, fr, ff)
    draws = bootstrap(x, y, fr, ff, indices)
    np.savez_compressed(OUT / f'{fam}_BOOTSTRAP_DISTRIBUTIONS_v283M.npz', **draws)
    valid = (np.abs(x) > fr) & (np.abs(y) > ff)
    family = {'n_coordinates': 288, 'n_seed_clusters': 8, 'seeds': seeds, 'sign_eligible_coordinates': int(np.sum(valid)), 'metrics': {}}
    for name, estimate in point.items():
        ci = np.percentile(draws[name], [2.5, 97.5])
        assert np.isfinite(draws[name]).all()
        threshold = 0.5 if name == 'sign_agreement' else 0.0
        family['metrics'][name] = {'estimate': float(estimate), 'ci95': [float(z) for z in ci], 'lower_bound_required_strictly_above': threshold, 'criterion_pass': bool(ci[0] > threshold)}
    family['H2_confirmation_criteria_pass'] = all((z['criterion_pass'] for z in family['metrics'].values()))
    families[fam] = family
    h0[fam] = all((by[s, 1, i]['B_full'] == 0 and by[s, 1, i]['B_restricted'] == 0 and (by[s, 1, i]['P_B_full'] == 0) and (by[s, 1, i]['P_B_restricted'] == 0) for s in seeds for i in range(12)))
    h1[fam] = {}
    for j, g in enumerate([2, 4, 8]):
        xx = x[:, j, :].reshape(-1)
        yy = y[:, j, :].reshape(-1)
        metric = correlations(xx, yy)
        eligible = (np.abs(xx) > fr[j, 0]) & (np.abs(yy) > ff[j, 0])
        sig = float(np.mean(np.sign(xx[eligible]) == np.sign(yy[eligible])))
        pbr = np.mean(x[:, j, :] ** 2, axis=1)
        pbf = np.mean(y[:, j, :] ** 2, axis=1)
        fl = next((z for z in floor if z['family'] == fam and z['g'] == g))
        n = int(np.sum(np.isfinite(pbf) & (pbf > fl['tau_P'])))
        gg = next((v for v in geo if v['family'] == fam and v['g'] == g))
        cell = {'family': fam, 'activation': 'GELU' if fam == 'S1' else 'SwiGLU', 'g': g, 'seed_count': 8, 'coordinate_count': 96, 'resolved_seed_count': n, 'H1_pass': n >= 7, 'tau_P': fl['tau_P'], 'epsilon_B': fl['epsilon_B'], **{k: float(v) for k, v in metric.items()}, 'sign_agreement': sig, 'sign_eligible_coordinate_count': int(eligible.sum()), 'P_B_restricted_mean': float(pbr.mean()), 'P_B_full_mean': float(pbf.mean()), 'P_B_full_min': float(pbf.min()), 'P_B_full_max': float(pbf.max()), 'power_ratio_of_seed_means': float(pbf.mean() / pbr.mean()), 'power_ratio_seed_median': float(np.median(pbf / pbr)), 'power_ratio_seed_min': float(np.min(pbf / pbr)), 'power_ratio_seed_max': float(np.max(pbf / pbr)), 'K_A': gg['K_A'], 'K_C': gg['K_C'], 'K_M': gg['K_M'], 'rho_A': gg['rho_A'], 'rho_M': gg['rho_M'], 'H4_pass': gg['H4_pass']}
        cells.append(cell)
        h1[fam][str(g)] = {'resolved_seed_count': n, 'required': 7, 'pass': n >= 7}
    assert abs(point['scale_free_slope'] - np.mean([z['cosine'] for z in cells if z['family'] == fam])) <= 1e-12
    print('STAGE6', fam, json.dumps(family['metrics']), flush=True)
all_h0 = all(h0.values())
all_h1 = all((v['pass'] for f in h1.values() for v in f.values()))
all_h4 = all((v['H4_pass'] and (v['K_A'] == 0 and v['K_M'] == 0 if v['g'] == 1 else max(v['rho_A'], v['rho_M']) <= 1e-10) for v in geo))
criteria = all((v['H2_confirmation_criteria_pass'] for v in families.values()))
point_positive = all((v['metrics'][k]['estimate'] > (0.5 if k == 'sign_agreement' else 0) for v in families.values() for k in v['metrics']))
if not all_h0 or not all_h4:
    label = 'SCIENTIFIC_CONFLICT'
elif not all_h1:
    label = 'TECHNICAL_PROTOCOL_FAILURE'
elif criteria:
    label = 'FULL_TANGENT_CHANNEL_CONFIRMED'
elif point_positive:
    label = 'PARTIAL_CHANNEL_AGREEMENT'
else:
    label = 'RESTRICTED_OBSERVABLE_NOT_VALIDATED'
failed = [fam + ':' + k for fam, v in families.items() for k, z in v['metrics'].items() if not z['criterion_pass']]
results = {'classification': label, 'primary_freeze_utc': stamp(), 'preregistration_sha256': sha(OUT / 'FULL_TANGENT_PREREGISTRATION_v283M.md'), 'source_provenance_audit': 'PASS', 'historical_reproduction': read(OUT / 'HISTORICAL_REPRODUCTION_v283M.json'), 'implementation_check': read(OUT / 'FULL_TANGENT_IMPLEMENTATION_CHECK_v283M.json'), 'hypotheses': {'H0': {'pass': all_h0, 'families': h0}, 'H1': {'pass': all_h1, 'families': h1}, 'H2': {'confirmation_criteria_pass': criteria, 'failed_criteria': failed}, 'H3': {'status': 'ALL_SIX_CELLS_REPORTED_DIAGNOSTIC', 'cells': cells}, 'H4': {'pass': all_h4, 'geometry': geo}}, 'families': families, 'bootstrap': {'replicates': 20000, 'rng': 'PCG64', 'seed': 2026101201, 'unit': 'whole paired seed, retain 3 granularities x 12 coordinates', 'indices_sha256': sha(OUT / 'BOOTSTRAP_SEED_INDICES_v283M.npy'), 'ci': '95% percentile', 'RMS_normalization': 'recomputed per family x g within each bootstrap draw'}, 'primary_coordinate_records': 768, 'manuscript_v284M_warranted': label == 'FULL_TANGENT_CHANNEL_CONFIRMED', 'secondary_does_not_change_primary': True, 'optional_background_extension': 'OMITTED_OPTIONAL_NOT_PART_OF_PRIMARY_DECISION', 'interpretation_limit': 'Isotropic Euclidean full independently addressable child-expert tangent; no arbitrary residual/probe general B_ts or downstream utility conclusion.'}
write('FULL_TANGENT_RESULTS_v283M.json', results)
csvwrite('FULL_TANGENT_CELL_SUMMARY_v283M.csv', cells)
write('PRIMARY_CLASSIFICATION_FREEZE_v283M.json', {'utc': stamp(), 'classification': label, 'results_sha256': sha(OUT / 'FULL_TANGENT_RESULTS_v283M.json'), 'coordinates_sha256': sha(OUT / 'FULL_TANGENT_COORDINATES_v283M.csv'), 'cell_summary_sha256': sha(OUT / 'FULL_TANGENT_CELL_SUMMARY_v283M.csv'), 'failed_confirmation_criteria': failed})
write('RUN_STATUS.json', {'status': 'PRIMARY_CLASSIFICATION_FROZEN', 'stage': 6, 'classification': label, 'utc': stamp()})
print('PRIMARY CLASSIFICATION', label, flush=True)
recs = read(OUT / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')['primary_records']
secondary = []
sec_cells = []
sec_family = {}
probes = {}
rng2 = np.random.Generator(np.random.PCG64(2026101202))
for fam in ['S1', 'S2']:
    d, D = DIM[fam]
    normal = rng2.standard_normal((d, 4))
    Q, R = np.linalg.qr(normal, mode='reduced')
    assert np.max(np.abs(Q.T @ Q - np.eye(4))) < 1e-12
    probes[fam] = Q.tolist()
    np.save(OUT / f'{fam}_DIRECTIONAL_PROBES_v283M.npy', Q)
    r = next((r for r in recs if r['family'] == fam))
    h = np.load(r['bound_geometry_path'])['h']
    p = prepared(h, state(fam), fam)
    seeds = sorted({r['seed'] for r in recs if r['family'] == fam})
    for probeidx in range(4):
        bx = []
        by = []
        srow = []
        for g in [1, 2, 4, 8]:
            rs = [r for r in recs if r['family'] == fam and r['g'] == g]
            z = np.load(rs[0]['bound_geometry_path'])
            src, tgt, group = (z['src'], z['tgt'], z['group'])
            k, _, _ = kernel(p, fam, g, src, tgt, Q[:, probeidx])
            raw, (ka, kc, km) = center_child(k)
            xx = []
            yy = []
            for r in rs:
                hist = np.load(OUT / 'historical_reproduction' / f"{fam}_g{g}_seed{r['seed']}.npz")
                Br = hist['B_restricted']
                B = coords(hist['q_C'], kc, group)
                P = float(np.mean(B ** 2))
                xx.extend(Br)
                yy.extend(B)
                for i in range(12):
                    secondary.append({'family': fam, 'g': g, 'seed': r['seed'], 'probe': probeidx, 'coordinate_index': i, 'B_restricted': float(Br[i]), 'B_directional_full': float(B[i]), 'P_B_directional_full': P})
            if g == 1:
                assert all((v == 0 for v in yy))
                continue
            xx = np.array(xx)
            yy = np.array(yy)
            met = correlations(xx, yy)
            eligible = (np.abs(xx) > 1e-10) & (np.abs(yy) > 1e-10)
            sec_cells.append({'family': fam, 'g': g, 'probe': probeidx, **{k: float(v) for k, v in met.items()}, 'sign_agreement': float(np.mean(np.sign(xx[eligible]) == np.sign(yy[eligible]))), 'P_B_directional_mean': float(np.mean(yy ** 2))})
            bx.extend(xx)
            by.extend(yy)
        bx = np.array(bx)
        by = np.array(by)
        met = correlations(bx, by)
        eligible = (np.abs(bx) > 1e-10) & (np.abs(by) > 1e-10)
        srow.append({'probe': probeidx, **{k: float(v) for k, v in met.items()}, 'sign_agreement': float(np.mean(np.sign(bx[eligible]) == np.sign(by[eligible])))})
        sec_family.setdefault(fam, []).extend(srow)
    print('STAGE7', fam, sec_family[fam], flush=True)
csvwrite('SECONDARY_DIRECTIONAL_COORDINATES_v283M.csv', secondary)
csvwrite('SECONDARY_DIRECTIONAL_CELL_SUMMARY_v283M.csv', sec_cells)
write('SECONDARY_DIRECTIONAL_RESULTS_v283M.json', {'rng': 'PCG64', 'seed': 2026101202, 'Gaussian_shape_by_family': {'S1': [512, 4], 'S2': [576, 4]}, 'stream_order': ['S1', 'S2'], 'QR': 'float64 reduced QR first 4 Q columns; no sign postprocessing', 'probes': probes, 'families': sec_family, 'primary_classification_unchanged': label, 'cells': sec_cells})
assert sha(OUT / 'FULL_TANGENT_RESULTS_v283M.json') == read(OUT / 'PRIMARY_CLASSIFICATION_FREEZE_v283M.json')['results_sha256']
write('RUN_STATUS.json', {'status': 'SCIENCE_COMPLETE', 'stage': 7, 'classification': label, 'utc': stamp()})
print('STAGE7 COMPLETE PRIMARY UNCHANGED', flush=True)
