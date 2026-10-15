# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import statistics
checks = []

def verify(name, ok, detail=None):
    checks.append({'check': name, 'pass': bool(ok), 'detail': detail})
    if not ok:
        write('FINAL_SCIENTIFIC_VERIFICATION_v283M.json', {'pass': False, 'checks': checks})
        raise AssertionError(name)
results = read(OUT / 'FULL_TANGENT_RESULTS_v283M.json')
freeze = read(OUT / 'PRIMARY_CLASSIFICATION_FREEZE_v283M.json')
man = read(OUT / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')
coordsraw = list(csv.DictReader((OUT / 'FULL_TANGENT_COORDINATES_v283M.csv').open()))
index = np.load(OUT / 'BOOTSTRAP_SEED_INDICES_v283M.npy')
seedcheck = np.random.Generator(np.random.PCG64(2026101201)).integers(0, 8, size=(20000, 8))
verify('bootstrap draws reproduce frozen RNG', np.array_equal(index, seedcheck))
verify('primary JSON frozen before secondary', sha(OUT / 'FULL_TANGENT_RESULTS_v283M.json') == freeze['results_sha256'])
verify('all source hashes unchanged at completion', all((sha(z['path']) == z['expected_sha256'] for z in man['entries'])))
verify('immutable preregistration unchanged', sha(OUT / 'FULL_TANGENT_PREREGISTRATION_v283M.md') == (OUT / 'FULL_TANGENT_PREREGISTRATION_v283M.sha256').read_text().split()[0])
verify('complete 768 signed-coordinate records', len(coordsraw) == 768)

def ranks(v):
    ordered = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    start = 0
    while start < len(v):
        stop = start + 1
        while stop < len(v) and v[ordered[stop]] == v[ordered[start]]:
            stop += 1
        for j in range(start, stop):
            out[ordered[j]] = (start + stop + 1) / 2
        start = stop
    return out

def scalar_stats(x, y):
    dot = math.fsum((a * b for a, b in zip(x, y)))
    nx = math.fsum((a * a for a in x))
    ny = math.fsum((a * a for a in y))
    return {'cosine': dot / math.sqrt(nx * ny), 'pearson': statistics.correlation(x, y), 'spearman': statistics.correlation(ranks(x), ranks(y)), 'sign_agreement': sum(((a > 0) == (b > 0) for a, b in zip(x, y) if abs(a) > 1e-10 and abs(b) > 1e-10)) / sum((abs(a) > 1e-10 and abs(b) > 1e-10 for a, b in zip(x, y)))}
for fam in ['S1', 'S2']:
    rows = [r for r in coordsraw if r['family'] == fam]
    seeds = results['families'][fam]['seeds']
    by = {(int(r['seed']), int(r['g']), int(r['coordinate_index'])): r for r in rows}
    xx = [float(by[s, g, i]['B_restricted']) for s in seeds for g in [2, 4, 8] for i in range(12)]
    yy = [float(by[s, g, i]['B_full']) for s in seeds for g in [2, 4, 8] for i in range(12)]
    stats = scalar_stats(xx, yy)
    for k, v in stats.items():
        verify(f'{fam} independent primary {k}', abs(v - results['families'][fam]['metrics'][k]['estimate']) < 1e-12, abs(v - results['families'][fam]['metrics'][k]['estimate']))
    draws = np.load(OUT / f'{fam}_BOOTSTRAP_DISTRIBUTIONS_v283M.npz')
    for iteration in [0, 1, 100, 499, 9999, 19999]:
        ss = [seeds[j] for j in index[iteration]]
        x = [float(by[s, g, i]['B_restricted']) for s in ss for g in [2, 4, 8] for i in range(12)]
        y = [float(by[s, g, i]['B_full']) for s in ss for g in [2, 4, 8] for i in range(12)]
        v = scalar_stats(x, y)
        slopes = []
        for g in [2, 4, 8]:
            gx = [float(by[s, g, i]['B_restricted']) for s in ss for i in range(12)]
            gy = [float(by[s, g, i]['B_full']) for s in ss for i in range(12)]
            slopes.append(scalar_stats(gx, gy)['cosine'])
        v['scale_free_slope'] = statistics.mean(slopes)
        verify(f'{fam} independent whole-seed draw {iteration}', all((abs(val - draws[k][iteration]) < 1e-12 for k, val in v.items())), max((abs(val - draws[k][iteration]) for k, val in v.items())))
    for k in draws.files:
        ci = np.percentile(draws[k], [2.5, 97.5])
        verify(f'{fam} {k} percentile CI', np.array_equal(ci, np.array(results['families'][fam]['metrics'][k]['ci95'])))
    for g in [1, 2, 4, 8]:
        geometry = np.load(OUT / 'full_geometry' / f'{fam}_g{g}.npz')
        kc = geometry['kappa_C']
        verify(f'{fam} g{g} finite geometry', np.isfinite(kc).all())
        for s in seeds:
            pair = np.load(OUT / 'full_geometry' / f'{fam}_g{g}_seed{s}_pair_response.npz')
            full = np.array([pair['full_pair'][pair['group'] == i].mean() for i in range(12)])
            restricted = np.array([pair['restricted_pair'][pair['group'] == i].mean() for i in range(12)])
            v = np.array([float(by[s, g, i]['B_full']) for i in range(12)])
            u = np.array([float(by[s, g, i]['B_restricted']) for i in range(12)])
            verify(f'{fam} g{g} seed{s} coordinate regeneration', np.array_equal(full, v) and np.array_equal(restricted, u) and (abs(float(np.mean(full ** 2)) - float(by[s, g, 0]['P_B_full'])) < 1e-16))
    Q = np.load(OUT / f'{fam}_DIRECTIONAL_PROBES_v283M.npy')
    verify(f'{fam} probes orthonormal', np.max(np.abs(Q.T @ Q - np.eye(4))) < 1e-12)
expected = 'PARTIAL_CHANNEL_AGREEMENT'
verify('exact frozen rule classification', results['classification'] == expected and results['hypotheses']['H0']['pass'] and results['hypotheses']['H1']['pass'] and results['hypotheses']['H4']['pass'] and all((results['families'][f]['metrics'][k]['estimate'] > (0.5 if k == 'sign_agreement' else 0) for f in ['S1', 'S2'] for k in results['families'][f]['metrics'])) and results['families']['S1']['H2_confirmation_criteria_pass'] and (not results['families']['S2']['H2_confirmation_criteria_pass']))
verify('secondary does not alter label', read(OUT / 'SECONDARY_DIRECTIONAL_RESULTS_v283M.json')['primary_classification_unchanged'] == expected)
write('FINAL_SCIENTIFIC_VERIFICATION_v283M.json', {'utc': stamp(), 'pass': True, 'checks': checks, 'checks_passed': len(checks), 'independent_statistics_method': 'Python statistics correlation and scalar tie-averaged ranks, separate from vectorized implementation; 6 fixed draws per family cross-checked', 'historical_source_files_modified': False})
print('FINAL SCIENCE VERIFICATION', len(checks), 'PASS')
