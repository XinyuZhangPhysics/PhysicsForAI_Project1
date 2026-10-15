# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *

def main():
    from experiment import verify_freeze
    fr = verify_freeze()
    amps = np.array(fr['amplitudes'])
    grid = fr['background_grid']
    coef = list(csv.DictReader((ROOT / 'audit/COEFFICIENTS.csv').open()))
    raw = list(csv.DictReader((ROOT / 'audit/RAW.csv').open()))
    assert len(raw) == 51840
    lookup = {}
    for r in raw:
        key = (r['family'], int(r['g']), int(r['seed']), int(r['direction']), r['sector'], float(r['s']), float(r['delta']))
        assert key not in lookup
        lookup[key] = r
    err = 0.0
    prefs = {}
    for c in coef:
        base = (c['family'], int(c['g']), int(c['seed']), int(c['direction']), c['sector'], float(c['s']))
        odd = []
        for d in amps:
            plus = lookup[base + (float(d),)]
            minus = lookup[base + (-float(d),)]
            k = base[:3]
            p = float(plus['P_ref'])
            assert p > 0 and p == float(minus['P_ref'])
            if k in prefs:
                assert p == prefs[k]
            else:
                prefs[k] = p
            odd.append(np.array([float(plus[f'B_{i:02}']) - float(minus[f'B_{i:02}']) for i in range(12)]) / 2)
        fitted = np.linalg.lstsq(np.array([amps, amps ** 3]).T, np.array(odd), rcond=None)[0][0]
        actual = np.array([float(c[f'b1_{i:02}']) for i in range(12)])
        err = max(err, float(np.max(abs(actual - fitted))))
        assert rms(actual) == float(c['Lambda'])
    assert err < 1e-11 and len(prefs) == 48
    primary = list(csv.DictReader((ROOT / 'outputs/BACKGROUND_MODE_CONVERSION_R2_PRIMARY.csv').open()))
    assert len(primary) == 48
    for p in primary:
        for sector in ['A', 'C', 'M']:
            cc = [c for c in coef if (c['family'], c['g'], c['seed'], c['sector'], float(c['s'])) == (p['family'], p['g'], p['seed'], sector, 0)]
            assert len(cc) == 4 and abs(np.mean([float(c['Lambda']) for c in cc]) - float(p[f'Lambda_{sector}_0'])) < 1e-15
    sets = []
    for item in bind()['fresh_banks'].values():
        assert sha(item['path']) == item['sha256']
        ss = set((json.loads(l)['text_sha256'] for l in Path(item['path']).open()))
        assert len(ss) == 2048
        sets.append(ss)
    assert len(set.union(*sets)) == 8192
    write(ROOT / 'outputs/ESTIMATOR_VERIFICATION.json', {'timestamp': now(), 'signed_records': len(raw), 'signed_coordinates': len(raw) * 12, 'unique_signed_keys': len(lookup), 'coefficient_rows': len(coef), 'unscaled_independent_OLS_max_abs_difference': err, 'fixed_reference_groups': len(prefs), 'primary_seed_rows': len(primary), 'direction_aggregation_verified': True, 'fresh_document_disjointness_verified': True, 'scientific_freeze_unchanged': True})
    print('ESTIMATOR_VERIFICATION_PASS', len(raw), err, flush=True)
if __name__ == '__main__':
    main()
