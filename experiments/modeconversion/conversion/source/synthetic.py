# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from experiment import *

def main():
    checks = []
    for f in ['S1', 'S2']:
        for g in [2, 4, 8]:
            for seed in seeds(f):
                w = weight(f, g, seed)
                rng = np.random.default_rng(np.random.SeedSequence([20261011963, g, seed]))
                h = rng.normal(size=(16, w.shape[1]))
                z = h @ w.T
                z -= z.mean(1, keepdims=True)
                src = np.arange(12288) % 16
                tgt = (np.arange(12288) + 3) % 16
                child = rng.normal(size=(12288, g))
                child -= child.mean(1, keepdims=True)
                cache = {'src': src, 'tgt': tgt, 'kappa_C': np.tile(child, (1, 4))}
                pref = reference(z, cache, g)
                v, es, _ = directions(f, g, seed)[0]
                t = h @ v
                zero = response(np.zeros((16, 4 * g)), cache, pref, g)[3]
                err = 0.0
                if seed == seeds(f)[0]:
                    for s in GRID:
                        for sector, e in es.items():
                            a = np.zeros(12) if sector == 'A' else chain(z, t, e, s, cache, pref, g)[0]
                            ag = autograd(z, t, e, s, cache, pref, g)
                            err = max(err, float(np.max(abs(a - ag))))
                            assert np.max(abs(a - ag)) < 1e-10
                assert zero == 0 and rms(chain(z, t, es['M'], 0, cache, pref, g)[0]) < 1e-10 and (rms(chain(z, t, es['C'], 0, cache, pref, g)[0]) > 0)
                checks.append({'family': f, 'g': g, 'seed': seed, 'probe_type': 'synthetic only, no corpus documents', 'P_ref_positive': pref, 'qC_zero': zero, 'chain_autograd_max_abs': err})
    write(ROOT / 'preflight/SYNTHETIC_ANALYTIC_CHECKS.json', checks)
    print('SYNTHETIC_ANALYTIC_GATE_PASS', len(checks), flush=True)
if __name__ == '__main__':
    main()
