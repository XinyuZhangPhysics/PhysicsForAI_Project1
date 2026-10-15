# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
import time

def verify_freeze():
    fr = read(OUT / 'PROTOCOL_FREEZE_v284M.json')
    for p, h in fr['assets'].items():
        assert sha(p) == h, p
    return fr

def implementation(phase):
    fr = verify_freeze()
    binding = read(OUT / 'EXPERT_SIDE_SELECTION_BINDINGS_v284M.json')
    checks = []
    scalar = []
    for f in ['S1', 'S2']:
        item = next((x for x in binding['banks'] if x['family'] == f and x['phase'] == phase))
        z = np.load(item['path'])
        h = z['h'].astype(np.float64)
        s = state(f)
        p = prep(h, s, f)
        ids = np.array([np.flatnonzero(z['group'] == i)[0] for i in range(12)])
        src, tgt = (z['src'][ids], z['tgt'][ids])
        group = np.arange(12)
        for g in [1, 2, 4, 8]:
            point = next((x for x in binding['copied_points'] if x['family'] == f and x['g'] == g))
            seed = point['seed']
            v = direction(f, g, seed, 0)
            w = load_file(point['path'])['router.weight'].double().numpy()
            d, _ = derivative(p, f, g, v, src, tgt)
            if g in [1, 8]:
                ref = scalar_derivative(torch.from_numpy(h[src[0]]), torch.from_numpy(h[tgt[0]]), local_params(s, f, g, 0), local_direction(v, f), f)
                error = abs(d[0, 0] - ref)
                ok = error <= fr['closure_atol'] + fr['closure_rtol'] * abs(ref)
                row = {'family': f, 'g': g, 'phase': phase, 'pair_index': int(ids[0]), 'child': 0, 'seed': seed, 'direction': 0, 'analytic': float(d[0, 0]), 'independent_scalar_double_backprop': ref, 'abs_error': float(error), 'pass': ok}
                scalar.append(row)
                print('SCALAR', phase, f, g, 'error', error, 'PASS', ok, flush=True)
            for cls in ['parent'] + (['child'] if g > 1 else []):
                dk = field_for(d, f, g, v, cls)
                for bg in [1, 0]:
                    q, u, root, pref = qfield(h, w, g, bg, src, tgt)
                    analytic = actual(dk, u, root, group)
                    slopes = []
                    for eps in fr['amplitudes']:
                        bs = []
                        for sign in [1, -1]:
                            k = np.empty((12, 4 * g))
                            with torch.no_grad():
                                for parent in range(4):
                                    for c in range(g):
                                        coeff = v['parent'][parent] / math.sqrt(g) if cls == 'parent' else v['child'][c] / 2
                                        params = tuple((t + sign * eps * coeff * vv for t, vv in zip(local_params(s, f, g, c), local_direction(v, f))))
                                        k[:, parent * g + c] = torch_kernel(torch.from_numpy(h[src]), torch.from_numpy(h[tgt]), params, f).numpy()
                            bs.append(actual(k, u, root, group))
                        slopes.append((bs[0] - bs[1]) / (2 * eps))
                    rich = (4 * slopes[-1] - slopes[-2]) / 3
                    err = float(np.max(abs(rich - analytic)))
                    con = float(np.max(abs(slopes[-1] - slopes[-2])))
                    tol = fr['fd_atol'] + fr['fd_rtol'] * float(np.max(abs(analytic)))
                    checks.append({'family': f, 'g': g, 'seed': seed, 'direction': 0, 'class': cls, 's': bg, 'phase': phase, 'pair_indices': ids.tolist(), 'analytic': analytic.tolist(), 'fd_slopes': [x.tolist() for x in slopes], 'richardson': rich.tolist(), 'richardson_max_abs_error': err, 'small_amplitude_convergence_error': con, 'tolerance': tol, 'pass': err <= tol and con <= tol})
            print('FD', phase, f, g, 'PASS', all((x['pass'] for x in checks if x['family'] == f and x['g'] == g)), flush=True)
    result = {'timestamp': stamp(), 'phase': phase, 'scalar_checks': scalar, 'finite_difference_checks': checks, 'pass': all((x['pass'] for x in scalar + checks))}
    write(f'{phase.upper()}_IMPLEMENTATION_CHECK_v284M.json', result)
    assert result['pass'], 'implementation cross-check failed'
    return result
if __name__ == '__main__':
    implementation(sys.argv[1])
