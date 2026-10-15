# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
from kernels import *
torch.set_num_threads(1)
assert read(OUT / 'HISTORICAL_REPRODUCTION_v283M.json')['pass']
assert not (OUT / 'FULL_TANGENT_COORDINATES_v283M.csv').exists()
plan = read(OUT / 'IMPLEMENTATION_PLAN_FREEZE_v283M.json')
man = read(OUT / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')
recs = man['primary_records']
pre = {}
states = {}
checks = []
for fam in ['S1', 'S2']:
    states[fam] = state(fam)
    r = next((r for r in recs if r['family'] == fam))
    h = np.load(r['bound_geometry_path'])['h']
    pre[fam] = prepared(h, states[fam], fam)
for item in plan['subset']:
    fam, g, c, idx = (item['family'], item['g'], item['child'], item['pair_index'])
    r = next((r for r in recs if r['family'] == fam and r['g'] == g))
    v = np.load(r['bound_geometry_path'])
    src, tgt = (v['src'], v['tgt'])
    full, out, _ = kernel(pre[fam], fam, g, src[idx:idx + 1], tgt[idx:idx + 1])
    ref, refout = scalar_gradient_reference(pre[fam]['h'][src[idx]], pre[fam]['h'][tgt[idx]], states[fam], fam, g, c)
    fe = abs(full[0, c] - ref)
    oe = abs(out[0, c] - refout)
    histphi = features(pre[fam]['h'], states[fam], fam, 'mps', torch.float32).cpu().double().numpy()
    m = DIM[fam][1] // g
    sl = slice(c * m, (c + 1) * m)
    hist = float(np.mean(histphi[src[idx], sl] * histphi[tgt[idx], sl]))
    he = abs(out[0, c] - hist)
    same_precision_ref = down_reference(histphi[src[idx], sl], histphi[tgt[idx], sl], states[fam], fam, g, c)
    same_precision_error = abs(hist - same_precision_ref)
    row = {**item, 'optimized_full': float(full[0, c]), 'autograd_full': ref, 'full_abs_error': fe, 'full_relative_error': fe / max(abs(ref), 1e-300), 'optimized_down': float(out[0, c]), 'autograd_down': refout, 'down_abs_error': oe, 'down_relative_error': oe / max(abs(refout), 1e-300), 'historical_restricted_kernel': hist, 'historical_restriction_abs_error': he, 'historical_restriction_relative_error': he / max(abs(hist), 1e-300), 'historical_float32_down_autograd': same_precision_ref, 'same_feature_historical_identity_abs_error': same_precision_error, 'same_feature_historical_identity_relative_error': same_precision_error / max(abs(hist), 1e-300), 'cross_precision_bridge_is_diagnostic_only': True}
    row['pass'] = fe <= 1e-12 and row['full_relative_error'] <= 1e-10 and (oe <= 1e-12) and (row['down_relative_error'] <= 1e-10) and (same_precision_error <= 1e-12) and (row['same_feature_historical_identity_relative_error'] <= 1e-10)
    checks.append(row)
    write('FULL_TANGENT_IMPLEMENTATION_CHECK_v283M.json', {'utc': stamp(), 'status': 'IN_PROGRESS', 'checks': checks})
    print('CHECK', fam, g, c, 'full errors', fe, row['full_relative_error'], 'historical', he, row['historical_restriction_relative_error'], flush=True)
    if not row['pass']:
        write('RUN_STATUS.json', {'status': 'TECHNICAL_PROTOCOL_FAILURE', 'stage': 4, 'failed_check': row})
        raise RuntimeError('mandatory implementation check failed')
err = max((c['full_abs_error'] for c in checks))
write('FULL_TANGENT_IMPLEMENTATION_CHECK_v283M.json', {'utc': stamp(), 'status': 'PASS', 'checks': checks, 'max_full_abs_error': err, 'max_full_relative_error': max((c['full_relative_error'] for c in checks)), 'max_output_abs_error': max((c['down_abs_error'] for c in checks)), 'max_historical_restriction_abs_error': max((c['historical_restriction_abs_error'] for c in checks)), 'derivation': 'GELU: output feature/m + (hx dot hy + 1) sum_a ||g V_a||^2 gelu_prime(zx_a)gelu_prime(zy_a)/(d m). SwiGLU: output feature/m + (hx dot hy) sum_a ||g V_a||^2 [gate_derivative_x*up_x*gate_derivative_y*up_y+silu_x*silu_y]/(d m). Common S1 output bias excluded.'})
floors = []
for fam in ['S1', 'S2']:
    for g in [1, 2, 4, 8]:
        rs = [r for r in recs if r['family'] == fam and r['g'] == g]
        l1 = []
        for r in rs:
            z = np.load(OUT / 'historical_reproduction' / f"{fam}_g{g}_seed{r['seed']}.npz")
            q = z['q_C']
            group = z['group']
            l1.extend((np.mean(np.sum(np.abs(q[group == i]), axis=1)) for i in range(12)))
        eps = 4 * err * max(l1)
        floors.append({'family': fam, 'g': g, 'epsilon_B': eps, 'tau_P': max(1e-20, 100 * eps * eps), 'restricted_sign_floor': 1e-10, 'full_sign_floor': max(1e-10, eps), 'max_group_mean_q_C_L1': float(max(l1)), 'kernel_error_bound': err, 'conservative_factor': 4})
write('NUMERICAL_FLOORS_FREEZE_v283M.json', {'utc_before_primary_coordinates': stamp(), 'cells': floors, 'sign_floor_policy': 'fixed max(1e-10, propagated floor); historical reproduction error zero', 'precision': 'independent analytic/autograd f64; historical bridge f32/MPS'})
(OUT / 'full_geometry').mkdir(exist_ok=True)
rows = []
geom = []
for fam in ['S1', 'S2']:
    for g in [1, 2, 4, 8]:
        rs = [r for r in recs if r['family'] == fam and r['g'] == g]
        z = np.load(rs[0]['bound_geometry_path'])
        src, tgt, group = (z['src'], z['tgt'], z['group'])
        full, out, inp = kernel(pre[fam], fam, g, src, tgt)
        raw, (ka, kc, km) = center_child(full)
        powers = {name: float(np.mean(np.sum(v * v, axis=1))) for name, v in [('A', ka), ('C', kc), ('M', km)]}
        ratios = {name: powers[name] / powers['C'] if powers['C'] > 0 else None for name in ['A', 'M']}
        j4 = np.ones((4, 4)) / 4
        jg = np.ones((g, g)) / g
        p4 = np.eye(4) - j4
        pg = np.eye(g) - jg
        P = {'A': np.kron(p4, jg), 'C': np.kron(j4, pg), 'M': np.kron(p4, pg)}
        pcheck = {name: float(np.max(np.abs(raw @ P[name] - v))) for name, v in [('A', ka), ('C', kc), ('M', km)]}
        passed = np.array_equal(kc, np.zeros_like(kc)) if g == 1 else max(ratios.values()) <= 1e-10
        path = OUT / 'full_geometry' / f'{fam}_g{g}.npz'
        np.savez_compressed(path, full_child=full, output_projection_child=out, input_projection_child=inp, kappa_C=kc, src=src, tgt=tgt, group=group)
        cachehash = sha(path)
        geometry = {'family': fam, 'g': g, 'K_A': powers['A'], 'K_C': powers['C'], 'K_M': powers['M'], 'rho_A': ratios['A'], 'rho_M': ratios['M'], 'projector_max_abs_errors': pcheck, 'H4_pass': passed, 'cache': str(path.relative_to(OUT)), 'sha256': cachehash}
        geom.append(geometry)
        if not passed:
            write('RUN_STATUS.json', {'status': 'TECHNICAL_PROTOCOL_FAILURE', 'stage': 5, 'detail': geometry})
            raise RuntimeError('sector invariant failure')
        floor = next((v for v in floors if v['family'] == fam and v['g'] == g))
        for r in rs:
            qpath = OUT / 'historical_reproduction' / f"{fam}_g{g}_seed{r['seed']}.npz"
            v = np.load(qpath)
            q = v['q_C']
            B = coords(q, kc, group)
            Br = v['B_restricted']
            pb = float(np.mean(B * B))
            pbr = float(np.mean(Br * Br))
            keys = read(OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.json')['families'][fam]['ordered_pair_keys']
            np.savez_compressed(OUT / 'full_geometry' / f"{fam}_g{g}_seed{r['seed']}_pair_response.npz", restricted_pair=np.sum(q * v['kappa_restricted'], axis=1), full_pair=np.sum(q * kc, axis=1), group=group)
            for i in range(12):
                rows.append({'family': fam, 'activation': r['activation'], 'g': g, 'arm': r['arm'], 'seed': r['seed'], 'domain_pair': '->'.join(keys[i]), 'coordinate_index': i, 'B_restricted': float(Br[i]), 'B_full': float(B[i]), 'B_restricted_power': float(Br[i] ** 2), 'B_full_power': float(B[i] ** 2), 'P_B_restricted': pbr, 'P_B_full': pb, 'router_source_hash': r['router_checkpoint_sha256'], 'q_C_cache_hash': sha(qpath), 'historical_response_hash': r['source_sha256'], 'full_geometry_cache_hash': cachehash, 'K_A': powers['A'], 'K_C': powers['C'], 'K_M': powers['M'], 'rho_A': ratios['A'], 'rho_M': ratios['M'], 'epsilon_B': floor['epsilon_B'], 'tau_P': floor['tau_P'], 'restricted_sign_floor': floor['restricted_sign_floor'], 'full_sign_floor': floor['full_sign_floor']})
        print('STAGE5', fam, g, 'K_C', powers['C'], 'A/M', ratios, flush=True)
csvwrite('FULL_TANGENT_COORDINATES_v283M.csv', rows)
write('FULL_TANGENT_GEOMETRY_AUDIT_v283M.json', {'cells': geom, 'all_H4_pass': all((r['H4_pass'] for r in geom))})
write('RUN_STATUS.json', {'status': 'PRIMARY_COORDINATES_COMPLETE', 'stage': 5, 'utc': stamp(), 'coordinate_records': len(rows), 'classification_not_yet_computed': True})
print('STAGE5 COMPLETE', len(rows), flush=True)
