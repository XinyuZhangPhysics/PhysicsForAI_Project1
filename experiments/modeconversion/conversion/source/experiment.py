# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *

def project(x, g):
    y = x.reshape(-1, 4, g)
    return np.broadcast_to(y.mean(1, keepdims=True) - y.mean((1, 2), keepdims=True), y.shape).reshape(-1, 4 * g)

def probability(logits):
    z = logits - logits.max(1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(1, keepdims=True)

def response(logits, cache, pref, g):
    assert pref > 0 and np.isfinite(pref)
    p = probability(logits)
    u = p[cache['src']] * p[cache['tgt']]
    center = u - u.mean(1, keepdims=True)
    qc = project(center, g) / np.sqrt(pref)
    B = (qc * cache['kappa_C']).sum(1).reshape(12, 1024).mean(1)
    pq = float(np.mean((center * center).sum(1) / (4 * g - 1)))
    assert np.isfinite(B).all()
    return (B, pq, float(p.min()), float(np.max(abs(qc))))

def reference(z, cache, g):
    p = probability(np.repeat(z, g, axis=1))
    u = p[cache['src']] * p[cache['tgt']]
    u -= u.mean(1, keepdims=True)
    pref = float(np.mean((u * u).sum(1) / (4 * g - 1)))
    assert pref > 0 and np.isfinite(pref)
    return pref

def independent_response64(logits, cache, pref, g):
    p = torch.from_numpy(logits).softmax(1)
    u = p[cache['src']] * p[cache['tgt']]
    u -= u.mean(1, keepdim=True)
    q = u.reshape(-1, 4, g)
    qc = (q.mean(1, keepdim=True) - q.mean((1, 2), keepdim=True)).expand_as(q).reshape(-1, 4 * g) / np.sqrt(pref)
    return (qc * torch.from_numpy(cache['kappa_C'])).sum(1).reshape(12, 1024).mean(1).numpy()

def chain(z, t, e, s, cache, pref, g):
    alpha = probability(s * z)
    x = cache['src']
    y = cache['tgt']
    e = e.reshape(4, g)
    field = (t[x] + t[y])[:, None] / (4 * g * g) * (alpha[x] * alpha[y]) @ e
    field = np.tile(field, (1, 4)) / np.sqrt(pref)
    return ((field * cache['kappa_C']).sum(1).reshape(12, 1024).mean(1), field)

def mixed_chain(z, t, e, cache, pref, g):
    x = cache['src']
    y = cache['tgt']
    field = (z[x] + z[y]) @ e.reshape(4, g) * (t[x] + t[y])[:, None] / (64 * g * g * np.sqrt(pref))
    field = np.tile(field, (1, 4))
    return (field * cache['kappa_C']).sum(1).reshape(12, 1024).mean(1)

def autograd(z, t, e, s, cache, pref, g, mixed=False, dtype=torch.float64):
    zt = torch.tensor(z, dtype=dtype)
    tt = torch.tensor(t, dtype=dtype)
    et = torch.tensor(e, dtype=dtype)
    kc = torch.tensor(cache['kappa_C'], dtype=dtype)
    st = torch.tensor(s, dtype=dtype, requires_grad=mixed)
    dt = torch.tensor(0.0, dtype=dtype, requires_grad=True)
    logits = (st * zt).repeat_interleave(g, 1) + dt * tt[:, None] * et[None, :]
    p = logits.softmax(1)
    u = p[cache['src']] * p[cache['tgt']]
    u = u - u.mean(1, keepdim=True)
    q = u.reshape(-1, 4, g)
    qc = (q.mean(1, keepdim=True) - q.mean((1, 2), keepdim=True)).expand_as(q).reshape(-1, 4 * g) / np.sqrt(pref)
    B = (qc * kc).sum(1).reshape(12, 1024).mean(1)
    out = []
    for i in range(12):
        grad = torch.autograd.grad(B[i], dt, retain_graph=True, create_graph=mixed)[0]
        if mixed:
            grad = torch.autograd.grad(grad, st, retain_graph=True)[0]
        out.append(float(grad.detach()))
    return np.array(out)

def load(f, g, m):
    if m == 'audit':
        verify_freeze()
        assert (ROOT / 'audit/AUDIT_OPENING_RECORD.json').exists()
    meta = read(ROOT / m / f'{f}_BANK_BUILD.json')
    item = meta['geometry'][str(g)]
    assert sha(item['path']) == item['sha256']
    with np.load(item['path']) as zz:
        cache = {k: zz[k] for k in zz.files}
    assert np.array_equal(cache['group'], np.repeat(np.arange(12), 1024))
    assert len(cache['h']) == 2048
    return cache

def run(m):
    fr = verify_freeze() if m == 'audit' else None
    amps = np.array(fr['amplitudes']) if fr else AMPS
    grid = np.array(fr['background_grid']) if fr else GRID
    rows = []
    coeff = []
    controls = []
    chains = []
    dirmanifest = []
    floors = {}
    references = []
    scalings = []
    for f in ['S1', 'S2']:
        for g in [2, 4, 8]:
            cache = load(f, g, m)
            h = cache['h'].astype(np.float64)
            cell = f'{f}_g{g}'
            roundoff = 1e-12
            float32error = 0.0
            fiterror = 0.0
            mixedfiterror = 0.0
            for seed in seeds(f):
                w = weight(f, g, seed)
                z = h @ w.T
                z -= z.mean(1, keepdims=True)
                pref = reference(z, cache, g)
                references.append({'family': f, 'g': g, 'seed': seed, 'bank': m, 'P_ref': pref, 'definition': 'Pq(s=1,delta=0); frozen within this bank/seed/g'})
                assert response(np.zeros((2048, 4 * g)), cache, pref, g)[3] == 0
                out = cache['child_output'][:32]
                baseline = out.sum(1)
                maxforward = 0.0
                for s in grid:
                    alpha = probability(s * z[:32])
                    p = np.repeat(alpha / g, g, axis=1)
                    represented = np.einsum('npc,npcd->nd', p.reshape(32, 4, g), np.broadcast_to(g * out[:, None, :, :], (32, 4, g, out.shape[-1])))
                    maxforward = max(maxforward, float(np.max(abs(represented - baseline))))
                assert maxforward < 1e-09 * max(1, float(np.max(abs(baseline))))
                for r, (v, es, manifest) in enumerate(directions(f, g, seed)):
                    dirmanifest.append({'family': f, 'g': g, 'seed': seed, 'direction': r, **manifest})
                    t = h @ v
                    b1s = {sector: [] for sector in es}
                    minprob = 1.0
                    maxlogit = 0.0
                    maxchain = 0.0
                    maxrelative = 0.0
                    maxAfield = 0.0
                    for s in grid:
                        for sector, e in es.items():
                            vals = []
                            vals32 = []
                            base = np.repeat(s * z, g, axis=1)
                            pert = t[:, None] * e[None, :]
                            for d in amps:
                                bs = []
                                for sign in [1, -1]:
                                    b, pq, pmin, qmax = response(base + sign * d * pert, cache, pref, g)
                                    bs.append(b)
                                    minprob = min(minprob, pmin)
                                    maxlogit = max(maxlogit, float(np.max(abs(d * pert))))
                                    if sector == 'A':
                                        maxAfield = max(maxAfield, qmax)
                                    if m == 'calibration' and s in [0, 1] and (d == amps[0]):
                                        independent = independent_response64(base + sign * d * pert, cache, pref, g)
                                        roundoff = max(roundoff, 32 * float(np.max(abs(independent - b))))
                                    rows.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'sector': sector, 's': float(s), 'delta': float(sign * d), 'P_ref': pref, 'Pq_diagnostic': pq, **{f'B_{i:02}': float(b[i]) for i in range(12)}})
                                vals.append((bs[0] - bs[1]) / 2)
                                if m == 'calibration' and s in [0, 1]:
                                    bplus = response((base + d * pert).astype(np.float32), cache, pref, g)[0]
                                    bminus = response((base - d * pert).astype(np.float32), cache, pref, g)[0]
                                    vals32.append((bplus - bminus) / 2)
                            vals = np.array(vals)
                            design = np.array([amps / amps[-1], (amps / amps[-1]) ** 3]).T
                            fit = np.linalg.lstsq(design, vals, rcond=None)[0]
                            b1 = fit[0] / amps[-1]
                            b3 = fit[1] / amps[-1] ** 3
                            b1s[sector].append(b1)
                            if vals32:
                                b132 = np.linalg.lstsq(design, np.array(vals32), rcond=None)[0][0] / amps[-1]
                                float32error = max(float32error, float(np.max(abs(b132 - b1))))
                            analytic = np.zeros(12) if sector == 'A' else chain(z, t, e, s, cache, pref, g)[0]
                            fiterror = max(fiterror, rms(b1 - analytic) / max(rms(analytic), 1e-08)) if sector != 'A' and (sector != 'M' or s != 0) else fiterror
                            if m == 'calibration':
                                ag = autograd(z, t, e, s, cache, pref, g)
                                err = float(np.max(abs(ag - analytic)))
                                rel = rms(ag - analytic) / max(rms(analytic), 1e-10)
                                maxchain = max(maxchain, err)
                                maxrelative = max(maxrelative, rel)
                                assert err <= 1e-10 + 1e-08 * float(np.max(abs(analytic)))
                                if s == 0 or s == 1:
                                    ag32 = autograd(z, t, e, s, cache, pref, g, dtype=torch.float32)
                                    float32error = max(float32error, float(np.max(abs(ag32 - ag))))
                            elif sector in ['C', 'M']:
                                ag = autograd(z, t, e, s, cache, pref, g)
                                err = float(np.max(abs(ag - analytic)))
                                maxchain = max(maxchain, err)
                                assert err <= fr['chain_atol'] + fr['chain_rtol'] * float(np.max(abs(analytic)))
                            chains.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'sector': sector, 's': float(s), 'analytic': analytic.tolist(), 'b1_fit': b1.tolist(), 'autograd': ag.tolist() if m == 'calibration' or sector != 'A' else None})
                            coeff.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'sector': sector, 's': float(s), 'Lambda': rms(b1), **{f'b1_{i:02}': float(b1[i]) for i in range(12)}, **{f'b3_{i:02}': float(b3[i]) for i in range(12)}})
                            powers = np.sqrt(np.mean(vals ** 2, axis=1))
                            exponent = None
                            if sector == 'C' or (sector == 'M' and s != 0):
                                exponent = float(np.polyfit(np.log(amps), np.log(powers), 1)[0])
                            scalings.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'sector': sector, 's': float(s), 'odd_response_exponent': exponent, 'status': 'resolved' if exponent is not None else 'structural_null_no_log_fit'})
                    mask = np.abs(grid) <= 0.25
                    ss = grid[mask]
                    mfit = np.linalg.lstsq(np.array([ss, ss ** 3]).T, np.array(b1s['M'])[mask], rcond=None)[0]
                    xi = rms(mfit[0])
                    mc = mixed_chain(z, t, es['M'], cache, pref, g)
                    mag = autograd(z, t, es['M'], 0.0, cache, pref, g, mixed=True)
                    assert np.max(abs(mc - mag)) < 1e-10 + 1e-08 * np.max(abs(mc))
                    mixedfiterror = max(mixedfiterror, rms(mfit[0] - mc) / max(rms(mc), 1e-10))
                    controls.append({'family': f, 'g': g, 'seed': seed, 'direction': r, 'P_ref': pref, 'qC_at_symmetry': 0.0, 'A_field_max': maxAfield, 'function_preservation_max_abs': maxforward, 'min_probability': minprob, 'max_logit_intervention': maxlogit, 'chain_max_abs': maxchain, 'chain_max_relative': maxrelative, 'M_exact_at_zero': chain(z, t, es['M'], 0.0, cache, pref, g)[0].tolist(), 'M_autograd_at_zero': autograd(z, t, es['M'], 0.0, cache, pref, g).tolist(), 'C_exact_at_zero_Lambda': rms(chain(z, t, es['C'], 0.0, cache, pref, g)[0]), 'Xi_M': xi, 'm1': mfit[0].tolist(), 'mixed_chain': mc.tolist(), 'mixed_autograd': mag.tolist(), 'Gamma': {str(sig): ((np.array(b1s['M'])[np.where(grid == sig)[0][0]] - np.array(b1s['M'])[np.where(grid == -sig)[0][0]]) / (2 * sig)).tolist() for sig in [0.125, 0.25, 0.5, 1.0]}})
                print('RUN', m, cell, seed, flush=True)
            floor = roundoff * np.sum(abs(np.linalg.pinv(np.array([amps, amps ** 3]).T)[0]))
            mixedfloor = floor * np.sum(abs(np.linalg.pinv(np.array([ss, ss ** 3]).T)[0]))
            floors[cell] = {'response_float64_bound': roundoff, 'b1_bound': float(floor), 'mixed_m1_bound': float(mixedfloor), 'float32_derivative_max_discrepancy': float32error, 'relative_b1_fit_error_max': fiterror, 'relative_mixed_fit_error_max': mixedfiterror}
    table(ROOT / m / 'RAW.csv', rows)
    table(ROOT / m / 'COEFFICIENTS.csv', coeff)
    table(ROOT / m / 'SCALING.csv', scalings)
    write(ROOT / m / 'CONTROLS.json', controls)
    write(ROOT / m / 'CHAIN_RULE.json', chains)
    write(ROOT / m / 'REFERENCE_SCALES.json', references)
    if m == 'calibration':
        write(ROOT / 'preregistration/DIRECTION_MANIFEST.json', dirmanifest)
        write(ROOT / 'calibration/RESOLUTION_WINDOW_CHECKS.json', floors)
    else:
        assert dirmanifest == read(ROOT / 'preregistration/DIRECTION_MANIFEST.json')
    return (controls, floors)

def freeze():
    controls = read(ROOT / 'calibration/CONTROLS.json')
    floors = read(ROOT / 'calibration/RESOLUTION_WINDOW_CHECKS.json')
    assert len(controls) == 192
    assert all((c['P_ref'] > 0 and c['A_field_max'] < 1e-10 and (c['C_exact_at_zero_Lambda'] > 0) and (c['min_probability'] > 1e-12) and (c['max_logit_intervention'] <= 0.25) for c in controls))
    assert all((rms(c['M_exact_at_zero']) < 1e-10 and rms(c['M_autograd_at_zero']) < 1e-10 for c in controls))
    assert all((x['relative_b1_fit_error_max'] < 0.01 and x['relative_mixed_fit_error_max'] < 0.01 for x in floors.values()))
    assets = {str(p): sha(p) for folder in ['source', 'inputs', 'calibration'] for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for p in [ROOT / 'preregistration/INPUT_BINDINGS.json', ROOT / 'preregistration/DIRECTION_MANIFEST.json', *(ROOT / 'preregistration').glob('*identities.jsonl')]:
        assets[str(p)] = sha(p)
    assets.update(bind()['input_hashes'])
    fr = {'status': 'FROZEN_BEFORE_AUDIT', 'timestamp': now(), 'amplitudes': AMPS.tolist(), 'background_grid': GRID.tolist(), 'fitting_backgrounds': [-0.25, -0.125, 0, 0.125, 0.25], 'mixed_fit': 'unweighted OLS b1_M(s)=m1*s+m3*s^3, no intercept; symmetric subset includes s=0', 'fit': 'unweighted coordinate OLS B_odd=b1*delta+b3*delta^3; scaled design for stability', 'precision': 'checkpoint weights exactly promoted to CPU float64; rank-one interventions and softmax CPU float64; base representations/features MPS float32; float32 derivative diagnostic only', 'resolution': floors, 'directions_per_seed': 4, 'direction_rule': 'v282M-r2-new-directions SHA256, PCG64; normalize projected e and v; reject entire triple if any projection norm<=1e-12; same v in A/C/M', 'bootstrap': {'draws': 20000, 'unit': 'eight paired whole training seeds', 'rng': 'SeedSequence([20261011964,family_number,g])', 'CI': 'linear percentile 2.5 and 97.5, arithmetic mean magnitude; paired indices reused across all s/sectors/Xi in each cell'}, 'aggregation': 'RMS12 coefficient per direction then arithmetic mean of 4 direction magnitudes inside each seed; directions are nuisance replicates', 'H1': 'all A coordinate b1 magnitudes at all seeds/directions/backgrounds <= cell b1_bound', 'H2': 'all 8 seed mean-direction Lambda_C(0)>b1_bound and lower95CI arithmetic mean Lambda_C(0)>b1_bound', 'H3': 'all 8 seed Lambda_M(0)<=b1_bound OR every direction exact-chain and autograd coordinate derivative at 0 <= chain_atol (structural null); finite-amplitude residual still reported', 'H4': 'all 8 seed mean-direction Xi_M>mixed_m1_bound and lower95CI arithmetic mean Xi_M>mixed_m1_bound; chain-rule coordinate agreement required', 'chain_atol': 1e-10, 'chain_rtol': 1e-08, 'mandatory_analytic_checks_pass': True, 'regular_observable': 'C centered(u)/sqrt(P_ref); P_ref=Pq(s=1,delta=0) once per family/g/seed/bank; hard failure if nonpositive; diagnostic Pq never denominator', 'technical_controls': 'finite values, fixed hashes/directions, positive reference, represented function tolerance 1e-9 relative, no audit content before opening; max intervention logit <=0.25; no seed exclusion', 'classification': 'all H1-H4 six cells -> VALIDATED; reproducible violation of frozen theoretical null/chain prediction -> SCIENTIFIC_CONFLICT; strict subset brightness cells -> PARTIAL; unadjudicable technical control -> TECHNICAL_PROTOCOL_FAILURE', 'audit_content_opened': False, 'asset_hashes': assets}
    write(ROOT / 'preregistration/FREEZE.json', fr)
    (ROOT / 'preregistration/FREEZE.md').write_text('# v282M-r2 preregistration freeze\n\nFrozen before audit content opening.\n\n' + json.dumps({k: v for k, v in fr.items() if k != 'asset_hashes'}, indent=2) + '\n')
    write(ROOT / 'preregistration/FREEZE_LOCK.json', {'timestamp': now(), 'freeze_sha256': sha(ROOT / 'preregistration/FREEZE.json'), 'freeze_md_sha256': sha(ROOT / 'preregistration/FREEZE.md')})
    print('FROZEN', sha(ROOT / 'preregistration/FREEZE.json'), flush=True)
if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'freeze':
        freeze()
    elif sys.argv[1] == 'open-audit':
        fr = verify_freeze()
        assert not (ROOT / 'audit/AUDIT_OPENING_RECORD.json').exists()
        write(ROOT / 'audit/AUDIT_OPENING_RECORD.json', {'timestamp': now(), 'freeze_sha256': sha(ROOT / 'preregistration/FREEZE.json'), 'all_asset_hashes_verified': True, 'audit_content_previously_unopened': True})
    else:
        run(sys.argv[1])
