# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""R4 primary frozen estimators and seal, followed by R5 saved-trajectory diagnostics."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import math, numpy as np
import pyarrow.parquet as pq
from common import *

def ols(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    g = float(xc @ yc / (xc @ xc))
    res = yc - g * xc
    ss = float(yc @ yc)
    return (g, 1 - float(res @ res) / ss if ss > 0 else 0.0, float(y.mean() - g * x.mean()), math.sqrt(float(res @ res) / len(y)))

def run():
    seal = verify_seal()
    completion = read(OUT / 'P2J_TRAINING_COMPLETION.json')
    assert completion['status'] == 'COMPLETE'
    raw = pq.read_table(OUT / 'P2J_TRAJECTORIES.parquet').to_pylist()
    by = {}
    for row in raw:
        by.setdefault((row['cell'], row['sign']), []).append(row)
    pred = {int(p['cell']): {k: int(v) if k in ['cell', 'N_cell'] else v if k == 'primary_role' else float(v) for k, v in p.items()} for p in csvread(OUT / 'P2J_FROZEN_CELL_PREDICTIONS.csv')}
    fits = []
    cells = []
    marginal = []
    for c, p in pred.items():
        arms = []
        for sign in [-1, 1]:
            rs = sorted(by[c, sign], key=lambda r: r['update'])
            assert [r['update'] for r in rs] == list(range(len(rs)))
            local = any((r['locality_violation'] for r in rs))
            horizon = len(rs) == p['N_cell'] + 1
            g, R2, b, rmse = ols(np.log([r['A1'] for r in rs]))
            isprimary = p['epsilon'] != 0
            rec = dict(cell=c, epsilon=p['epsilon'], r_eta=p['r_eta'], sign=sign, role=p['primary_role'], N_cell=p['N_cell'], n_observed=len(rs) - 1, window='ALL_UPDATES_0_TO_FROZEN_N', gamma_hat=g if isprimary else None, R2=R2 if isprimary else None, intercept=b if isprimary else None, log_amplitude_RMSE=rmse if isprimary else None, min_F1=min((r['F1'] for r in rs)), min_C_plus=min((r['C_plus'] for r in rs)), max_E_sign=max((r['sign_pair_antisymmetry'] for r in rs)), min_full_amplitude_ratio=min((r['full_amplitude_ratio'] for r in rs)), max_full_amplitude_ratio=max((r['full_amplitude_ratio'] for r in rs)), locality_violation=local, full_frozen_horizon_observed=horizon, arm_fit_eligible=isprimary and horizon and (not local) and (R2 >= 0.995))
            fits.append(rec)
            arms.append(rec)
            if not isprimary:
                marginal.append(dict(cell=c, r_eta=p['r_eta'], sign=sign, N_cell=p['N_cell'], completed_update=len(rs) - 1, relative_A1_drift=rs[-1]['A1'] / rs[0]['A1'] - 1, max_abs_relative_A1_drift=max((abs(r['A1'] / rs[0]['A1'] - 1) for r in rs)), relative_A_full_drift=rs[-1]['A_full'] / rs[0]['A_full'] - 1, min_F1=rec['min_F1'], min_C_plus=rec['min_C_plus'], max_E_sign=rec['max_E_sign'], locality_violation=local, primary_aggregate_eligible=False))
        primary = p['epsilon'] != 0
        mean = (arms[0]['gamma_hat'] + arms[1]['gamma_hat']) / 2 if primary else None
        delta = abs(arms[0]['gamma_hat'] - arms[1]['gamma_hat']) / max(abs(mean), 1e-08) if primary else None
        valid = primary and all((r['arm_fit_eligible'] for r in arms)) and (delta <= 0.1)
        cells.append(dict(cell=c, epsilon=p['epsilon'], r_eta=p['r_eta'], gamma_plus=p['gamma_plus'], gamma_mean=mean, sign_rate_asymmetry=delta, rate_valid=valid, status='RATE_VALID' if valid else 'MARGINAL_CONTROL' if not primary else 'RATE_UNRESOLVED', min_F1=min((r['min_F1'] for r in arms)), min_C_plus=min((r['min_C_plus'] for r in arms)), max_E_sign=max((r['max_E_sign'] for r in arms)), locality_violation=any((r['locality_violation'] for r in arms))))
    csvout('P2J_RATE_FITS.csv', fits)
    csvout('P2J1_MARGINAL_CONTROLS.csv', marginal)
    static = []
    for ratio in [0.25, 1.0, 4.0]:
        left = next((c for c in cells if c['epsilon'] == -0.08 and c['r_eta'] == ratio))
        right = next((c for c in cells if c['epsilon'] == 0.08 and c['r_eta'] == ratio))
        lc = None
        if not left['rate_valid'] or not right['rate_valid']:
            status = 'P2J_STATIC_BOUNDARY_UNRESOLVED'
        elif left['gamma_mean'] * right['gamma_mean'] >= 0:
            status = 'P2J_STATIC_BOUNDARY_NOT_BRACKETED'
        else:
            lc = -0.08 - left['gamma_mean'] * 0.16 / (right['gamma_mean'] - left['gamma_mean'])
            assert -0.08 <= lc <= 0.08
            status = 'P2J_STATIC_BOUNDARY_BRACKETED'
        static.append(dict(r_eta=ratio, epsilon_left=-0.08, epsilon_right=0.08, left_rate=left['gamma_mean'], right_rate=right['gamma_mean'], left_rate_valid=left['rate_valid'], right_rate_valid=right['rate_valid'], epsilon_c=lc, status=status, within_abs_0_04=lc is not None and abs(lc) <= 0.04, extrapolation=False))
    csvout('P2J_STATIC_BOUNDARY.csv', static)
    if all((r['epsilon_c'] is not None for r in static)):
        spread = max((r['epsilon_c'] for r in static)) - min((r['epsilon_c'] for r in static))
        staticstatus = 'P2J_STATIC_INVARIANCE_SUPPORTED' if all((r['within_abs_0_04'] for r in static)) and spread <= 0.04 else 'P2J_STATIC_INVARIANCE_NOT_SUPPORTED'
    else:
        spread = None
        staticstatus = 'P2J_STATIC_INVARIANCE_UNRESOLVED' if any((r['status'].endswith('UNRESOLVED') for r in static)) else 'P2J_STATIC_INVARIANCE_NOT_SUPPORTED'
    kinetic = []
    for c in cells:
        if c['epsilon'] == 0:
            continue
        kinetic.append(dict(cell=c['cell'], epsilon=c['epsilon'], r_eta=c['r_eta'], gamma_predicted=c['gamma_plus'], gamma_observed=c['gamma_mean'], rate_valid=c['rate_valid'], relative_error=abs(c['gamma_mean'] - c['gamma_plus']) / abs(c['gamma_plus']), sign_matches=bool(np.sign(c['gamma_mean']) == np.sign(c['gamma_plus']))))
    csvout('P2J_DISCRETE_KINETICS.csv', kinetic)
    valid = [c for c in kinetic if c['rate_valid']]
    E = None
    corr = None
    if valid:
        obs = np.array([c['gamma_observed'] for c in valid])
        th = np.array([c['gamma_predicted'] for c in valid])
        E = float(np.linalg.norm(obs - th) / np.linalg.norm(th))
        corr = float(np.corrcoef(obs, th)[0, 1]) if len(valid) > 1 else None
    signok = all((c['sign_matches'] for c in valid))
    kineticstatus = 'P2J_DISCRETE_KINETICS_UNRESOLVED' if len(valid) < 10 else 'P2J_DISCRETE_KINETICS_SUPPORTED' if E <= 0.1 and corr >= 0.995 and signok else 'P2J_DISCRETE_KINETICS_NOT_SUPPORTED'
    mobility = []
    for eps in [0.08, 0.24]:
        seq = [next((c for c in cells if c['epsilon'] == eps and c['r_eta'] == ratio)) for ratio in [0.25, 1.0, 4.0]]
        allvalid = all((c['rate_valid'] for c in seq))
        ordering = list(np.argsort([c['gamma_mean'] for c in seq])) == list(np.argsort([c['gamma_plus'] for c in seq]))
        for k in [0, 2]:
            obsratio = seq[k]['gamma_mean'] / seq[1]['gamma_mean'] if seq[1]['gamma_mean'] else None
            predratio = seq[k]['gamma_plus'] / seq[1]['gamma_plus']
            err = abs(obsratio / predratio - 1) if obsratio is not None else None
            mobility.append(dict(epsilon=eps, r_eta=seq[k]['r_eta'], reference_r_eta=1.0, observed_rate_ratio=obsratio, predicted_rate_ratio=predratio, ratio_relative_error=err, ordering_matches=ordering, all_three_cells_rate_valid=allvalid, passed=allvalid and ordering and (err is not None) and (err <= 0.1)))
    csvout('P2J_MOBILITY_KINETICS.csv', mobility)
    mobilitystatus = 'P2J_MOBILITY_KINETICS_UNRESOLVED' if not all((c['all_three_cells_rate_valid'] for c in mobility)) else 'P2J_MOBILITY_KINETICS_SUPPORTED' if all((c['passed'] for c in mobility)) else 'P2J_MOBILITY_KINETICS_NOT_SUPPORTED'
    modes = [dict(cell=c['cell'], epsilon=c['epsilon'], r_eta=c['r_eta'], rate_valid=c['rate_valid'], min_F1=c['min_F1'], min_C_plus=c['min_C_plus'], locality_violation=c['locality_violation'], passed=c['rate_valid'] and c['min_F1'] >= 0.95 and (c['min_C_plus'] >= 0.95)) for c in cells if c['epsilon'] != 0]
    csvout('P2J_MODE_PRESERVATION.csv', modes)
    tested = [c for c in modes if c['rate_valid']]
    modestatus = 'P2J_MODE_PRESERVATION_UNRESOLVED' if not tested else 'P2J_MODE_PRESERVATION_SUPPORTED' if all((c['passed'] for c in tested)) else 'P2J_MODE_PRESERVATION_NOT_SUPPORTED'
    components = dict(static=staticstatus, kinetic=kineticstatus, mobility=mobilitystatus, mode=modestatus)
    if all((x.endswith('_SUPPORTED') and (not x.endswith('_NOT_SUPPORTED')) for x in components.values())):
        decision = 'DETERMINISTIC_FIXED_BANK_R1_BRIDGE_SUPPORTED'
    elif any((x.endswith('_NOT_SUPPORTED') for x in components.values())):
        decision = 'DETERMINISTIC_FIXED_BANK_R1_BRIDGE_NOT_SUPPORTED'
    else:
        decision = 'DETERMINISTIC_FIXED_BANK_R1_BRIDGE_UNRESOLVED'
    write('P2J_PRIMARY_DECISION.json', dict(decision=decision, components=components, valid_nonzero_cells=len(valid), required_valid_nonzero_cells=10, total_nonzero_cells=12, marginal_controls_excluded=True, static_bracket=[-0.08, 0.08], static_epsilon_c_range=spread, kinetic_relative_l2_error=E, kinetic_pearson=corr, kinetic_all_valid_signs_match=signok, mode_valid_cells_tested=len(tested), locality_violating_cells=sum((c['locality_violation'] for c in cells)), primary_sealed_utc=utc(), preflight_seal_sha256=sha(OUT / 'P2J_PREFLIGHT_SEAL.json'), primary_artifact_sha256={n: sha(OUT / n) for n in ['P2J_TRAJECTORIES.parquet', 'P2J_CHECKPOINT_INDEX.json', 'P2J_TRAINING_COMPLETION.json', 'P2J_RATE_FITS.csv', 'P2J_STATIC_BOUNDARY.csv', 'P2J_DISCRETE_KINETICS.csv', 'P2J_MOBILITY_KINETICS.csv', 'P2J_MODE_PRESERVATION.csv', 'P2J1_MARGINAL_CONTROLS.csv']}, counters=COUNTERS, interpretation_ceiling='SmolLM2-135M zero-based layer 20 four-expert dense-softmax private construction; deterministic same-bank local law only; no stochastic or native-MoE claim; P2H historical negative unchanged'))
    dsha = sha(OUT / 'P2J_PRIMARY_DECISION.json')
    celltable = '\n'.join((f"| {c['epsilon']:+.2f} | {c['r_eta']:g} | {c['gamma_plus']:.9g} | {c['gamma_mean']:.9g} | {c['status']} | {c['min_F1']:.9g} | {c['min_C_plus']:.9g} |" for c in cells if c['epsilon'] != 0))
    statictable = '\n'.join((f"| {r['r_eta']:g} | {r['left_rate']:.9g} | {r['right_rate']:.9g} | {(r['epsilon_c'] if r['epsilon_c'] is not None else 'unresolved')} | {r['status']} |" for r in static))
    report = f'# P2J.1 Primary Report\n\n**Decision: `{decision}`**\n\nAll 30 original deterministic sign arms are represented. Nonzero RATE_VALID cells: {len(valid)}/12. Marginal ε=0 arms are controls and are excluded from the required rate counts, kinetic/mobility/mode aggregates, and static interpolation.\n\nComponents:\n\n- Static: `{staticstatus}`\n- Discrete kinetics: `{kineticstatus}`\n- Mobility kinetics: `{mobilitystatus}`\n- Mode preservation: `{modestatus}`\n\nRelative kinetic L2 error: {E}. Pearson: {corr}. All valid kinetic signs match: {signok}.\nStatic boundary range across mobility: {spread}. Frozen thresholds: error ≤0.10, Pearson ≥0.995, at least 10 valid nonzero cells; |εc|≤0.04 and cross-mobility range≤0.04. No thresholds, fit windows, horizons, banks or operators were changed after outcomes.\n\n## Nonzero cells\n\nRates are OLS fits over every integer update 0..N_cell, with R²≥0.995 and sign rate asymmetry≤0.10 required. Both signs must pass locality throughout the horizon.\n\n| ε | rη | frozen γ+ | observed sign mean | status | min F1 | min C+ |\n|---|---|---|---|---|---|---|\n{celltable}\n\n## Prespecified static brackets\n\n| rη | rate at −0.08 | rate at +0.08 | εc | status |\n|---|---|---|---|---|\n{statictable}\n\nOnly the frozen −0.08,+0.08 pair is used. No extrapolation. Marginal drift is reported separately in `P2J1_MARGINAL_CONTROLS.csv`.\n\nThe primary decision was sealed before secondary analysis; SHA-256 `{dsha}`. The conclusion applies only to this model/layer/private soft-MoE deterministic fixed-bank local experiment. It does not validate stochastic minibatch SGD or native MoE. P2H remains `STATIC_KINETIC_SEPARATION_NOT_SUPPORTED`.\n'
    textout('P2J_PRIMARY_REPORT.md', report)
    diagnostics = []
    for c, p in pred.items():
        minus = sorted(by[c, -1], key=lambda r: r['update'])
        plus = sorted(by[c, 1], key=lambda r: r['update'])
        xp = np.array([[r['m1'], r['u1']] for r in plus])
        xm = np.array([[r['m1'], r['u1']] for r in minus])
        even = (xp + xm) / 2
        odd = (xp - xm) / 2
        n = np.arange(len(plus))
        T = np.array([[p['T00'], p['T01']], [p['T10'], p['T11']]])
        theory = np.empty_like(odd)
        theory[0] = odd[0]
        for i in range(1, len(n)):
            theory[i] = T @ theory[i - 1]
        pairerr = float(np.linalg.norm(odd - theory) / max(np.linalg.norm(theory), 1e-30))
        evenmax = float(np.max(np.linalg.norm(even, axis=1)) / p['a0'])
        for sign, rs in [(-1, minus), (1, plus)]:
            logA = np.log([r['A1'] for r in rs])
            g, R2, b, rmse = ols(logA)
            quad = np.polyfit(n, logA, 2)
            ortho = np.sqrt(np.maximum(np.array([r['A_full'] ** 2 - r['A1'] ** 2 for r in rs]), 0))
            diagnostics.append(dict(cell=c, epsilon=p['epsilon'], abs_epsilon=abs(p['epsilon']), r_eta=p['r_eta'], sign=sign, role=p['primary_role'], observed_updates=len(rs) - 1, logA_linear_R2=R2, logA_curvature_quadratic_coefficient=float(quad[0]), logA_linear_RMSE=rmse, max_sign_even_pair_norm_over_a0=evenmax, sign_odd_pair_theory_relative_L2=pairerr, initial_orthogonal_parameter_norm=float(ortho[0]), final_orthogonal_parameter_norm=float(ortho[-1]), max_orthogonal_parameter_norm_over_a0=float(np.max(ortho) / p['a0']), initial_F0_NLL=rs[0]['F0_NLL'], final_F0_NLL=rs[-1]['F0_NLL'], F0_NLL_change=rs[-1]['F0_NLL'] - rs[0]['F0_NLL'], gamma_error=g - p['gamma_plus'], rate_error_over_abs_predicted_rate=abs(g - p['gamma_plus']) / abs(p['gamma_plus']) if p['epsilon'] else None, primary_decision_sha256=dsha))
    csvout('P2J_NONLINEARITY_DIAGNOSTICS.csv', diagnostics)
    index = read(OUT / 'P2J_CHECKPOINT_INDEX.json')
    lookup = {(r['cell'], r['sign'], r['update']): r for r in index['checkpoints']}
    vectorrows = []
    for c, sign, n in sorted(lookup):
        if sign != -1:
            continue
        mrec = lookup[c, -1, n]
        prec = lookup[c, 1, n]
        assert sha(mrec['path']) == mrec['sha256'] and sha(prec['path']) == prec['sha256']
        with np.load(mrec['path']) as z:
            mm = z['M'].astype(float)
            mu = z['U'].astype(float)
        with np.load(prec['path']) as z:
            pm = z['M'].astype(float)
            pu = z['U'].astype(float)
        evennorm = math.sqrt(float(np.sum(((pm + mm) / 2) ** 2) + np.sum(((pu + mu) / 2) ** 2)))
        oddnorm = math.sqrt(float(np.sum(((pm - mm) / 2) ** 2) + np.sum(((pu - mu) / 2) ** 2)))
        vectorrows.append(dict(cell=c, epsilon=pred[c]['epsilon'], r_eta=pred[c]['r_eta'], update=n, full_sign_even_norm=evennorm, full_sign_odd_norm=oddnorm, full_sign_even_over_a0=evennorm / pred[c]['a0'], primary_decision_sha256=dsha))
    csvout('P2J1_SIGN_FULL_VECTOR_DIAGNOSTICS.csv', vectorrows)
    textout('P2J_SECONDARY_REPORT.md', f"# P2J.1 Secondary Diagnostics\n\nPrimary decision was sealed first: SHA-256 `{dsha}`. This stage reads only saved trajectories and checkpoints; model forward/backward/HVP/update/download counts in this stage are all zero.\n\n- Per-update sign-even/odd leading coordinates are compared to the frozen discrete map; full-vector even/odd norms are reported at saved checkpoints (every 4 updates plus endpoints).\n- `P2J_NONLINEARITY_DIAGNOSTICS.csv` reports log-amplitude quadratic curvature, linear-fit residuals, orthogonal parameter growth, F0 NLL changes, and rate errors indexed by ε and mobility.\n- Maximum sign-even leading norm/a0: {max((r['max_sign_even_pair_norm_over_a0'] for r in diagnostics)):.9g}.\n- Maximum sign-odd pair prediction relative L2 error: {max((r['sign_odd_pair_theory_relative_L2'] for r in diagnostics)):.9g}.\n- Maximum full-vector sign-even norm/a0 over saved checkpoints: {max((r['full_sign_even_over_a0'] for r in vectorrows)):.9g}.\n\nMarginal ε=0 rows remain numerical/locality controls. No objective susceptibility, stochastic extension, new probe, rescue fit, or amended science criterion is computed.\n")
    write('P2J1_SECONDARY_SEAL.json', dict(status='COMPLETE', primary_decision_sha256=dsha, secondary_created_utc=utc(), artifacts={n: sha(OUT / n) for n in ['P2J_NONLINEARITY_DIAGNOSTICS.csv', 'P2J_SECONDARY_REPORT.md', 'P2J1_SIGN_FULL_VECTOR_DIAGNOSTICS.csv']}, new_model_evaluations=0))
    print(decision, components, 'kinetic_error', E, 'Pearson', corr, flush=True)
if __name__ == '__main__':
    run()
