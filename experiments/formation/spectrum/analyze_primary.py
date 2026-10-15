# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""H2. Frozen windows, arm medians, paired bootstrap, componentwise decision."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, csv, math
import numpy as np
import pyarrow.parquet as pq
from audit_assets import ROOT, OUT, sha

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def csvout(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def ols(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    b = float(np.dot(xc, yc) / np.dot(xc, xc))
    a = float(y.mean() - b * x.mean())
    res = y - a - b * x
    ss = float(np.dot(yc, yc))
    r2 = 1 - float(np.dot(res, res)) / ss if ss > 0 else 0.0
    return (a, b, r2)

def run():
    assert json.loads((OUT / 'P2H_TRAINING_COMPLETION.json').read_text())['status'] == 'COMPLETE'
    pred = json.loads((OUT / 'P2H_PREDICTIONS.json').read_text())['predictions']
    rows = pq.read_table(OUT / 'P2H_TRAJECTORIES.parquet').to_pylist()
    assert len(rows) == 7800
    series = {}
    rate = np.zeros((8, 15))
    r2s = np.zeros((8, 15))
    fits = []
    for row in rows:
        series.setdefault((row['seed'], row['arm']), []).append(row)
    for (j, k), rs in sorted(series.items()):
        rs.sort(key=lambda r: r['update'])
        assert [r['update'] for r in rs] == list(range(65))
        n = np.arange(65, dtype=np.float64)
        a = np.array([r['A1'] for r in rs])
        assert np.all(a > 0)
        intercept, b, r2 = ols(n[8:33], np.log(a[8:33]))
        rate[j, k] = b
        r2s[j, k] = r2
        sens = {}
        for lo, hi in [(4, 16), (16, 48)]:
            _, sb, sr = ols(n[lo:hi + 1], np.log(a[lo:hi + 1]))
            sens.update({f'sensitivity_rate_{lo}_{hi}': sb, f'sensitivity_r2_{lo}_{hi}': sr})
        fits.append(dict(seed=j, arm=k, epsilon=pred[k]['epsilon'], r_eta=pred[k]['r_eta'], sigma_hat=b, intercept=intercept, r2=r2, resolved=r2 >= 0.9, primary_window='8:32', **sens))
    csvout('P2H_GROWTH_RATES.csv', fits)
    resolved = (r2s >= 0.9).sum(0) >= 6
    med = np.median(rate, axis=0)
    means = rate.mean(0)
    rng = np.random.default_rng(2026100801)
    indices = rng.integers(0, 8, (10000, 8))
    bootmed = np.median(rate[indices, :], axis=1)
    mu = pred[0]['mu']
    ratios = [0.25, 1.0, 4.0]
    threshold = []
    draws = np.zeros((10000, 3))
    threshold_vals = []
    threshold_res = []
    for ir, r in enumerate(ratios):
        kk = [k for k, p in enumerate(pred) if p['r_eta'] == r and p['epsilon'] in [-0.08, 0.04, 0.12]]
        x = np.array([pred[k]['lambda_'] for k in kk], dtype=np.float64)
        xc = x - x.mean()
        den = np.dot(xc, xc)
        rates = bootmed[:, kk]
        slope = ((rates - rates.mean(1, keepdims=True)) * xc).sum(1) / den
        intercept = rates.mean(1) - slope * x.mean()
        draws[:, ir] = -intercept / slope
        a, b, _ = ols(x, med[kk])
        lc = -a / b if b != 0 else None
        ok = bool(resolved[kk].all() and lc is not None and math.isfinite(lc))
        threshold_vals.append(lc)
        threshold_res.append(ok)
        interval = np.quantile(draws[:, ir], [0.025, 0.975])
        threshold.append(dict(r_eta=r, lambda_c_hat=lc, lambda_c_hess=mu, ratio_to_hess=lc / mu if lc is not None else None, resolved=ok, ci95_low=float(interval[0]), ci95_high=float(interval[1]), central_resolved_cells=int(resolved[kk].sum()), status='RESOLVED' if ok else 'NO_RESOLVED_FORMATION_TRANSITION'))
    csvout('P2H_STATIC_THRESHOLD.csv', threshold)
    diffs = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        ci = np.quantile(draws[:, i] - draws[:, j], [0.025, 0.975])
        diffs.append(dict(r_eta_a=ratios[i], r_eta_b=ratios[j], ci95_low=float(ci[0]), ci95_high=float(ci[1]), contains_zero=bool(ci[0] <= 0 <= ci[1])))
    np.savez(OUT / 'P2H_BOOTSTRAP_DRAWS.npz', seed_indices=indices, arm_median_rates=bootmed, threshold_draws=draws, bootstrap_seed=2026100801, bootstrap_reps=10000)
    static_checks = dict(all_thresholds_resolved=all(threshold_res), all_within_ten_percent=all((v is not None and abs(v / mu - 1) <= 0.1 for v in threshold_vals)), cross_ratio_range_le_ten_percent=(max(threshold_vals) - min(threshold_vals)) / mu <= 0.1 if all((v is not None for v in threshold_vals)) else False, all_pairwise_ci_contains_zero=all((x['contains_zero'] for x in diffs)))
    static = 'STATIC_BOUNDARY_INVARIANCE_SUPPORTED' if all(static_checks.values()) else 'STATIC_BOUNDARY_MOBILITY_DEPENDENCE_RESOLVED' if all(threshold_res) else 'NO_RESOLVED_FORMATION_TRANSITION'
    kk = [k for k, p in enumerate(pred) if p['epsilon'] > 0]
    rp = np.array([pred[k]['sigma_plus'] for k in kk])
    rh = med[kk]
    error = float(np.linalg.norm(rh - rp) / np.linalg.norm(rp))
    pc = means[kk] - means[kk].mean()
    qc = rp - rp.mean()
    rho = float(np.dot(pc, qc) / (np.linalg.norm(pc) * np.linalg.norm(qc))) if np.linalg.norm(pc) > 0 else None
    signs = bool(all((np.sign(med[k]) == np.sign(pred[k]['sigma_plus']) for k in kk if resolved[k])))
    nr = int(resolved[kk].sum())
    kinetic = 'KINETIC_RATE_UNRESOLVED' if nr < 7 else 'KINETIC_RATE_PREDICTION_SUPPORTED' if error <= 0.2 and rho is not None and (rho >= 0.95) and signs else 'KINETIC_RATE_MISMATCH'
    csvout('P2H_KINETIC_PREDICTION.csv', [dict(arm=k, epsilon=pred[k]['epsilon'], r_eta=pred[k]['r_eta'], predicted_sigma=pred[k]['sigma_plus'], median_measured_sigma=float(med[k]), mean_measured_sigma=float(means[k]), resolved_seeds=int((r2s[:, k] >= 0.9).sum()), resolved=bool(resolved[k]), sign_agreement=bool(np.sign(med[k]) == np.sign(pred[k]['sigma_plus'])), relative_l2_error_all9=error, pearson_across_cell_means=rho) for k in kk])
    onset = {}
    onset_rows = []
    for j in range(8):
        for k in kk:
            rs = series[j, k]
            a = np.array([r['A1'] for r in rs])
            times = [n for n in range(62) if bool((a[n:n + 4] / a[0] >= 4).all())]
            onset[j, k] = times[0] if times else None
    for ep in [0.12, 0.3]:
        base = next((k for k, p in enumerate(pred) if p['epsilon'] == ep and p['r_eta'] == 1.0))
        for j in range(8):
            for r in [0.25, 4.0]:
                k = next((k for k, p in enumerate(pred) if p['epsilon'] == ep and p['r_eta'] == r))
                t = onset[j, k]
                tb = onset[j, base]
                ok = t is not None and tb is not None
                d = abs(math.log(t / tb) - math.log(pred[base]['sigma_plus'] / pred[k]['sigma_plus'])) if ok else None
                onset_rows.append(dict(seed=j, epsilon=ep, r_eta=r, onset_time=t, reference_onset_time=tb, right_censored=t is None, reference_right_censored=tb is None, comparison_uncensored=ok, delta_onset=d, predicted_inverse_rate_ratio=pred[base]['sigma_plus'] / pred[k]['sigma_plus']))
    csvout('P2H_ONSET_SCALING.csv', onset_rows)
    ds = [r['delta_onset'] for r in onset_rows if r['comparison_uncensored']]
    frac = len(ds) / 32
    median_delta = float(np.median(ds)) if ds else None
    onset_status = 'ONSET_SCALING_UNRESOLVED' if frac < 0.75 else 'ONSET_SCALING_SUPPORTED' if median_delta <= math.log(1.25) else 'ONSET_SCALING_MISMATCH'
    write('P2H_ONSET_TIMES.json', dict(horizon=64, confirmation_updates=3, records=[dict(seed=j, arm=k, time=onset[j, k], right_censored=onset[j, k] is None) for j in range(8) for k in kk]))
    mode_rows = []
    for k, p in enumerate(pred):
        if p['epsilon'] in [0.12, 0.3]:
            f = float(np.median([series[j, k][32]['F_lead'] for j in range(8)]))
            c = float(np.median([series[j, k][32]['C_plus'] for j in range(8)]))
            mode_rows.append(dict(arm=k, epsilon=p['epsilon'], r_eta=p['r_eta'], median_F_lead_32=f, median_C_plus_32=c, capture_pass=f >= 0.75, alignment_pass=c >= 0.9))
    csvout('P2H_MODE_LOCKING.csv', mode_rows)
    fc = sum((r['capture_pass'] for r in mode_rows))
    cc = sum((r['alignment_pass'] for r in mode_rows))
    low = min((r['median_C_plus_32'] for r in mode_rows))
    mode = 'EARLY_MODE_LOCKING_SUPPORTED' if fc >= 5 and cc >= 5 and (low >= 0.75) else 'EARLY_MODE_LOCKING_NOT_SUPPORTED'
    supported = static == 'STATIC_BOUNDARY_INVARIANCE_SUPPORTED' and kinetic == 'KINETIC_RATE_PREDICTION_SUPPORTED' and (onset_status == 'ONSET_SCALING_SUPPORTED') and (mode == 'EARLY_MODE_LOCKING_SUPPORTED')
    files = ['P2H_TRAJECTORIES.parquet', 'P2H_CHECKPOINT_INDEX.json', 'P2H_GROWTH_RATES.csv', 'P2H_STATIC_THRESHOLD.csv', 'P2H_KINETIC_PREDICTION.csv', 'P2H_ONSET_SCALING.csv', 'P2H_MODE_LOCKING.csv', 'P2H_BOOTSTRAP_DRAWS.npz']
    decision = dict(status='SEALED', combined='STATIC_KINETIC_SEPARATION_SUPPORTED' if supported else 'STATIC_KINETIC_SEPARATION_NOT_SUPPORTED', components=dict(static_boundary=static, kinetic_rate=kinetic, onset_scaling=onset_status, early_mode_locking=mode), static_checks=static_checks, pairwise_threshold_difference_ci=diffs, kinetic_relative_l2_error=error, kinetic_pearson_cell_means=rho, kinetic_resolved_cells=nr, kinetic_all_resolved_signs_agree=signs, onset_uncensored_fraction=frac, onset_median_log_error=median_delta, mode_capture_passing_cells=fc, mode_alignment_passing_cells=cc, mode_min_median_alignment=low, primary_fit_window=[8, 32], primary_artifact_sha256={n: sha(OUT / n) for n in files}, preflight_seal_sha256=sha(OUT / 'P2H_PREFLIGHT_SEAL.json'), analysis_source_sha256=sha(__file__), objective_susceptibility_computed=False, Q3_recommendation=False)
    write('P2H_PRIMARY_DECISION.json', decision)
    (OUT / 'P2H_PRIMARY_DECISION.json').chmod(292)
    text = '# P2H primary formation result\n\nCombined verdict: **' + decision['combined'] + '**.\n\n'
    text += '| Component | Frozen verdict |\n|---|---|\n' + ''.join((f'| {k} | {v} |\n' for k, v in decision['components'].items()))
    text += f'\nKinetic relative L2 error: {error:.8g}; Pearson correlation across cell means: {rho}; resolved unstable cells: {nr}/9. Onset uncensored comparisons: {len(ds)}/32. Mode capture passes: {fc}/6; eigenvector alignment passes: {cc}/6.\n\nThe eight paired seeds were resampled jointly across all 15 cells in 10,000 bootstrap draws, seed 2026100801. Every seed rate uses OLS on log A1 at updates 8–32. Arm medians and means use all eight predeclared seeds; the resolved flag requires at least six R² >= 0.90. Sensitivity windows do not replace primary fits. Zero crossings use the three frozen central cells and are retained even when extrapolated.\n\nThis verdict is restricted to the controlled SmolLM2 layer-20 bridge. It provides no native OLMoE onset measurement and no Q3 training recommendation. An unresolved onset is not support.\n'
    (OUT / 'P2H_PRIMARY_REPORT.md').write_text(text)
    print(json.dumps(decision['components']), flush=True)
if __name__ == '__main__':
    run()
