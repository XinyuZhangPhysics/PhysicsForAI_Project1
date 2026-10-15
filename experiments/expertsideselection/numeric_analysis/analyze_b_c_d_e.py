# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
from pathlib import Path
ROOT = Path(_p1_path('path_fd01557d84f7'))
OUT = ROOT / 'outputs/v285M_archive_completion'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/mplconfig')
import csv, hashlib, json, shutil
import numpy as np
import pandas as pd
import sys
sys.path.append(_p1_path('path_d4fe3b336f94'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(_p1_path('path_65e87eb57756'))
B = BASE / '2026-10-11/an-3/Project1_BackgroundModeConversion_v282M_r2'
C = BASE / '2026-10-12/jie-2/outputs/v284M_r1'
D = BASE / '2026-10-08/physicsforai-project-1-p2d-agents-md/outputs/P2D_CERTIFIED_RESUME_20261008_103610'
V = Path(_p1_path('path_9daa8b272cd4'))
SOURCES = {}
CHECKS = []
REG = []
RESULT = {}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def source(p, branch, role):
    p = Path(p)
    h = sha(p)
    SOURCES[str(p)] = dict(source_path=str(p), sha256=h, size_bytes=p.stat().st_size, branch=branch, role=role)
    return p

def readcsv(p, branch='authority', role='sealed source'):
    return pd.read_csv(source(p, branch, role), float_precision='round_trip')

def readjson(p, branch, role):
    return json.loads(source(p, branch, role).read_text())

def npz(p, branch, role):
    with np.load(source(p, branch, role), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def check(label, actual, expected, rtol=1e-10, atol=1e-12):
    ok = bool(np.isclose(actual, expected, rtol=rtol, atol=atol))
    CHECKS.append(dict(branch=label.split(':')[0], check=label, actual=float(actual), expected=float(expected), rtol=rtol, atol=atol, pass_check=ok))
    assert ok, (label, actual, expected)

def ci(x):
    return [float(v) for v in np.quantile(x, [0.025, 0.975])]

def table(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format='%.17g')

def savefig(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def note(name, text):
    (OUT / name).write_text(text + '\n')

def reg(branch, stat, value, unit, scope, status='new_derived'):
    REG.append(dict(branch=branch, statistic=stat, value=value, unit=unit, scope=scope, status=status))
for p in V.rglob('*'):
    if p.is_file():
        source(p, 'authority', 'v284M read-only integrity baseline')
for p in [Path(_p1_path('path_5eb61c4666c4')), Path(_p1_path('path_c7aaf2b90a15'))]:
    source(p, 'instructions', 'user-invoked handoff or supporting plan')
raw = readcsv(B / 'audit/RAW.csv', 'B', 'full signed s/delta response')
coef = readcsv(B / 'audit/COEFFICIENTS.csv', 'B', 'archived signed fitted coefficients')
chains = readjson(B / 'audit/CHAIN_RULE.json', 'B', 'analytic signed coordinate coefficients')
controls = readjson(B / 'audit/CONTROLS.json', 'B', 'mixed signed OLS coefficients')
fr = readjson(B / 'preregistration/FREEZE.json', 'B', 'preregistered estimator and bootstrap')
sealed = readjson(B / 'outputs/BACKGROUND_MODE_CONVERSION_R2_RESULTS.json', 'B', 'sealed six cells and exact paired draw indices')
source(B / 'source/experiment.py', 'B', 'original fit/chain code')
source(B / 'source/finalize.py', 'B', 'original aggregation code')
assert len(raw) == 51840 and len(coef) == 5184 and (len(chains) == 5184)
amps = np.asarray(fr['amplitudes'])
grid = np.asarray(fr['background_grid'])
design = np.array([amps / amps[-1], (amps / amps[-1]) ** 3]).T
keys = ['family', 'g', 'seed', 'direction', 'sector', 's']
coidx = coef.set_index(keys)
chainidx = {tuple((r[k] for k in keys)): r for r in chains}
rows = []
fit_max = 0.0
fit_relative_max = 0.0
for key, x in raw.groupby(keys, sort=True):
    assert len(x) == 10 and x.P_ref.nunique() == 1
    signed = x.set_index('delta')
    vals = np.asarray([(signed.loc[d, [f'B_{i:02}' for i in range(12)]].to_numpy(dtype=float) - signed.loc[-d, [f'B_{i:02}' for i in range(12)]].to_numpy(dtype=float)) / 2 for d in amps])
    fit = np.linalg.lstsq(design, vals, rcond=None)[0]
    b1 = fit[0] / amps[-1]
    b3 = fit[1] / amps[-1] ** 3
    archived = coidx.loc[key, [f'b1_{i:02}' for i in range(12)]].to_numpy(dtype=float)
    fit_max = max(fit_max, float(np.max(abs(b1 - archived))))
    assert np.allclose(b1, archived, rtol=1e-12, atol=1e-18)
    pred = np.asarray(chainidx[key]['analytic'])
    assert np.allclose(b1, chainidx[key]['b1_fit'], rtol=1e-12, atol=1e-18)
    for i in range(12):
        rows.append({**dict(zip(keys, key)), 'coordinate': i, 'P_ref': float(x.P_ref.iloc[0]), 'predicted_analytic_coefficient': pred[i], 'measured_signed_coefficient': b1[i], 'residual': b1[i] - pred[i], 'cubic_signed_coefficient': b3[i]})
curve = pd.DataFrame(rows)
od = curve.set_index(keys + ['coordinate'])
curve['odd_in_s_measured'] = [(r.measured_signed_coefficient - float(od.loc[(r.family, r.g, r.seed, r.direction, r.sector, -r.s, r.coordinate), 'measured_signed_coefficient'])) / 2 for r in curve.itertuples()]
curve['odd_in_s_predicted'] = [(r.predicted_analytic_coefficient - float(od.loc[(r.family, r.g, r.seed, r.direction, r.sector, -r.s, r.coordinate), 'predicted_analytic_coefficient'])) / 2 for r in curve.itertuples()]
curve.to_csv(OUT / 'MODE_CONVERSION_SIGNED_CURVE.csv', index=False, float_format='%.17g')
shutil.copyfile(B / 'audit/RAW.csv', OUT / 'MODE_CONVERSION_SIGNED_RESPONSES.csv')
summary = []
for (family, g), cell in curve.groupby(['family', 'g'], sort=True):
    indices = np.random.default_rng(np.random.SeedSequence([20261011964, 1 if family == 'S1' else 2, g])).integers(0, 8, size=(20000, 8))
    assert hashlib.sha256(indices.tobytes()).hexdigest() == sealed['bootstrap_indices'][f'{family}_g{g}']['indices_sha256']
    np.save(OUT / f'MODE_CONVERSION_BOOTSTRAP_{family}_g{g}.npy', indices)
    seedids = sorted(cell.seed.unique())
    seedcoef = []
    mixcoef = []
    for seed in seedids:
        z = cell[(cell.seed == seed) & (cell.sector == 'C') & (cell.s == 0)]
        seedcoef.append(np.mean([np.sqrt(np.mean(y.measured_signed_coefficient.to_numpy() ** 2)) for _, y in z.groupby('direction')]))
        m = cell[(cell.seed == seed) & (cell.sector == 'M')]
        magnitudes = []
        for direction, y in m.groupby('direction'):
            y = y[y.s.abs() <= 0.25]
            ss = sorted(y.s.unique())
            values = np.array([y[y.s == s].sort_values('coordinate').measured_signed_coefficient.to_numpy() for s in ss])
            ss = np.array(ss)
            mfit = np.linalg.lstsq(np.array([ss, ss ** 3]).T, values, rcond=None)[0]
            magnitudes.append(np.sqrt(np.mean(mfit[0] ** 2)))
        mixcoef.append(np.mean(magnitudes))
    authority = readcsv(V / 'figure_data/MODE_CONVERSION_SIX_CELLS.csv')
    expected = authority[authority.family.str.startswith(family) & (authority.g == g)].iloc[0]
    for field, values, lo, hi in [('lambda_C_s0', seedcoef, 'lambda_C_ci_low', 'lambda_C_ci_high'), ('xi_M', mixcoef, 'xi_M_ci_low', 'xi_M_ci_high')]:
        values = np.array(values)
        boot = values[indices].mean(1)
        interval = ci(boot)
        check(f'B:{family}/g{g}:{field}', values.mean(), expected[field], rtol=5e-08, atol=1e-12)
        check(f'B:{family}/g{g}:{lo}', interval[0], expected[lo], rtol=5e-08, atol=1e-12)
        check(f'B:{family}/g{g}:{hi}', interval[1], expected[hi], rtol=5e-08, atol=1e-12)
    for (sector, s, coordinate), z in cell.groupby(['sector', 's', 'coordinate'], sort=True):
        z = z.groupby('seed', sort=True)[['predicted_analytic_coefficient', 'measured_signed_coefficient', 'residual', 'odd_in_s_measured', 'odd_in_s_predicted']].mean().reindex(seedids)
        row = dict(family=family, g=g, sector=sector, s=s, coordinate=coordinate, seeds=8, directions_per_seed=4, bootstrap_draws=20000)
        for field in z:
            values = z[field].to_numpy()
            low, high = ci(values[indices].mean(1))
            row[field] = float(values.mean())
            row[field + '_ci_low'] = low
            row[field + '_ci_high'] = high
        summary.append(row)
table('MODE_CONVERSION_SIGNED_SUMMARY.csv', summary)
summary = pd.DataFrame(summary)
fig, axes = plt.subplots(2, 3, figsize=(10, 5.7), sharex=True, layout='constrained')
for ax, (family, g) in zip(axes.flat, [(f, g) for f in ['S1', 'S2'] for g in [2, 4, 8]]):
    x = summary[(summary.family == family) & (summary.g == g) & (summary.sector == 'M') & (summary.coordinate == 0)].sort_values('s')
    ax.axhline(0, color='.7', lw=0.6)
    ax.plot(x.s, x.predicted_analytic_coefficient, color='#bd3b36', label='Analytic')
    ax.plot(x.s, x.measured_signed_coefficient, 'o', ms=3, color='#1d6089', label='Odd-amplitude fit')
    ax.fill_between(x.s, x.measured_signed_coefficient_ci_low, x.measured_signed_coefficient_ci_high, color='#1d6089', alpha=0.18)
    ax.set_title(f'{family}, g={g}')
    ax.set_xlabel('Background s')
    ax.set_ylabel('Signed M coefficient, coordinate 0')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[0, 0].legend(fontsize=8)
savefig(fig, 'FIG_MODE_CONVERSION_SIGNED')
for sector in ['C', 'M']:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    for ax, (family, g) in zip(axes.flat, [(f, g) for f in ['S1', 'S2'] for g in [2, 4, 8]]):
        for coord in range(12):
            x = summary[(summary.family == family) & (summary.g == g) & (summary.sector == sector) & (summary.coordinate == coord)].sort_values('s')
            color = plt.cm.tab20(coord)
            ax.plot(x.s, x.predicted_analytic_coefficient, color=color, lw=0.9)
            ax.scatter(x.s, x.measured_signed_coefficient, color=color, s=5)
        ax.axhline(0, color='.7', lw=0.6)
        ax.set_title(f'{family}, g={g}, sector {sector}')
        ax.set_xlabel('s')
        ax.set_ylabel('Signed coefficient')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    savefig(fig, f'FIG_MODE_CONVERSION_SIGNED_{sector}_ALL_COORDINATES')
RESULT['B'] = dict(status='COMPLETE', raw_signed_records=51840, coordinate_coefficients=len(curve), refit_max_absolute_difference=fit_max, sealed_scalar_and_CI_checks=36)
note('MODE_CONVERSION_METHOD_NOTE.md', f'# Signed mode-conversion archive recovery\n\nThe exact frozen estimator is unweighted coordinate OLS of B_odd(delta) = [B(delta)-B(-delta)]/2 onto delta and delta^3, with the original scaled design. The archive contains five positive magnitudes and both signs (ten signed records per background), nine s values, eight paired seeds/cell, four nuisance directions/seed, and twelve ordered source-target coordinates. P_ref is fixed within each family/g/seed/audit-bank and never replaced by Pq. No model was evaluated in this branch.\n\nThe analytic signed vectors come from archived CHAIN_RULE.json and the original code. Every fitted coefficient was recomputed from RAW.csv; maximum absolute discrepancy with the archived fit: {fit_max:.17g}. The six C-at-zero and mixed-M RMS values and their confidence limits are source checks only, kept separate from signed results. For the mixed check the original b1_M(s) = m1*s + m3*s^3 fit uses s in [-.25,-.125,0,.125,.25].\n\nSigned summaries average four nuisance directions inside each seed and then eight seeds, using the exact 20,000 archived whole-seed paired indices for every sector/background/coordinate. Percentile intervals are 2.5/97.5. Odd-in-s coefficients use paired plus/minus backgrounds before resampling. Signed direction vectors remain fully available; averaging random nuisance directions can cancel, so a mean near zero does not imply each direction is dark. The primary figure uses coordinate 0 selected without inspecting outcomes, and supplementary figures show all twelve coordinates. No sign alignment, absolute/RMS replacement, coordinate selection by effect, or checkpoint interpolation is used.\n\nScope: fixed-reference restricted response under the archived coordinate convention; these results do not establish a full B_ts or utility claim.')
print('B COMPLETE', flush=True)
coord = readcsv(C / 'EXPERT_SIDE_SELECTION_COORDINATES_v284M.csv', 'C', 'full measured/predicted coordinate derivatives')
directions = readcsv(C / 'EXPERT_SIDE_SELECTION_DIRECTION_SUMMARY_v284M.csv', 'C', 'A field, dark RMS and field references')
source(C / 'source/core.py', 'C', 'original field projection definitions')
source(C / 'source/run.py', 'C', 'original derivative computation')
res = coord.dot_B_actual - coord.dot_B_prediction_A
assert np.allclose(res, coord.residual, rtol=0, atol=0)
check('C:max H1 coordinate residual', np.max(abs(res)), 3.469446951953614e-18, rtol=0, atol=1e-30)
field_records = []
for r in directions.itertuples():
    p = Path(r.field_path)
    if not p.exists():
        p = C / 'fields' / p.name
    assert sha(p) == r.field_sha256
    source(p, 'C', 'archived field arrays; field_sha256 matched')
    with np.load(p, allow_pickle=False) as z:
        actual_field_max = float(np.max(abs(z['dot_kappa_A'])))
    assert actual_field_max == r.field_A_max_abs
    field_records.append(dict(field_path=str(p), sha256=r.field_sha256, family=r.family, g=r.g, seed=r.seed, direction=r.direction, direction_class=directions.loc[r.Index, 'class'], s=r.s, field_A_max_abs=r.field_A_max_abs))
null = directions[directions['class'] == 'child']
dark = directions[(directions['class'] == 'child') | (directions.s == 0)]
check('C:max A-null field component', null.field_A_max_abs.max(), 4.336808689942018e-19, rtol=0, atol=1e-30)
assert (dark.actual_RMS12 < 1e-10).all()
coord.to_csv(OUT / 'R63_COORDINATE_CLOSURE.csv', index=False, float_format='%.17g')
table('R63_FIELD_SOURCE_CHECKS.csv', field_records)
cs = []
for key, x in coord.groupby(['family', 'g', 'class', 's'], sort=True):
    cs.append({**dict(zip(['family', 'g', 'direction_class', 's'], key)), 'coordinates': len(x), 'max_abs_residual': float(x.residual.abs().max()), 'rms_residual': float(np.sqrt(np.mean(x.residual ** 2))), 'measured_RMS': float(np.sqrt(np.mean(x.dot_B_actual ** 2))), 'predicted_RMS': float(np.sqrt(np.mean(x.dot_B_prediction_A ** 2)))})
table('R63_COORDINATE_CLOSURE_SUMMARY.csv', cs)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout='constrained')
for family, color in [('S1', '#1d6089'), ('S2', '#bd3b36')]:
    x = coord[coord.family == family]
    axes[0].scatter(x.dot_B_prediction_A, x.dot_B_actual, s=4, alpha=0.5, label=family, color=color)
lim = max(coord.dot_B_actual.abs().max(), coord.dot_B_prediction_A.abs().max()) * 1.05
axes[0].plot([-lim, lim], [-lim, lim], 'k--', lw=0.7)
axes[0].set(xlabel='A-projected prediction', ylabel='Measured derivative')
axes[0].legend()
axes[1].hist(coord.residual / 1e-18, bins=35, color='#1d6089')
axes[1].set(xlabel='Residual (10^-18)', ylabel='Coordinate count')
x = coord[coord.g == 1]
axes[2].scatter(x.dot_B_prediction_A, x.dot_B_actual, s=5, alpha=0.5, color='#725aa8')
li = x.dot_B_actual.abs().max() * 1.05
axes[2].plot([-li, li], [-li, li], 'k--', lw=0.7)
axes[2].set(title='g=1 expert complement', xlabel='Prediction', ylabel='Measured')
for ax in (axes[0], axes[2]):
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
savefig(fig, 'FIG_R63_COORDINATE_CLOSURE')
RESULT['C'] = dict(status='COMPLETE', coordinates=len(coord), max_H1_residual=float(abs(res).max()), max_A_null_field=float(null.field_A_max_abs.max()), dark_RMS_max=float(dark.actual_RMS12.max()))
note('R63_METHOD_NOTE.md', '# R63 coordinate-level closure\n\nActual derivatives and A-projected predictions are read directly from the archived EXPERT_SIDE_SELECTION_COORDINATES_v284M.csv, with all family/g/seed/direction/class/background/domain-pair labels preserved. No coordinate is inferred from a RMS cell summary. Residuals are recomputed as actual minus prediction and agree exactly with the saved column. All referenced field NPZ files are SHA256 matched to the archived direction ledger. Family/g/class/s stratified summaries retain bright and dark cases separately.\n\nThe H1 maximum and A-null field sentinel are source checks. The dark set is defined by the original child directions and parent directions at s=0, rather than selected by observed small response. The g=1 panel concerns the saved expert-side complement; the archive has no router-side g=1 derivative array in this R63 table, so this panel does not establish a newly measured router comparison. Exact/near-exact closure is an implementation/theory check in this declared response, not a precision claim for all neural observables.')
print('C COMPLETE', flush=True)
d = npz(D / 'T2_5033_TOMOGRAPHY.npz', 'D/E', 'canonical endpoint per-document matrices and sufficient contractions')
d0 = npz(D / 'T1_41B_TOMOGRAPHY.npz', 'E', 'canonical 41B per-document sufficient contractions')
draw = npz(D / 'T4_BOOTSTRAP_DRAWS.npz', 'D/E', 'original balanced-domain paired draw indices')['indices']
assert draw.shape == (2000, 7, 64)
source(D / 'work/final_t4.py', 'D/E', 'original physical whitening, bootstrap and factorization conventions')
for name in ['T1_41B_TOMOGRAPHY.npz', 'T2_5033_TOMOGRAPHY.npz']:
    p = BASE / 'Project1_B2_B3_SelectedArtifacts/selected/B3' / name
    source(p, 'D/E', 'staged candidate compared against canonical')
    assert sha(p) == sha(D / name)
assert np.array_equal(d['document_ids'], d0['document_ids']) and np.array_equal(d['domain'], d0['domain'])
hi = d['L15_H_inverse_sqrt']
H = d['L15_H']
assert np.allclose(hi @ H @ hi, np.eye(6), rtol=1e-10, atol=1e-12)

def eig(x):
    x = (x + np.swapaxes(x, -1, -2)) / 2
    e, u = np.linalg.eigh(x)
    return (e[..., ::-1], u[..., ::-1])

def geom(u, v, k):
    s = np.linalg.svd(np.swapaxes(u[..., :, :k], -1, -2) @ v[..., :, :k], compute_uv=False)
    s = np.clip(s, 0, 1)
    return (np.degrees(np.arccos(s)), np.mean(s * s, axis=-1))
mat = {}
ev = {}
vec = {}
bootmat = {}
boote = {}
bootv = {}
dh = []
br = []
for probe in ['local', 'objective']:
    key = f'L15_{probe}_J'
    rawmat = d[key]
    mat[probe] = hi @ rawmat.mean(0) @ hi
    assert np.allclose(mat[probe], d[key + '_physical'], rtol=1e-10, atol=1e-14)
    ev[probe], vec[probe] = eig(mat[probe])
    bm = hi @ rawmat[draw].mean((1, 2)) @ hi
    bootmat[probe] = bm
    boote[probe], bootv[probe] = eig(bm)
    ratio = ev[probe][1] / ev[probe][2]
    ratios = boote[probe][:, 1] / boote[probe][:, 2]
    ilo, ihi = ci(ratios)
    theta, qst = geom(vec[probe][None], bootv[probe], 2)
    align = np.abs(np.einsum('i,bi->b', vec[probe][:, 1], bootv[probe][:, :, 1])) ** 2
    dh.append(dict(record_type='eigenspectrum', probe=probe, metric='lambda2_over_lambda3', point_estimate=ratio, ci_low=ilo, ci_high=ihi, unit='ratio', individual_e2_claim='UNRESOLVED; inspect stability and gap', bootstrap_e2_overlap_median=float(np.median(align)), bootstrap_e2_overlap_ci_low=ci(align)[0], bootstrap_e2_overlap_ci_high=ci(align)[1], top2_to_point_angle2_median=float(np.median(theta[:, 1])), top2_to_point_angle2_ci_low=ci(theta[:, 1])[0], top2_to_point_angle2_ci_high=ci(theta[:, 1])[1]))
    for n in range(2000):
        br.append(dict(replicate=n, comparison=probe, lambda2_over_lambda3=ratios[n], e2_squared_overlap_with_point=align[n], top2_angle1_with_point_degrees=theta[n, 0], top2_angle2_with_point_degrees=theta[n, 1]))
    reg('D', f'{probe}_lambda2_over_lambda3', ratio, 'ratio', 'L15/5033B; physical control metric')
for k, expected in [(1, 0.9935), (2, 0.5065), (3, 0.7395)]:
    angle, q = geom(vec['local'], vec['objective'], k)
    ba, bq = geom(bootv['local'], bootv['objective'], k)
    check(f'D:Q{k}', q, expected, rtol=0, atol=5.01e-05)
    dh.append(dict(record_type='cross_probe', probe='local_vs_objective', metric=f'Q{k}', point_estimate=q, ci_low=ci(bq)[0], ci_high=ci(bq)[1], unit='overlap'))
    for j in range(k):
        dh.append(dict(record_type='cross_probe', probe='local_vs_objective', metric=f'top{k}_principal_angle{j + 1}', point_estimate=angle[j], ci_low=ci(ba[:, j])[0], ci_high=ci(ba[:, j])[1], unit='degree'))
        reg('D', f'top{k}_principal_angle{j + 1}', angle[j], 'degree', 'exact matrix-derived principal angle')
    for n in range(2000):
        br.append(dict(replicate=n, comparison=f'cross_probe_top{k}', Q=bq[n], **{f'angle{j + 1}_degrees': ba[n, j] for j in range(k)}))
for half in [0, 1]:
    el, vl = eig(d[f'L15_local_J_half{half}_physical'])
    eo, vo = eig(d[f'L15_objective_J_half{half}_physical'])
    ang, q = geom(vl, vo, 2)
    for j in range(2):
        dh.append(dict(record_type='split_half', probe='local_vs_objective', metric=f'half{half}_top2_principal_angle{j + 1}', point_estimate=ang[j], unit='degree'))
    for probe in ['local', 'objective']:
        eh, vh = eig(d[f'L15_{probe}_J_half{half}_physical'])
        ang2, qq = geom(vh, vec[probe], 2)
        dh.append(dict(record_type='split_half', probe=probe, metric=f'half{half}_lambda2_over_lambda3', point_estimate=eh[1] / eh[2], unit='ratio', e2_squared_overlap_with_point=float(abs(vh[:, 1] @ vec[probe][:, 1]) ** 2), top2_angle2_with_point_degrees=ang2[1]))
table('TOMO_L15_5033_SECONDARY_GEOMETRY.csv', dh)
table('TOMO_L15_5033_BOOTSTRAP.csv', br)
np.savez_compressed(OUT / 'TOMO_L15_5033_RECOVERED_OPERATORS.npz', H=H, H_inverse_sqrt=hi, local=mat['local'], objective=mat['objective'], local_bootstrap=bootmat['local'], objective_bootstrap=bootmat['objective'], draw_indices=draw)
angle, q = geom(vec['local'], vec['objective'], 2)
ba, bq = geom(bootv['local'], bootv['objective'], 2)
proxy1 = float(np.degrees(np.arccos(np.sqrt(0.9935))))
proxy2 = float(np.degrees(np.arccos(np.sqrt(2 * 0.5065 - 0.9935))))
identity = dict(Q1_scope='squared overlap of top-1 eigenvectors', Q2_scope='mean squared singular values of top-2 overlap', handoff_top2_identity_valid=False, top1_angle_from_rounded_Q1=proxy1, handoff_proxy_secondary_angle=proxy2, actual_top2_angle1=float(angle[0]), actual_top2_angle2=float(angle[1]), reason='Top-1 eigenvector overlap is not generally the first squared singular value of the top-2 overlap matrix.')
(OUT / 'TOMO_HANDOFF_IDENTITY_AUDIT.json').write_text(json.dumps(identity, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout='constrained')
for probe, color in [('local', '#1d6089'), ('objective', '#bd3b36')]:
    axes[0].hist(boote[probe][:, 1] / boote[probe][:, 2], bins=40, alpha=0.5, label=probe, color=color)
    axes[0].axvline(ev[probe][1] / ev[probe][2], color=color)
axes[0].set(xlabel='lambda2 / lambda3', ylabel='Paired bootstrap count')
axes[0].legend(fontsize=8)
for j, color in [(0, '#1d6089'), (1, '#bd3b36')]:
    axes[1].hist(ba[:, j], bins=40, alpha=0.5, color=color, label=f'angle {j + 1}')
    axes[1].axvline(angle[j], color=color)
axes[1].set(xlabel='Top-2 principal angle (degrees)', ylabel='Count')
axes[1].legend(fontsize=8)
for probe, color in [('local', '#1d6089'), ('objective', '#bd3b36')]:
    align = np.abs(np.einsum('i,bi->b', vec[probe][:, 1], bootv[probe][:, :, 1])) ** 2
    axes[2].hist(align, bins=40, alpha=0.5, label=probe, color=color)
axes[2].set(xlabel='Second-vector squared overlap with point', ylabel='Count')
axes[2].legend(fontsize=8)
savefig(fig, 'FIG_TOMO_SECONDARY_GEOMETRY')
RESULT['D'] = dict(status='COMPLETE_WITH_HANDOFF_IDENTITY_CORRECTION', actual_top2_angles=angle.tolist(), top2_angle_CI=[ci(ba[:, j]) for j in range(2)], local_gap=float(ev['local'][1] / ev['local'][2]), objective_gap=float(ev['objective'][1] / ev['objective'][2]), identity_audit=identity)
note('TOMO_SECONDARY_METHOD_NOTE.md', f"# Native secondary-subspace geometry\n\nGeneralized eigenproblem: Gamma v = lambda H v in the frozen six-control physical metric H. It is solved as eigh(H^(-1/2) Gamma H^(-1/2)), exactly as in final_t4.py, using saved H_inverse_sqrt. H is a control-basis Gram, not an internal feature metric. Direct per-document means reproduce saved physical matrices. Top-k principal angles use the SVD of the orthonormal whitened eigenspace overlap. No eigenvector sign is physically interpreted.\n\nUncertainty uses the exact archived 2,000 balanced-domain paired document draws (64 draws within each of seven 64-document domains, PCG64 seed 2026100701). Identical indices are reused across probes and checkpoints. CSV contains every replicate; point-versus-bootstrap subspace angles and individual second-vector squared overlaps are descriptive stability diagnostics. Split halves are kept separately. Ordering eigenvalues guarantees lambda2/lambda3 >= 1, so a bootstrap interval above 1 alone is not evidence that a second direction is stable. The objective secondary gap is weak; this recovery does not promote a resolved individual second-eigenvector claim.\n\nThe three Q sentinels are reproduced. However, the handoff's identity for the top-2 angles fails: Q1 is the top-1 eigenvector squared overlap, while Q2 is the mean squared singular values of the top-2 overlap. Q1 need not equal the leading singular value squared in top-2. Direct top-2 angles are {angle[0]:.8f} and {angle[1]:.8f} degrees; 95% paired intervals are {ci(ba[:, 0])} and {ci(ba[:, 1])}. The {proxy1:.5f}-degree leading-vector angle remains valid at rounded Q1; the {proxy2:.5f}-degree expression is an algebraic proxy using an unjustified identification, not a recovered top-2 angle. TOMO_HANDOFF_IDENTITY_AUDIT.json records this discrepancy without altering sealed Q values.")
print('D COMPLETE', flush=True)
perdoc = []
factor = []
erows = []
endpoint = readcsv(BASE / 'Project1_B2_B3_Reanalysis/B3_RATIO_COMPARISON.csv', 'E', 'archived factorization ratios')
for layer in [7, 15]:
    arrays = {}
    for t, z in [(41, d0), (5033, d)]:
        a = z[f'L{layer}_a_norm_sq']
        u = np.diagonal(z[f'L{layer}_u_inner_products'], axis1=1, axis2=2)
        g = z[f'L{layer}_g']
        den = a[:, None] * u
        c = np.divide(g * g, den, out=np.full_like(g, np.nan), where=den > 0)
        assert np.isfinite(c).all()
        assert np.max(c) <= 1 + 1e-05
        signed_cos = g / np.sqrt(den)
        arrays[t] = (a, u, g, c, signed_cos)
        for mode in range(6):
            for i in range(448):
                perdoc.append(dict(checkpoint_B=t, layer=layer, mode=mode + 1, document_id=str(z['document_ids'][i]), domain=str(z['domain'][i]), half=int(z['half'][i]), signed_g=g[i, mode], A=a[i], U=u[i, mode], c=c[i, mode], signed_cosine=signed_cos[i, mode], g_squared=g[i, mode] ** 2, document_weight=1 / 448, coordinate_convention='G0_EUCLIDEAN_FIXED_TENSOR_COORDINATES'))
    for mode in range(6):
        aggregates = {}
        boots = {}
        for t, (a, u, g, c, sc) in arrays.items():
            vals = {'A_bar': a.mean(), 'U_bar': u[:, mode].mean(), 'c_bar': c[:, mode].mean(), 'S': np.mean(g[:, mode] ** 2)}
            vals['chi'] = vals['S'] / (vals['A_bar'] * vals['U_bar'] * vals['c_bar'])
            vals['signed_g_mean'] = g[:, mode].mean()
            vals['signed_cosine_mean'] = sc[:, mode].mean()
            aggregates[t] = vals
            bt = {'A_bar': a[draw].mean((1, 2)), 'U_bar': u[:, mode][draw].mean((1, 2)), 'c_bar': c[:, mode][draw].mean((1, 2)), 'S': (g[:, mode] ** 2)[draw].mean((1, 2))}
            bt['chi'] = bt['S'] / (bt['A_bar'] * bt['U_bar'] * bt['c_bar'])
            bt['signed_g_mean'] = g[:, mode][draw].mean((1, 2))
            bt['signed_cosine_mean'] = sc[:, mode][draw].mean((1, 2))
            boots[t] = bt
            for field, value in vals.items():
                factor.append(dict(layer=layer, mode=mode + 1, checkpoint_B=t, statistic=field, estimate=value, ci_low=ci(bt[field])[0], ci_high=ci(bt[field])[1], scope='fixed tensor convention', uncertainty='2000 original balanced-domain paired document draws'))
        expected = endpoint[(endpoint.layer == layer) & (endpoint['mode'] == mode + 1) & (endpoint.metric_convention == 'G0_EUCLIDEAN')].iloc[0]
        for field in ['A_bar', 'U_bar', 'c_bar', 'S', 'chi']:
            r = aggregates[5033][field] / aggregates[41][field]
            rb = boots[5033][field] / boots[41][field]
            check(f'E:L{layer}/mode{mode + 1}/{field}_ratio', r, expected[field + '_ratio'])
            erows.append(dict(layer=layer, mode=mode + 1, statistic=field + '_5033_over_41', estimate=r, ci_low=ci(rb)[0], ci_high=ci(rb)[1], bootstrap_draws=2000, coordinate_convention='G0_EUCLIDEAN_FIXED_TENSOR_COORDINATES'))
        r = aggregates[5033]['S'] / aggregates[41]['S'] / (aggregates[5033]['U_bar'] / aggregates[41]['U_bar'])
        rb = boots[5033]['S'] / boots[41]['S'] / (boots[5033]['U_bar'] / boots[41]['U_bar'])
        check(f'E:L{layer}/mode{mode + 1}/objective_per_local', r, expected['objective_per_local_ratio'])
        erows.append(dict(layer=layer, mode=mode + 1, statistic='objective_per_local_ratio', estimate=r, ci_low=ci(rb)[0], ci_high=ci(rb)[1], bootstrap_draws=2000, coordinate_convention='G0_EUCLIDEAN_FIXED_TENSOR_COORDINATES'))
table('LOCAL_OBJECTIVE_FACTORIZATION.csv', perdoc)
table('LOCAL_OBJECTIVE_FACTOR_AGGREGATES.csv', factor)
table('LOCAL_OBJECTIVE_FACTOR_UNCERTAINTY.csv', erows)
RESULT['E'] = dict(status='COMPLETE', documentwise_rows=len(perdoc), documents=448, paired_endpoint_ratio_checks=72, coordinate_dependent=True)
note('LOCAL_OBJECTIVE_METHOD_NOTE.md', '# Local-to-objective factorization robustness\n\nThe archived six-mode per-document arrays provide signed g = a dot u, A = ||a||^2 and all <u_i,u_j>, sufficient to recover U, c = g^2/(A U), and signed cosine under G0, the fixed Euclidean internal tensor-coordinate convention. Full a/u vectors are not reconstructed from these sufficient contractions. Every row retains signed g; signed-cosine and signed-g averages are explicitly new descriptive summaries of saved signed contractions.\n\nEqual domains/equal documents means uniform 1/448 weighting (seven domains, 64 each). Aggregate S = E[g^2] is factored exactly as A_bar U_bar c_bar chi, with chi = S/(A_bar U_bar c_bar), the saved covariance/compensation residual. All five endpoint ratios and objective-per-local ratios reproduce the archived B3 G0 ratios. Confidence limits use the original 2,000 balanced-domain whole-document paired draws across endpoints, layers, controls and all factors; no independent factor resampling is performed. Within each draw the nonlinear factors are recomputed before taking endpoint ratios. All component arrays are finite and positive in this recovered cache, so zero-norm c is not silently assigned a value.\n\nThis decomposition depends on rescaling of internal tensor factors. H is a six-control coefficient metric and is not a replacement feature covariance. No coordinate-invariant causal attribution or intervention claim is made. Bootstrap intervals quantify this fixed Bank T sampling convention and do not measure arbitrary data/model uncertainty.')
print('E COMPLETE', flush=True)
table('SEALED_SOURCE_CHECKS_BCDE.csv', CHECKS)
table('RESULTS_REGISTRY_BCDE.csv', REG)
(OUT / 'SOURCE_MANIFEST_BCDE.json').write_text(json.dumps(SOURCES, indent=2))
(OUT / 'BRANCH_RESULTS_BCDE.json').write_text(json.dumps(RESULT, indent=2))
print(json.dumps(RESULT, indent=2), flush=True)
