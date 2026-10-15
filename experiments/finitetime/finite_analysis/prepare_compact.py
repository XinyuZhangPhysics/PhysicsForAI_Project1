# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
W = R / 'work'
O = R / 'outputs'
A = R.parents[1] / '2026-10-12/g-j/outputs/v285M_archive_completion'
V = Path(_p1_path('path_9daa8b272cd4'))
PACK = O / 'v286M_compact_evidence_pack'
PACK.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(W / 'mplconfig')
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, pandas as pd, shutil, json, hashlib, zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sources = {}
dependencies = []
omitted = []

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def record(p, dest, role='direct publication input'):
    p = Path(p)
    sources[str(p)] = dict(source_path=str(p), sha256=sha(p), size_bytes=p.stat().st_size, publication_outputs=dest, role=role)

def copy(p, name=None):
    p = Path(p)
    name = name or p.name
    dst = PACK / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dst)
    record(p, name)
    return dst

def csv(p):
    record(p, 'derived compact tables')
    return pd.read_csv(p, float_precision='round_trip')
for name in ['RESULTS_SUMMARY.md', 'RESULTS_REGISTRY_APPEND.csv', 'FINITE_TIME_THREE_WAY_SUMMARY.csv', 'FINITE_TIME_THREE_WAY_CASES.csv', 'MODE_CONVERSION_SIGNED_SUMMARY.csv', 'R63_COORDINATE_CLOSURE_SUMMARY.csv', 'TOMO_L15_5033_SECONDARY_GEOMETRY.csv', 'TOMO_HANDOFF_IDENTITY_AUDIT.json', 'LOCAL_OBJECTIVE_FACTOR_AGGREGATES.csv', 'LOCAL_OBJECTIVE_FACTOR_UNCERTAINTY.csv']:
    copy(A / name)
for p in A.glob('*METHOD_NOTE.md'):
    copy(p)
for name in ['FINITE_TIME_THREE_WAY', 'MODE_CONVERSION_SIGNED', 'R63_COORDINATE_CLOSURE', 'TOMO_SECONDARY_GEOMETRY']:
    copy(A / f'FIG_{name}.pdf')
    im = Image.open(A / f'FIG_{name}.png')
    im.thumbnail((1500, 1100))
    im.save(PACK / f'FIG_{name}_PREVIEW.png')
    record(A / f'FIG_{name}.png', f'FIG_{name}_PREVIEW.png')
for sector in ['C', 'M']:
    copy(A / f'FIG_MODE_CONVERSION_SIGNED_{sector}_ALL_COORDINATES.pdf')
x = csv(A / 'MODE_CONVERSION_SIGNED_CURVE.csv')
x[x.coordinate == 0].to_csv(PACK / 'MODE_CONVERSION_COORDINATE0_FIGURE_DATA.csv', index=False, float_format='%.17g')
x = csv(A / 'MODE_CONVERSION_SIGNED_SUMMARY.csv')
x.to_csv(PACK / 'MODE_CONVERSION_ALL_COORDINATES_COMPACT.csv', index=False, float_format='%.17g')
p = A / 'R63_COORDINATE_CLOSURE.csv'
copy(p)
x = csv(p)
x[['family', 'g', 'class', 's', 'coordinate', 'dot_B_actual', 'dot_B_prediction_A', 'residual']].to_csv(PACK / 'R63_MAIN_FIGURE_DATA.csv', index=False, float_format='%.17g')
x = csv(A / 'TOMO_L15_5033_BOOTSTRAP.csv')
rows = []
for comp, y in x.groupby('comparison'):
    for f in y.select_dtypes(include=np.number):
        if f == 'replicate' or y[f].notna().sum() == 0:
            continue
        zz = y[f].dropna()
        q = zz.quantile([0, 0.025, 0.25, 0.5, 0.75, 0.975, 1])
        rows.append(dict(comparison=comp, statistic=f, draws=len(zz), minimum=q[0], q025=q[0.025], q25=q[0.25], median=q[0.5], q75=q[0.75], q975=q[0.975], maximum=q[1]))
pd.DataFrame(rows).to_csv(PACK / 'TOMO_BOOTSTRAP_QUANTILES.csv', index=False, float_format='%.17g')
copy(A / 'TOMO_L15_5033_BOOTSTRAP.csv')
p = A / 'TOMO_L15_5033_RECOVERED_OPERATORS.npz'
record(p, 'TOMO_ENDPOINT_EIGENVALUES.csv')
with np.load(p) as z:
    print('tomography operator keys', z.files, flush=True)
D = R.parents[1] / '2026-10-08/physicsforai-project-1-p2d-agents-md/outputs/P2D_CERTIFIED_RESUME_20261008_103610'
rows = []
for chk, name in [(41, 'T1_41B_TOMOGRAPHY.npz'), (5033, 'T2_5033_TOMOGRAPHY.npz')]:
    p = D / name
    record(p, 'TOMO_ENDPOINT_EIGENVALUES.csv')
    with np.load(p) as z:
        for layer in [7, 15]:
            hi = z[f'L{layer}_H_inverse_sqrt']
            for probe in ['local', 'objective']:
                h = hi @ z[f'L{layer}_{probe}_J'].mean(0) @ hi
                ev = np.linalg.eigvalsh((h + h.T) / 2)[::-1]
                for i, v in enumerate(ev):
                    rows.append(dict(checkpoint_B=chk, layer=layer, probe=probe, index=i + 1, eigenvalue=v, next_eigenvalue=ev[i + 1] if i < 5 else np.nan, gap=v - ev[i + 1] if i < 5 else np.nan, ratio_to_next=v / ev[i + 1] if i < 5 else np.nan))
pd.DataFrame(rows).to_csv(PACK / 'TOMO_ENDPOINT_EIGENVALUES.csv', index=False, float_format='%.17g')
x = csv(A / 'LOCAL_OBJECTIVE_FACTOR_UNCERTAINTY.csv')
assert len(x) == 72
x.to_csv(PACK / 'LOCAL_OBJECTIVE_ENDPOINT_72_RATIOS.csv', index=False, float_format='%.17g')
y = csv(A / 'LOCAL_OBJECTIVE_FACTORIZATION.csv')
group = ['layer', 'mode', 'checkpoint_B']
s = []
for key, z in y.groupby(group):
    row = dict(zip(group, key))
    for f in z.select_dtypes(include=np.number):
        if any((t in f.lower() for t in ['signed', 'cos', 'g'])):
            row[f + '_mean'] = z[f].mean()
            row[f + '_median'] = z[f].median()
            row[f + '_rms'] = float(np.sqrt(np.mean(z[f] ** 2)))
    s.append(row)
pd.DataFrame(s).to_csv(PACK / 'LOCAL_OBJECTIVE_SIGNED_COHERENCE.csv', index=False, float_format='%.17g')
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), layout='constrained')
for ax, (layer, z) in zip(axes, x.groupby('layer')):
    for i, (stat, t) in enumerate(z.groupby('statistic')):
        ax.errorbar(t['mode'] + i * 0.035 - 0.1, t.estimate, yerr=[t.estimate - t.ci_low, t.ci_high - t.estimate], fmt='o', ms=3, capsize=2, label=stat.replace('_5033_over_41', '').replace('objective_per_local_ratio', 'objective/local'))
    ax.set_yscale('log')
    ax.set_title(f'L{layer}: endpoint factor ratios')
    ax.set_xlabel('Control direction')
    ax.set_ylabel('5033B / 41B')
axes[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5))
fig.savefig(PACK / 'FIG_LOCAL_OBJECTIVE_FACTORS.pdf')
fig.savefig(PACK / 'FIG_LOCAL_OBJECTIVE_FACTORS_PREVIEW.png', dpi=130)
plt.close(fig)
copy(V / 'figure_data/FULL_TANGENT_CELL_DIAGNOSTICS.csv')
copy(V / 'evidence_snapshots/FULL_TANGENT_AUDIT_v283M.md')
p = V / '17_NUMERICAL_RESULT_REGISTRY.csv'
x = csv(p)
x[x.experiment.str.contains('Full tangent|Q3', case=False, na=False)].to_csv(PACK / 'FULL_TANGENT_Q3_EXISTING_SUMMARIES.csv', index=False)
p = V / '02_EXPERIMENT_AUTHORING_SOURCE.md'
record(p, 'Q3_METHOD_AND_SCIENTIFIC_BOUNDARY.md')
s = p.read_text()
part = s[s.index('# 10. Q3'):s.index('# 11. Finite-time')]
(PACK / 'Q3_METHOD_AND_SCIENTIFIC_BOUNDARY.md').write_text(part)
orig = pd.read_csv(A / 'SOURCE_MANIFEST.csv')
keep = orig[orig.branch.isin(['B', 'C', 'D/E', 'E']) & ~orig.role.str.contains('staged candidate|archived SHA256', case=False, na=False)]
for row in keep.to_dict('records'):
    p = Path(row['source_path'])
    record(p, 'v285M publication B-E tables and figures', 'original analytical dependency: ' + row['role'])
for p in sorted(A.glob('*.csv')):
    if p.name not in [Path(k).name for k in sources] and p.stat().st_size > 500000:
        omitted.append(dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size, reason='raw or redundant detail; compact summary retained'))
for p in [A / 'MODE_CONVERSION_SIGNED_RESPONSES.csv', A / 'TOMO_L15_5033_RECOVERED_OPERATORS.npz', A / 'LOCAL_OBJECTIVE_FACTORIZATION.csv']:
    if p.name not in [q.name for q in PACK.iterdir()]:
        omitted.append(dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size, reason='publication summaries and redraw data retained'))
(W / 'compact_sources.json').write_text(json.dumps(list(sources.values()), indent=2))
(W / 'compact_omitted.json').write_text(json.dumps(omitted, indent=2))
print('compact stage ready', len(list(PACK.iterdir())), flush=True)
