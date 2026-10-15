# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys
from pathlib import Path
ROOT = Path(_p1_path('path_fd01557d84f7'))
OUT = ROOT / 'outputs/v285M_archive_completion'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'work/mplconfig')
import json, hashlib
import numpy as np, pandas as pd
sys.path.append(_p1_path('path_d4fe3b336f94'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
x = pd.read_csv(OUT / 'FINITE_TIME_THREE_WAY_CASES.csv', float_precision='round_trip')
assert len(x) == 360
checks = []

def check(label, actual, expected, rtol=5e-06):
    ok = bool(np.isclose(actual, expected, rtol=rtol, atol=1e-12))
    checks.append(dict(branch='A', check=label, actual=actual, expected=expected, rtol=rtol, atol=1e-12, pass_check=ok))
    assert ok, (label, actual, expected)
p = x[x.population == 'primary']
r = x[x.population == 'replication']
check('A:primary full paired refmean/sequence', p[p.sector == 'full_state'].refmean_over_sequence.median(), 88.7564)
check('A:primary reference marginal median error', p[p.sector == 'full_state'].refmean_original_error.median(), 0.00149573)
check('A:primary sequence marginal median error', p[p.sector == 'full_state'].sequence_original_error.median(), 1.67644e-05)
check('A:primary expert paired refmean/sequence', p[p.sector == 'expert_private'].refmean_over_sequence.median(), 1.13718)
check('A:primary router paired refmean/sequence approx', p[p.sector == 'router_private'].refmean_over_sequence.median(), 1200000.0, rtol=0.005)
check('A:replication full paired refmean/sequence', r[r.sector == 'full_state'].refmean_over_sequence.median(), 146.1)
bootfile = Path(_p1_path('path_3f476833c93e'))
with np.load(bootfile) as z:
    primary_draws = z['seed_identity_draws'].copy()
assert np.array_equal(primary_draws, np.random.default_rng(2026100901).integers(0, 6, (20000, 6)))
rep_draws = np.random.default_rng(2026100901).integers(0, 2, (20000, 2))
np.savez_compressed(OUT / 'FINITE_TIME_STREAM_BOOTSTRAP_DRAWS.npz', primary=primary_draws, replication=rep_draws, bootstrap_source=np.array(str(bootfile)))
rows = []
bootarrays = {}
fields = ['sequence_original_error', 'seqmean_original_error', 'refmean_original_error', 'sequence_absolute_error', 'seqmean_absolute_error', 'refmean_absolute_error', 'refmean_over_sequence', 'seqmean_over_sequence', 'refmean_over_seqmean']
for population, draws in [('primary', primary_draws), ('replication', rep_draws)]:
    for sector in ['full_state', 'expert_private', 'router_private']:
        y = x[(x.population == population) & (x.sector == sector)].sort_values(['stream', 'arm'])
        ns = y.stream.nunique()
        assert len(y) == ns * 15
        for f in fields:
            val = y[f].to_numpy().reshape(ns, 15)
            bt = np.median(val[draws].reshape(20000, -1), axis=1)
            lo, hi = np.quantile(bt, [0.025, 0.975])
            bootarrays[f'{population}_{sector}_{f}'] = bt
            rows.append(dict(population=population, sector=sector, statistic=f, paired_case_median=float(np.median(val)), marginal_mean=float(val.mean()), ci_low=float(lo), ci_high=float(hi), stream_clusters=ns, arms_per_stream=15, cases=len(y), bootstrap_draws=20000, uncertainty='whole-stream clustered percentile bootstrap', replication_boundary='two clusters only' if population == 'replication' else 'six historical clusters'))
s = pd.DataFrame(rows)
s.to_csv(OUT / 'FINITE_TIME_THREE_WAY_SUMMARY.csv', index=False, float_format='%.17g')
np.savez_compressed(OUT / 'FINITE_TIME_THREE_WAY_BOOTSTRAP_STATISTICS.npz', **bootarrays)
pd.DataFrame(checks).to_csv(OUT / 'SEALED_SOURCE_CHECKS_A.csv', index=False, float_format='%.17g')
fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
for row, population in enumerate(['primary', 'replication']):
    for col, sector in enumerate(['full_state', 'expert_private', 'router_private']):
        ax = axes[row, col]
        y = s[(s.population == population) & (s.sector == sector)].set_index('statistic')
        for i, (key, color) in enumerate([('refmean_original_error', '#bd3b36'), ('seqmean_original_error', '#b58a1e'), ('sequence_original_error', '#1d6089')]):
            z = y.loc[key]
            ax.errorbar(i, z.paired_case_median, yerr=[[max(0, z.paired_case_median - z.ci_low)], [max(0, z.ci_high - z.paired_case_median)]], fmt='o', color=color, capsize=4)
        ax.set_yscale('log')
        ax.set_xticks([0, 1, 2], ['Ref mean', 'Seq mean', 'Ordered'], rotation=15)
        ax.set_title(f"{population}: {sector.replace('_', ' ')}")
        ax.set_ylabel('Original displacement-relative error')
fig.savefig(OUT / 'FIG_FINITE_TIME_THREE_WAY.pdf', bbox_inches='tight')
fig.savefig(OUT / 'FIG_FINITE_TIME_THREE_WAY.png', dpi=300, bbox_inches='tight')
plt.close(fig)
reg = []
for y in rows:
    if 'over_' in y['statistic']:
        reg.append(dict(branch='A', statistic=y['statistic'], value=y['paired_case_median'], unit='ratio', scope=f"{y['population']}/{y['sector']}", status='sealed_comparator_reproduced' if y['statistic'] == 'refmean_over_sequence' else 'new_derived'))
pd.DataFrame(reg).to_csv(OUT / 'RESULTS_REGISTRY_A.csv', index=False, float_format='%.17g')
summary = {}
for population in ['primary', 'replication']:
    y = x[(x.population == population) & (x.sector == 'full_state')]
    summary[population] = dict(refmean_over_sequence_paired_median=float(y.refmean_over_sequence.median()), seqmean_over_sequence_paired_median=float(y.seqmean_over_sequence.median()), refmean_over_seqmean_paired_median=float(y.refmean_over_seqmean.median()), sequence_original_median=float(y.sequence_original_error.median()), seqmean_original_median=float(y.seqmean_original_error.median()), refmean_original_median=float(y.refmean_original_error.median()))
(OUT / 'BRANCH_RESULTS_A.json').write_text(json.dumps(dict(status='COMPLETE', sentinel_checks=len(checks), full_state=summary), indent=2))
(OUT / 'FINITE_TIME_METHOD_NOTE.md').write_text('# Same-sequence mean versus ordered versus reference-mean propagator\n\nThe original local map on (M,U) is L_n = [[a_M I, eta_M C_n^T], [eta_U C_n, a_U I]], with a_M=1-eta_M mu and a_U=1-eta_U lambda. Exact archived h/phi/g factors give C_n(M)=-1/4 sum_tokens h (g dot (phi M^T)) and its transpose. All 885,312 expert-private and 576 router-private coordinates are retained. Ordered application of L_31 ... L_0 and 32 repeated applications of the same-stream average map are computed at the original x0; a dense 885,888-square product is unnecessary for the original evaluation states. The factors, operator actions and all arm parameters are preserved; no pooled projection defines the primary error.\n\nThe per-step factors were not directly archived. They are recovered by the exact frozen local model/bank/capture code in its original runtime, with model weights held fixed, no optimizer and offline networking. All 256 h/phi/g triplets match the sealed tensor SHA256 values before float64 propagation. Every ordered per-case full/expert/router relative error reproduces the sealed full-space comparator (rtol=1e-10, atol=1e-12). The independent reference mean comparator is retained from the archived full-space M1 per-case errors, as required by the handoff. Every original target checkpoint used by recovery is byte-hashed by the original verification routine. This is deterministic frozen-model derivative recovery, not model training or a new sampled experiment.\n\nFor the same-stream average, Cbar Cbar^T is computed exactly using cross-token Gram algebra and the full M is reconstructed at step 32. The recurrence uses both right-hand sides at the previous step. Error targets are delta M=M32-M0 and U32, with the original displacement-relative denominators and epsilon=1e-30. Absolute Euclidean errors are also exported. Ratios are paired per case first, then the pooled 90-case primary or 30-case replication median is taken; the ratio of marginal medians is never substituted.\n\nConfidence limits newly quantify the same pooled case statistic by resampling whole streams, retaining all 15 paired arms. Primary uncertainty reuses the original P2Q 20,000 six-stream draw indices (seed 2026100901). Replication uses 20,000 two-stream paired draws with that seed; with only two clusters those limits have severely limited support and are descriptive. These intervals are not a newly redefined frozen decision criterion. Ordered-control results are not interpreted as a fraction of the total mean-map gap. No missing checkpoint was interpolated.\n')
print(json.dumps(summary, indent=2))
