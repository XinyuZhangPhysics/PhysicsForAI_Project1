# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys, json, math, hashlib, shutil, zipfile, ast
from pathlib import Path
R = Path(__file__).resolve().parents[1]
W = R / 'work'
O = R / 'outputs'
N = O / 'v286M_finite_time_mechanism'
K = O / 'v286M_compact_evidence_pack'
A = R.parents[1] / '2026-10-12/g-j/outputs/v285M_archive_completion'
os.environ['MPLCONFIGDIR'] = str(W / 'mplconfig')
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(name):
    return pd.read_csv(N / name, float_precision='round_trip')
for name in ['SOURCE_INTEGRITY_VERIFICATION.json', 'MAP_LOG_AND_SAMPLING_VERIFICATION.json', 'ALGEBRA_VERIFICATION.json', 'V285M_SEALED_PAYLOAD_VERIFICATION.json']:
    assert json.loads((N / name).read_text())['status'] == 'PASS'
x = read('FINITE_TIME_CUMULANTS.csv')
proj = read('FINITE_TIME_CUMULANT_PROJECTED_NORMS.csv')
assert len(x) == len(proj) == 360
x = x.drop(columns=[c for c in x if c.endswith('projected_frobenius') or c.endswith('projected_spectral')], errors='ignore')
x = x.merge(proj, on=['population', 'stream', 'arm', 'sector'], validate='one_to_one')
a = read('FINITE_TIME_MAP_LOG_STABILITY.csv')
q = a[a.bank == 'sequence'].groupby(['stream', 'arm']).norm_L_minus_I.max()
x['full_log_norm_upper_bound'] = [-np.log1p(-q.loc[s, k]) for s, k in zip(x.stream, x.arm)]
x['first_cumulant_spectral_upper_bound'] = 32 * x.full_log_norm_upper_bound
x['commutator_spectral_upper_bound'] = 496 * x.full_log_norm_upper_bound ** 2
xx = 32 * x.full_log_norm_upper_bound + 496 * x.full_log_norm_upper_bound ** 2
x['exponential_operator_18term_remainder_bound'] = np.exp(xx) * xx ** 19 / math.factorial(19)
x.to_csv(N / 'FINITE_TIME_CUMULANTS.csv', index=False, float_format='%.17g')
s = read('FINITE_TIME_STREAM_MEANS.csv')
ss = read('FINITE_TIME_SAMPLE_SIZE_CASES.csv')
sh = read('FINITE_TIME_MEAN_SHIFT_CASES.csv')
assert len(s) == 1440 and len(ss) == 2160 and (len(sh) == 120)
with np.load(A / 'FINITE_TIME_STREAM_BOOTSTRAP_DRAWS.npz') as z:
    boot = {p: z[p].copy() for p in ['primary', 'replication']}
rows = []
for (pop, sector), z in x.groupby(['population', 'sector']):
    for f in [c for c in x if c.endswith('projected_frobenius') or c.endswith('projected_spectral')]:
        val = z.sort_values(['stream', 'arm'])[f].to_numpy().reshape(6 if pop == 'primary' else 2, 15)
        bt = np.median(val[boot[pop]].reshape(20000, -1), 1)
        lo, hi = np.quantile(bt, [0.025, 0.975])
        rows.append(dict(population=pop, sector=sector, statistic=f, median=np.median(val), ci_low=lo, ci_high=hi, clusters=len(val), arms_per_cluster=15, bootstrap_draws=20000))
ag = read('FINITE_TIME_CUMULANT_AGGREGATES.csv')
ag = ag[~ag.statistic.str.contains('projected_frobenius|projected_spectral')]
ag = pd.concat([ag, pd.DataFrame(rows)], ignore_index=True)
ag.to_csv(N / 'FINITE_TIME_CUMULANT_AGGREGATES.csv', index=False, float_format='%.17g')
controls = []
for pop, streams in [('primary', list(range(6))), ('replication', [6, 7])]:
    e = np.stack([np.load(W / f'subsets_stream{j}_m4.npz')['errors'] for j in streams], axis=1)
    for t, sector in enumerate(['full_state', 'expert_private', 'router_private']):
        z = s[(s.population == pop) & (s.sector == sector) & (s.comparator == 'same_stream_reference')]
        stat = z.closure_error.median()
        dist = np.median(e[:, :, :, t].reshape(1024, -1), 1)
        lo, hi = np.quantile(dist, [0.025, 0.975])
        pct = (np.count_nonzero(dist < stat) + 0.5 * np.count_nonzero(dist == stat)) / len(dist)
        controls.append(dict(population=pop, sector=sector, same_stream_reference_case_median=stat, four_subset_joint_median=np.median(dist), four_subset_joint_q025=lo, four_subset_joint_q975=hi, same_stream_reference_sampling_percentile=pct, survives_four_map_sampling_control=bool(stat < lo or stat > hi), individual_cases_outside_control_fraction=z.same_stream_outside_95_sampling_control.mean(), monte_carlo_draws=1024))
co = pd.DataFrame(controls)
co.to_csv(N / 'FINITE_TIME_FOUR_MAP_SAMPLING_CONTROL.csv', index=False, float_format='%.17g')
old = pd.read_csv(A / 'FINITE_TIME_THREE_WAY_CASES.csv', float_precision='round_trip')
vr = []
for comp, col in [('global_reference', 'refmean_original_error'), ('same_sequence', 'seqmean_original_error')]:
    z = s[s.comparator == comp].merge(old[['stream', 'arm', 'sector', col]], on=['stream', 'arm', 'sector'], validate='one_to_one')
    assert np.allclose(z.closure_error, z[col], rtol=2e-08, atol=2e-12)
    vr.extend((dict(stream=r.stream, arm=r.arm, sector=r.sector, comparator=comp, full_coordinate_recomputed_error=r.closure_error, archived_error=getattr(r, col), absolute_difference=abs(r.closure_error - getattr(r, col))) for r in z.itertuples()))
pd.DataFrame(vr).to_csv(N / 'BASELINE_FULL_COORDINATE_REPROPAGATION.csv', index=False, float_format='%.17g')
v = read('NUMERICAL_VERIFICATION.csv').rename(columns={'max_relative_norm_identity_error': 'unanchored_quadratic_relative_cancellation_error'})
v.to_csv(N / 'NUMERICAL_VERIFICATION.csv', index=False, float_format='%.17g')

def get(frame, pop, sector, stat):
    z = frame[(frame.population == pop) & (frame.statistic == stat)]
    if 'sector' in z:
        z = z[z.sector == sector]
    y = z.iloc[0]
    return dict(median=float(y['median']), ci_low=float(y.ci_low), ci_high=float(y.ci_high))
shiftag = read('FINITE_TIME_MEAN_SHIFT_AGGREGATES.csv')
sampleag = read('FINITE_TIME_SAMPLE_SIZE.csv')
streamag = read('FINITE_TIME_STREAM_MEAN_AGGREGATES.csv')
headline = dict(status='COMPLETE', baseline=json.loads((N / 'BASELINE_VERIFICATION.json').read_text()), dimensions=dict(expert_private=576 * 1536, router_private=576, total=576 * 1536 + 576, declared_projection_expert=24, declared_projection_router=24), mean_shift={}, cumulants={}, four_map_sampling_control=controls, sample_size=[], stream_means=[], verification={})
for pop in ['primary', 'replication']:
    headline['mean_shift'][pop] = {f: get(shiftag, pop, None, f) for f in ['mean_shift_frobenius', 'mean_shift_spectral', 'action_full_norm', 'action_expert_norm', 'action_router_norm', 'action_router_projected_norm', 'alignment_observed_full', 'alignment_observed_router', 'projected_action_capture_fraction', 'projected_top_singular']}
    headline['cumulants'][pop] = {sector: {f: get(ag, pop, sector, f) for f in ['first_cumulant_action_norm', 'commutator_action_norm', 'ordered_vs_seqmean_residual_norm', 'second_order_fraction_residual_explained', 'second_cumulant_fraction_first_error_reduced', 'first_cumulant_vs_ordered_action_error', 'second_cumulant_vs_ordered_action_error', 'first_cumulant_projected_frobenius', 'commutator_projected_frobenius']} for sector in ['full_state', 'expert_private', 'router_private']}
headline['sample_size'] = sampleag.to_dict('records')
headline['stream_means'] = streamag.to_dict('records')
headline['verification']['map_log_and_sampling'] = json.loads((N / 'MAP_LOG_AND_SAMPLING_VERIFICATION.json').read_text())
headline['verification']['algebra'] = json.loads((N / 'ALGEBRA_VERIFICATION.json').read_text())
headline['verification']['source_integrity'] = json.loads((N / 'SOURCE_INTEGRITY_VERIFICATION.json').read_text())
headline['analysis_design'] = dict(grid=[1, 2, 4, 8, 16, 32], subset_draws={1: 32, 2: 256, 4: 1024, 8: 256, 16: 256, 32: 1}, sampling='uniform without replacement within each realized sequence; deterministic independent seeded draws', rng='NumPy PCG64 SeedSequence([20261012,286,stream,m])', bootstrap='20,000 original whole-stream draws; 15 paired arms retained', reference_bank='8 streams x 4 fixed maps, updates 1/17/33/49; shares sequence maps 1/17, so not wholly disjoint', replication_boundary='only two clusters, descriptive intervals', concentration_only=True)
(N / 'FINITE_TIME_MECHANISM_SUMMARY.json').write_text(json.dumps(headline, indent=2, allow_nan=False))
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.1), layout='constrained')
blue = '#20658b'
gold = '#bd8a20'
red = '#b84135'
green = '#318465'
base = pd.read_csv(A / 'FINITE_TIME_THREE_WAY_SUMMARY.csv')
ax = axes[0, 0]
for i, pop in enumerate(['primary', 'replication']):
    for t, key in enumerate(['refmean_original_error', 'seqmean_original_error', 'sequence_original_error']):
        y = base[(base.population == pop) & (base.sector == 'full_state') & (base.statistic == key)].iloc[0]
        ax.errorbar(t + i * 0.18 - 0.09, y.paired_case_median, yerr=[[y.paired_case_median - y.ci_low], [y.ci_high - y.paired_case_median]], fmt=['o', 's'][i], color=[red, gold, blue][t], capsize=3, label=pop if t == 0 else None)
ax.set_yscale('log')
ax.set_xticks([0, 1, 2], ['Reference', 'Sequence mean', 'Ordered'])
ax.set_ylabel('Full-state closure error')
ax.set_title('a  Three-way finite-time baseline', loc='left')
ax.legend(fontsize=8)
ax = axes[0, 1]
for pop, color in [('primary', blue), ('replication', gold)]:
    z = sampleag[(sampleag.population == pop) & (sampleag.sector == 'full_state') & (sampleag.statistic == 'subset_median')]
    ax.plot(z.m, z['median'], 'o-', color=color, label=pop)
    ax.fill_between(z.m, z.ci_low, z.ci_high, alpha=0.15, color=color)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
ax.set_xlabel('Maps in realized-sequence subset')
ax.set_ylabel('Median full-state closure error')
ax.set_title('b  Concentration diagnostic', loc='left')
ax.legend(fontsize=8)
ax = axes[0, 2]
labels = ['Pooled ref', 'Same-stream ref', 'Leave-out ref', 'Seq mean']
keys = ['global_reference', 'same_stream_reference', 'leave_one_stream_out_reference', 'same_sequence']
for pop, offset, col in [('primary', -0.07, blue), ('replication', 0.07, gold)]:
    for i, key in enumerate(keys):
        z = streamag[(streamag.population == pop) & (streamag.sector == 'full_state') & (streamag.comparator == key) & (streamag.statistic == 'closure_error')].iloc[0]
        ax.errorbar(i + offset, z['median'], yerr=[[z['median'] - z.ci_low], [z.ci_high - z['median']]], fmt='o', color=col, capsize=2)
    z = co[(co.population == pop) & (co.sector == 'full_state')].iloc[0]
    ax.hlines(z.four_subset_joint_median, 0.5, 1.5, color=col, ls='--')
    ax.fill_between([0.5, 1.5], z.four_subset_joint_q025, z.four_subset_joint_q975, color=col, alpha=0.13)
ax.set_yscale('log')
ax.set_xticks(range(4), labels, rotation=24, ha='right')
ax.set_ylabel('Full-state closure error')
ax.set_title('c  Four-map sampling control', loc='left')
ax = axes[1, 0]
for i, (f, label) in enumerate([('alignment_observed_router', 'Router alignment'), ('projected_action_capture_fraction', '24D action capture')]):
    for pop, off, col in [('primary', -0.12, blue), ('replication', 0.12, gold)]:
        z = shiftag[(shiftag.population == pop) & (shiftag.statistic == f)].iloc[0]
        ax.errorbar(i + off, z['median'], yerr=[[z['median'] - z.ci_low], [z.ci_high - z['median']]], fmt='o', color=col, capsize=3)
ax.set_xticks([0, 1], ['Router alignment', '24D capture'])
ax.set_ylim(-1.05, 1.05)
ax.set_ylabel('Cosine / norm fraction')
ax.set_title('d  Mean-shift action at original state', loc='left')
ax.text(0.03, 0.04, 'Expert action = 0 because U(0) = 0', transform=ax.transAxes, fontsize=8)
ax = axes[1, 1]
for pop, off, col in [('primary', -0.08, blue), ('replication', 0.08, gold)]:
    for i, stat in enumerate(['first_cumulant_action_norm', 'commutator_action_norm', 'ordered_vs_seqmean_residual_norm']):
        z = ag[(ag.population == pop) & (ag.sector == 'full_state') & (ag.statistic == stat)].iloc[0]
        ax.errorbar(i + off, z['median'], yerr=[[z['median'] - z.ci_low], [z.ci_high - z['median']]], fmt='o', color=col, capsize=2)
ax.set_xticks([0, 1, 2], ['First cumulant', 'Commutator', 'Ordered - mean'], rotation=18)
ax.set_yscale('log')
ax.set_ylabel('Action / residual Euclidean norm')
ax.set_title('e  Two finite-time scales', loc='left')
ax = axes[1, 2]
for pop, off, col in [('primary', -0.08, blue), ('replication', 0.08, gold)]:
    for i, sector in enumerate(['full_state', 'expert_private', 'router_private']):
        z = ag[(ag.population == pop) & (ag.sector == sector) & (ag.statistic == 'second_order_fraction_residual_explained')].iloc[0]
        ax.errorbar(i + off, 100 * z['median'], yerr=[[100 * (z['median'] - z.ci_low)], [100 * (z.ci_high - z['median'])]], fmt='o', color=col, capsize=3)
ax.set_ylim(96.8, 100.2)
ax.set_xticks([0, 1, 2], ['Full state', 'Expert', 'Router'])
ax.set_ylabel('Ordered-vs-mean residual\nexplained (%)')
ax.set_title('f  Second-order correction', loc='left')
fig.savefig(N / 'FIG_FINITE_TIME_MECHANISM.pdf')
fig.savefig(N / 'FIG_FINITE_TIME_MECHANISM.png', dpi=160)
plt.close(fig)
method = "# Finite-time mechanism method and numerical contract\n\nNo model is loaded by this analysis and no optimizer or training runs. Recovered float32 h/phi/g factors are promoted to float64. All 884,736 expert-private and 576 router-private coordinates are retained (885,312 total); the older method note's 885,312 expert count / 885,888 total is a prose arithmetic typo. The 24+24 archived orthonormal basis is used only for declared projection diagnostics.\n\n## Baseline and state convention\n\nL_i = [[a_M I, eta_M C_i^T], [eta_U C_i, a_U I]], C_i(M)=-1/4 sum_tokens h (g dot (phi M^T)); a_M=1-eta_M mu and a_U=1-eta_U lambda. Original M0 is retained and U0=0. Targets are M32-M0 and U32. Errors use full/expert/router observed-displacement norm denominators and epsilon=1e-30. The exact ordered product, pooled reference mean and realized-sequence mean are recomputed and matched case by case. A1 uses paired per-case ratios before pooled case medians, never ratios of marginal medians.\n\n## Mean-shift operator\n\nDelta L has zero diagonal and off-diagonal blocks eta_M Delta C^T and eta_U Delta C. Its exact Frobenius norm is sqrt(eta_M^2+eta_U^2)||Delta C||_F; spectral norm is max(eta_M,eta_U)||Delta C||_2. Delta C Delta C^T is computed with full-router cross-token Gram algebra. On x0=(M0,0), its expert component is exactly zero and router component eta_U Delta C M0. Expert displacement alignment is undefined for a zero action. Full and router cosines are reported separately. Declared projected singular vectors refer to P^T Delta L P, coordinates [M24,U24], Euclidean normalized; they are not full-space singular vectors or new modes.\n\n## Fixed references, sampling and uncertainty\n\nThe reference bank has four fixed maps per stream (updates 1,17,33,49), 32 maps globally. Maps 1/17 coincide with sequence maps, so this bank is not wholly disjoint from the realized sequence. Same-stream and leave-one-stream-out means are explicit averages of these original maps. Sampling uses uniform subsets without replacement within each realized 32-map sequence, seeded by NumPy PCG64 SeedSequence([20261012,286,stream,m]). Draw counts: m=1 exhausts 32 singletons; m=2 has 256 draws; m=4 has 1,024; m=8 and 16 have 256; m=32 is the unique full subset. Draws may repeat a subset. This is a fixed finite-population concentration diagnostic, not held-out prediction. Zero sampling variance at m=32 follows by construction and does not establish independent 32-map generalization.\n\nSampling-control intervals are within-sequence draw quantiles, separate from 20,000 whole-stream percentile bootstrap intervals over pooled per-case subset medians. Original stream bootstrap indices retain all 15 paired arms, six primary/two replication clusters. Replication intervals are descriptive due to only two clusters. A joint four-map control pools paired cases per simultaneous independently seeded subset draw. Its percentiles are conditional sampling diagnostics, not exchangeable hypothesis-test p values. Stream specificity is not claimed unless the same-stream result lies outside this finite-four-map control; surviving the control alone would still not establish a causal stream identity mechanism.\n\n## Exact mean propagation and numerical stability\n\nFull-state repeated-mean propagation uses exact degree-15 polynomials in G=Cbar Cbar^T. Recurrence p_{n+1}=a_M p_n+eta_M q_n, q_{n+1}=a_U q_n+eta_U a_M^n+eta_U G p_n, M_n=a_M^n M0+Cbar^T p_n(G) Cbar M0, U_n=q_n(G) Cbar M0. Publication comparator errors are computed after literal full-coordinate state reconstruction. Subset expert errors use an equivalent Gram norm identity anchored to the literal same-sequence residual, avoiding cancellation of almost equal squared norms. The unanchored identity's cancellation error is recorded as a diagnostic, not used for reported sampling outputs. Independent full-state subset reconstruction and two literal 32-step four-map replays verify the anchored computation.\n\n## Stable real logarithms and cumulants\n\nEach full L_i is similar by mobility scaling to a real symmetric block map. Its active eigenvalues are (a_M+a_U)/2 +/- sqrt((a_M-a_U)^2/4+eta_M eta_U sigma(C_i)^2), with scalar damping eigenvalues on null sectors. All 288 sequence/reference maps x 15 arms are checked for positive eigenvalues, nonsingularity and ||L-I||_2<1. Log(I+A) uses 12 convergent terms, with operator remainder <= ||A||^13/[13(1-||A||)]. The 2x2 singular blocks are independently checked against scipy logm/expm. Thus this is a controlled real matrix logarithm computation, not an unstable-log fallback.\n\nFull cumulants Omega1=sum_i K_i and Omega2=1/2 sum_{m>n}[K_m,K_n] act in the exact representation M=alpha M0+sum_i C_i^T R_i. The expert metric is the full cross-Gram matrix, not coefficient-space Euclidean norm. Condensed triangular cross-Gram formulas are verified against explicit dense commutator pairs. exp(Omega1) and exp(Omega1+Omega2) use 18 action Taylor terms with the explicit exponential remainder bound. Full spectral norm quantities are rigorous upper bounds, separately labeled; exact Frobenius/spectral norms are also supplied for P^T Omega P and sector-projected rows. Projected cumulants are computed after full log-map construction, not log(P^T L P).\n\nSecond-order residual explanation = 1-||r-c||^2/||r||^2, r=(ordered product - seqmean^32)x0 and c=[exp(Omega1+Omega2)-exp(Omega1)]x0. Negative values would mean non-explanatory corrections. The first-to-second improvement relative to the exact ordered product is separately reported. Neither quantity is a fraction of the 88.7564-fold reference failure or a fraction of observed-target closure error.\n"
(N / 'FINITE_TIME_MECHANISM_METHOD_NOTE.md').write_text(method)
P0 = headline['mean_shift']['primary']
C0 = headline['cumulants']['primary']['full_state']
ctrl = co[(co.population == 'primary') & (co.sector == 'full_state')].iloc[0]
summary = f"# Project 1 v286M finite-time mechanism\n\nA1 passed: primary reference/ordered 88.7564362071; sequence-mean/ordered 1.0248071872; reference/sequence-mean 86.4886740665; primary clustered 95% sequence-mean/ordered interval [1.0134040124,1.0271912740]; replication sequence-mean/ordered 1.0107399883. All 120 original reference/sequence-mean full-coordinate cases and ordered propagations were recomputed.\n\n## A2: location of the mean shift\n\nPrimary paired-case median ||Delta L||_F = {P0['mean_shift_frobenius']['median']:.10g}; ||Delta L||_2 = {P0['mean_shift_spectral']['median']:.10g}. Mean-shift action norm at the original initial state is {P0['action_full_norm']['median']:.10g}, entirely router-private because U0=0. Expert-private one-step action is exactly zero; this does not exclude later expert response through coupling. Median router displacement alignment is {P0['alignment_observed_router']['median']:.8g}; the archived 24D router projection captures {100 * P0['projected_action_capture_fraction']['median']:.5g}% of the action norm. Full-displacement cosine is {P0['alignment_observed_full']['median']:.8g}, diluted by the much larger expert displacement. Leading projected singular vectors/values and per-case/full operator norms are supplied.\n\n## A3/A4: sampling control limits stream-identity language\n\nPrimary same-stream four-reference case median closure error = {ctrl.same_stream_reference_case_median:.10g}; four-sequence-subset joint median = {ctrl.four_subset_joint_median:.10g}, 95% sampling band [{ctrl.four_subset_joint_q025:.10g},{ctrl.four_subset_joint_q975:.10g}]. Reference percentile = {ctrl.same_stream_reference_sampling_percentile:.6g}; fraction of individual cases outside their four-map control = {ctrl.individual_cases_outside_control_fraction:.6g}. The joint control survival flag is {bool(ctrl.survives_four_map_sampling_control)}. Sector/population results are all retained. No stream identity specificity is promoted from these means.\n\nThe deterministic m=1,2,4,8,16,32 grid reports per-case subset medians, draw bands and clustered aggregate intervals for full/expert/router errors. Finite-population m=32 variance vanishes by construction. The experiment supports realized-sample conditioning, while it cannot distinguish independent reference sampling noise from a changed operator population. It is not a forecasting experiment.\n\n## A5: ordering correction is small and explanatory\n\nAll real-log stability checks pass. Primary first-cumulant action median is {C0['first_cumulant_action_norm']['median']:.10g}; commutator action median is {C0['commutator_action_norm']['median']:.10g}; exact ordered-versus-sequence-mean residual median is {C0['ordered_vs_seqmean_residual_norm']['median']:.10g}. The second-order term explains a paired-case median {100 * C0['second_order_fraction_residual_explained']['median']:.7g}% of that small residual (clustered 95% [{100 * C0['second_order_fraction_residual_explained']['ci_low']:.7g}%,{100 * C0['second_order_fraction_residual_explained']['ci_high']:.7g}%]). It reduces first-cumulant-versus-exact-product squared action error by {100 * C0['second_cumulant_fraction_first_error_reduced']['median']:.7g}%.\n\nThis validates a subleading commutator interpretation over the tested 32 steps. It does not explain the dominant reference-to-realized-mean mismatch as noncommutativity, nor attribute any percentage of the 88.7564-fold failure to ordering. Exact full-space ordered products remain the comparator; physical target closure and product-approximation residuals are distinct.\n\nFull spectral cumulant norms are bounded, labeled as upper bounds; exact declared projected norms accompany all sector decompositions. No unstable logarithm, forced positive interpretation, retraining or frozen-criterion change was used.\n\n## Delivery and integrity\n\nFour required case/aggregate CSVs, additional sampling and numerical verification tables, projected direction/cumulant matrices, PDF/PNG, method note and machine-readable summary are delivered here. Compact evidence pack retains all v285M conclusions, including the weak objective secondary eigendirection and unresolved Q3 utility boundary. See COMPACT_PACK_MANIFEST.md for source paths, hashes, omissions, exact archive receipt and verification. No v284M or v285M archived file was overwritten.\n"
(N / 'FINITE_TIME_MECHANISM_SUMMARY.md').write_text(summary)
rep = N / 'reproduce'
rep.mkdir(exist_ok=True)
for name in ['finite_time_v286.py', 'verify_algebra.py', 'verify_samples_and_logs.py', 'projected_cumulant_norms.py', 'prepare_compact.py', 'finalize_v286.py']:
    shutil.copyfile(W / name, rep / name)
for src in [_p1_path('path_1256d42d8823'), _p1_path('path_671bfbb48bd3')]:
    shutil.copyfile(src, rep / Path(src).name)
(rep / 'README.md').write_text(f"# Reproduce v286M\n\nUse the original Python 3.12 runtime at {_p1_path('path_54a9e138cf70')} with the additional scientific-library path recorded in each script. Place scripts in PROJECT/work and outputs in PROJECT/outputs. Absolute original source paths are in ANALYSIS_SOURCE_MANIFEST.csv. Scripts load factors and observed checkpoints only; no model or optimizer is loaded.\n\nExecution order: finite_time_v286.py (requires BASELINE_VERIFICATION.json), projected_cumulant_norms.py, verify_algebra.py, verify_samples_and_logs.py, prepare_compact.py, finalize_v286.py. Reverify the baseline first using the attached handoff values and archived paired-case/whole-stream bootstrap data. Marker/visual QA are external artifact workflow steps. The delivered source includes the anchored sampling norm fix. Original source files remain read-only. Scratch deterministic draw arrays are regenerated under work/ and are intentionally omitted from the compact pack.\n")
for p in N.iterdir():
    if p.is_file() and p.name not in ['SOURCE_SEAL_BEFORE.json', 'ANALYSIS_SOURCE_MANIFEST.csv', 'FINITE_TIME_MAP_LOG_STABILITY.csv', 'NUMERICAL_VERIFICATION.csv', 'USED_SOURCE_INTEGRITY_VERIFICATION.csv', 'OUTPUT_MANIFEST.csv']:
        shutil.copyfile(p, K / p.name)
shutil.copytree(rep, K / 'reproduce', dirs_exist_ok=True)
registry = []
for df, name in [(shiftag, 'FINITE_TIME_MEAN_SHIFT_AGGREGATES.csv'), (sampleag, 'FINITE_TIME_SAMPLE_SIZE.csv'), (streamag, 'FINITE_TIME_STREAM_MEAN_AGGREGATES.csv'), (ag, 'FINITE_TIME_CUMULANT_AGGREGATES.csv')]:
    for rr in df.to_dict('records'):
        scope = '/'.join((str(rr[t]) for t in ['population', 'sector', 'm', 'comparator'] if t in rr))
        registry.append(dict(branch='A_v286M', statistic=rr['statistic'], value=rr['median'], ci_low=rr['ci_low'], ci_high=rr['ci_high'], unit='see method and source CSV', scope=scope, status='NEW_DERIVED_V286M', result_file=name))
pd.DataFrame(registry).to_csv(N / 'FINITE_TIME_RESULTS_REGISTRY_APPEND.csv', index=False, float_format='%.17g')
shutil.copyfile(N / 'FINITE_TIME_RESULTS_REGISTRY_APPEND.csv', K / 'FINITE_TIME_RESULTS_REGISTRY_APPEND.csv')
regold = pd.read_csv(A / 'RESULTS_REGISTRY_APPEND.csv', float_precision='round_trip')
pd.concat([regold, pd.DataFrame(registry)], ignore_index=True).to_csv(K / 'RESULTS_REGISTRY_APPEND.csv', index=False, float_format='%.17g')
(K / 'MODE_CONVERSION_ALL_COORDINATES_COMPACT.csv').unlink(missing_ok=True)
sources = {r['source_path']: r for r in json.loads((W / 'compact_sources.json').read_text()) if not r['source_path'].endswith('TOMO_L15_5033_RECOVERED_OPERATORS.npz')}
for r in read('ANALYSIS_SOURCE_MANIFEST.csv').to_dict('records'):
    sources[r['source_path']] = {**r, 'publication_outputs': 'v286M finite-time tables/figure', 'role': r['role']}
for p in rep.iterdir():
    if p.is_file():
        sources[str(p)] = dict(source_path=str(p), sha256=sha(p), size_bytes=p.stat().st_size, publication_outputs='reproduce/', role='new analysis code or user-invoked protocol')
source_checks = []
for path, row in sources.items():
    h = sha(path)
    assert h == row['sha256'], path
    source_checks.append(dict(source_path=path, sha256=h, unchanged=True))
pd.DataFrame(sources.values()).to_csv(K / 'SOURCE_MANIFEST.csv', index=False)
pd.DataFrame(source_checks).to_csv(N / 'USED_SOURCE_INTEGRITY_VERIFICATION.csv', index=False)
omitted = json.loads((W / 'compact_omitted.json').read_text())
for r in read('ANALYSIS_SOURCE_MANIFEST.csv').to_dict('records'):
    if 'FINITE_TIME_RECOVERED_MAP_FACTORS' in r['source_path']:
        omitted.append(dict(path=r['source_path'], sha256=r['sha256'], size_bytes=r['size_bytes'], reason='raw recovered step factor NPZ, excluded by handoff'))
zipold = A.parent / 'Project1_v285M_ArchiveCompletion.zip'
oldreceipt = json.loads((A.parent / 'Project1_v285M_ArchiveCompletion_VERIFICATION.json').read_text())
assert sha(zipold) == oldreceipt['sha256']
omitted.append(dict(path=str(zipold), sha256=oldreceipt['sha256'], size_bytes=zipold.stat().st_size, reason='full 2.5-GiB archive; compact publication evidence substituted'))
omitted = list({r['path']: r for r in omitted}.values())
pd.DataFrame(omitted).to_csv(K / 'OMITTED_LARGE_SOURCES.csv', index=False)
raw = [r for r in omitted if 'FINITE_TIME_RECOVERED_MAP_FACTORS' in r['path']]
assert len(raw) == 256
rawseal = ''.join((r['sha256'] + '  ' + Path(r['path']).name + '\n' for r in sorted(raw, key=lambda z: z['path'])))
(K / 'OMITTED_STEP_FACTOR_HASHES.sha256').write_text(rawseal)
inventorysha = hashlib.sha256(rawseal.encode()).hexdigest()
manifest = f"# Project 1 v286M compact publication evidence pack\n\nNo training, new model evaluation, source overwrite or frozen-criterion change. Source v285M archive completion plus finite-time mechanism A1-A6. The new finite-time interpretation supersedes obsolete ordering-dominance wording. This is an evidence pack, not a newly drafted manuscript.\n\n## Included evidence\n\n- Core: original RESULTS_SUMMARY.md and RESULTS_REGISTRY_APPEND.csv; filtered exact SOURCE_MANIFEST.csv; all five v285M method notes plus new finite-time and Q3 methods.\n- Finite time: three-way summary and per-sector statistics, small paired-case baseline, all required v286M CSVs and summaries, sampling controls, sector/projected norms and directions, final PDF and compact preview.\n- Mode conversion: signed 1,944-row summary (also compact all-coordinate robustness), fixed coordinate-0 main-figure CSV, signed analytic/measured/residual/background data, main PDF/preview and all-coordinate supplemental PDFs.\n- R63: coordinate closure summary and redraw CSV. Full 10,752-row table retained because its DEFLATE-compressed size is small: {len(__import__('zlib').compress((A / 'R63_COORDINATE_CLOSURE.csv').read_bytes(), 9)):,} bytes. Main figure PDF/preview and method note.\n- Tomography: all endpoint eigenvalues/gaps for L7/L15 and 41B/5033B; Q1/Q2/Q3 and all geometry intervals; second-vector stability; compact bootstrap quantiles plus original 2,000 replicate table; method and final figure. Leading eigenvector and top-2 first principal angle are distinct. Objective second direction remains weakly identifiable.\n- Local/objective: all 72 endpoint ratios, bootstrap intervals, signed-g/coherence summaries, fixed tensor-coordinate convention note, final factor figure and source table. No coordinate-invariant causal attribution.\n- Full tangent / Q3: existing cell diagnostics, audit and numerical summaries; exact Q3 protocol fragment. Q3 utility remains NULL_OR_UNRESOLVED. This utility experiment is distinct from tomography's rank-three Q3 overlap.\n- Reproduction: new no-training numerical scripts, exact invoked handoff and supporting scientific update; provenance and checks.\n\n## Omitted large sources\n\nAll 256 raw step-factor NPZ files are omitted, at `{A / 'FINITE_TIME_RECOVERED_MAP_FACTORS'}`. Every exact path/file size/SHA256 is recorded in OMITTED_LARGE_SOURCES.csv and OMITTED_STEP_FACTOR_HASHES.sha256. The canonical sorted step-factor hash-list SHA256 is `{inventorysha}` (a hash-list digest, not a directory byte hash).\n\nOriginal full archive: `{zipold}`; {zipold.stat().st_size:,} bytes; SHA256 `{oldreceipt['sha256']}`. Raw signed response/curve and document-level arrays or redundant tables are also omitted where summaries suffice; their exact paths/hashes are listed. Large model caches were not accessed by this analysis and are not packaged. No raw factor NPZ, large model cache, old intermediate diagnostic or duplicate all-coordinate summary is embedded.\n\n## Reading order and scientific boundaries\n\nRead FINITE_TIME_MECHANISM_SUMMARY.md and FINITE_TIME_MECHANISM_SUMMARY.json for new mechanisms; RESULTS_SUMMARY.md for accepted A-E evidence. Read the method notes before interpreting intervals. A four-map control cannot be replaced with a four-reference-versus-32-sequence comparison. The in-sequence subset study is a concentration diagnostic; it is not an independent-batch forecasting test. Full norm bounds and declared projected exact norms are separately labeled. Two-stream replication bootstrap intervals are descriptive.\n\n## Integrity\n\nPAYLOAD_MANIFEST.csv and SHA256SUMS.txt cover every payload file apart from circular/self-manifest records. ZIP readback, CRC and every member hash are verified. The exact final compressed size and archive SHA256 are reported by the companion external COMPACT_PACK_MANIFEST.md and ZIP_VERIFICATION.json. Archive source bytes and all publication dependencies are reverified after analysis.\n"
(K / 'COMPACT_PACK_MANIFEST.md').write_text(manifest)
seal = json.loads((N / 'SOURCE_SEAL_BEFORE.json').read_text())
assert all((sha(p) == h for p, h in seal.items()))
finalcheck = dict(status='PASS', sealed_v284M_v285M_files=len(seal), used_publication_sources=len(sources), source_changes=0, no_training=True, no_source_overwrite=True, baseline_full_coordinate_rows=len(vr), mean_shift_cases=len(sh), stream_mean_rows=len(s), sample_size_case_rows=len(ss), cumulant_sector_rows=len(x), map_arm_real_log_checks=len(a), raw_step_npz_in_compact_pack=0)
(N / 'FINAL_VERIFICATION.json').write_text(json.dumps(finalcheck, indent=2))
shutil.copyfile(N / 'FINAL_VERIFICATION.json', K / 'FINAL_VERIFICATION.json')
for p in rep.glob('*.py'):
    ast.parse(p.read_text())
files = sorted((p for p in K.rglob('*') if p.is_file() and p.name not in ['PAYLOAD_MANIFEST.csv', 'SHA256SUMS.txt']))
payload = [dict(relative_path=str(p.relative_to(K)), size_bytes=p.stat().st_size, sha256=sha(p)) for p in files]
pd.DataFrame(payload).to_csv(K / 'PAYLOAD_MANIFEST.csv', index=False)
(K / 'SHA256SUMS.txt').write_text(''.join((r['sha256'] + '  ' + r['relative_path'] + '\n' for r in payload)) + sha(K / 'PAYLOAD_MANIFEST.csv') + '  PAYLOAD_MANIFEST.csv\n')
zipout = O / 'Project1_v286M_CompactEvidencePack.zip'
with zipfile.ZipFile(zipout, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(K.rglob('*')):
        if not p.is_file():
            continue
        zi = zipfile.ZipInfo(str(p.relative_to(K)), date_time=(2026, 10, 12, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 33188 << 16
        z.writestr(zi, p.read_bytes(), compresslevel=9)
assert zipout.stat().st_size <= 100 * 1024 ** 2
with zipfile.ZipFile(zipout) as z:
    assert z.testzip() is None
    for r in payload:
        assert hashlib.sha256(z.read(r['relative_path'])).hexdigest() == r['sha256']
    assert hashlib.sha256(z.read('PAYLOAD_MANIFEST.csv')).hexdigest() == sha(K / 'PAYLOAD_MANIFEST.csv')
    assert all(('FINITE_TIME_RECOVERED_MAP_FACTORS/' not in n for n in z.namelist()))
    count = len(z.namelist())
hashzip = sha(zipout)
receipt = dict(status='PASS', archive=str(zipout), size_bytes=zipout.stat().st_size, size_mib=zipout.stat().st_size / 1024 ** 2, sha256=hashzip, members=count, payload_hashes_verified=len(payload), zip_crc_pass=True, source_integrity=finalcheck, omitted_step_factor_hash_list_sha256=inventorysha)
(O / 'ZIP_VERIFICATION.json').write_text(json.dumps(receipt, indent=2))
(O / 'Project1_v286M_CompactEvidencePack.zip.sha256').write_text(hashzip + '  Project1_v286M_CompactEvidencePack.zip\n')
(O / 'COMPACT_PACK_MANIFEST.md').write_text(manifest + f'\n## Final archive receipt\n\nCompressed size: **{zipout.stat().st_size:,} bytes ({zipout.stat().st_size / 1024 ** 2:.6f} MiB)**. Members: {count}. SHA256: `{hashzip}`. All CRC/member hash checks passed; all {len(seal)} archived files and {len(sources)} publication dependencies remain unchanged.\n')
payloadnew = [dict(relative_path=str(p.relative_to(N)), size_bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(N.rglob('*')) if p.is_file() and p.name != 'OUTPUT_MANIFEST.csv']
pd.DataFrame(payloadnew).to_csv(N / 'OUTPUT_MANIFEST.csv', index=False)
print(json.dumps(receipt, indent=2))
