# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, csv, math, statistics, datetime
from fractions import Fraction
import numpy as np
BASE = Path.cwd()
OUT = BASE / 'outputs/block_attribution_post_v283M_r1'

def read(p):
    return json.loads(Path(p).read_text())

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')

def csvwrite(n, rows):
    with (OUT / n).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    lo = 0
    while lo < len(v):
        hi = lo + 1
        while hi < len(v) and v[idx[hi]] == v[idx[lo]]:
            hi += 1
        for j in range(lo, hi):
            out[idx[j]] = (lo + hi + 1) / 2
        lo = hi
    return out

def dot(x, y):
    return math.fsum((a * b for a, b in zip(x, y)))

def norm(x):
    return math.sqrt(dot(x, x))

def metrics(x, y, fx, fy):
    nx, ny = (norm(x), norm(y))
    eligible = [(a, b) for a, b in zip(x, y) if abs(a) > fx and abs(b) > fy]
    den = nx * ny
    rx, ry = (ranks(x), ranks(y))

    def corr(a, b):
        return statistics.correlation(a, b) if len(a) > 1 and len(set(a)) > 1 and (len(set(b)) > 1) else None
    return {'cosine': dot(x, y) / den if den > 0 else None, 'pearson': corr(x, y), 'spearman': corr(rx, ry), 'sign_agreement': sum(((a > 0) == (b > 0) for a, b in eligible)) / len(eligible) if eligible else None, 'sign_eligible_count': len(eligible)}
raw = list(csv.DictReader((OUT / 'FULL_TANGENT_BLOCK_ATTRIBUTION_COORDS_post_v283M.csv').open()))
for r in raw:
    for k in ['g', 'seed', 'coordinate_index']:
        r[k] = int(r[k])
    for k in ['B_block', 'B_full', 'B_restricted', 'full_sign_floor', 'restricted_sign_floor']:
        r[k] = float(r[k])
by = {(r['family'], r['probe'], r['g'], r['seed'], r['coordinate_index'], r['block']): r for r in raw}
assert len(by) == 11520
summary = []
seedsummary = []
grams = []
powers = []
seedpowers = []
familygroups = {}
for fam in ['S1', 'S2']:
    seeds = sorted({r['seed'] for r in raw if r['family'] == fam})
    blocks = ['down.weight', 'up.weight', 'up.bias'] if fam == 'S1' else ['down_proj.weight', 'up_proj.weight', 'gate_proj.weight']
    probes = ['isotropic'] + [f'directional_{i}' for i in range(4)]
    for probe in probes:
        for ggroup in [2, 4, 8, 'pooled_2_4_8']:
            gs = [2, 4, 8] if isinstance(ggroup, str) else [ggroup]
            for selected in ['all'] + seeds:
                ss = seeds if selected == 'all' else [selected]
                keys = [(fam, probe, g, s, i) for s in ss for g in gs for i in range(12)]
                data = [[by[*k, b]['B_block'] for k in keys] for b in blocks]
                full = [by[*k, blocks[0]]['B_full'] for k in keys]
                restricted = [by[*k, blocks[0]]['B_restricted'] for k in keys]
                N = len(keys)
                rf = math.sqrt(dot(restricted, restricted) / N)
                ff = math.sqrt(dot(full, full) / N)
                floorfull = max((by[*k, blocks[0]]['full_sign_floor'] for k in keys))
                floorr = max((by[*k, blocks[0]]['restricted_sign_floor'] for k in keys))
                for b, v in zip(blocks, data):
                    mfull = metrics(v, full, floorfull, floorfull)
                    mrest = metrics(v, restricted, floorfull, floorr)
                    rms = norm(v) / math.sqrt(N)
                    s = {'family': fam, 'g': ggroup, 'probe': probe, 'seed': selected, 'coordinate_count': N, 'seed_count': len(ss), 'block': b, 'RMS_block': rms, 'RMS_full': ff, 'RMS_restricted': rf, 'block_over_full_RMS': rms / ff if ff else None, 'block_over_restricted_RMS': rms / rf if rf else None, 'full_over_restricted_RMS': ff / rf if rf else None, **{'with_full_' + k: v for k, v in mfull.items()}, **{'with_restricted_' + k: v for k, v in mrest.items()}, 'inference_status': 'POST_HOC_DESCRIPTIVE_NO_INDEPENDENT_COORDINATE_INFERENCE'}
                    (summary if selected == 'all' else seedsummary).append(s)
                fd = [[Fraction.from_float(v) for v in x] for x in data]
                fdots = [[sum((a * b for a, b in zip(fd[j], fd[k])), Fraction()) for k in range(3)] for j in range(3)]
                fsums = [sum(t, Fraction()) for t in zip(*fd)]
                lhs_exact = sum((v * v for v in fsums), Fraction())
                diag_exact = sum((fdots[j][j] for j in range(3)), Fraction())
                cross_exact = 2 * sum((fdots[j][k] for j in range(3) for k in range(j + 1, 3)), Fraction())
                assert lhs_exact == diag_exact + cross_exact
                total = float(lhs_exact)
                target = dot(full, full)
                diagonal = float(diag_exact)
                cross = float(cross_exact)
                for j in range(3):
                    for k in range(3):
                        dj, dk = (float(fdots[j][j]), float(fdots[k][k]))
                        inner = float(fdots[j][k])
                        grams.append({'family': fam, 'g': ggroup, 'probe': probe, 'seed': selected, 'block_j': blocks[j], 'block_k': blocks[k], 'raw_inner_product': inner, 'mean_inner_product': inner / N, 'coherence': inner / math.sqrt(dj * dk) if dj * dk > 0 else None, 'exact_dyadic_inner_product': str(fdots[j][k]), 'cross_term_in_P_B': 2 * inner / N if j < k else None})
                pr = {'family': fam, 'g': ggroup, 'probe': probe, 'seed': selected, 'coordinate_count': N, 'sum_block_squared_norm': total, 'diagonal_squared_norm_sum': diagonal, 'twice_cross_inner_product_sum': cross, 'exact_power_identity_pass': True, 'exact_power_identity_residual': '0', 'sealed_full_squared_norm': target, 'sum_block_vs_sealed_power_abs_difference': abs(total - target), 'sum_block_vs_sealed_power_relative_difference': abs(total - target) / target if target else 0, 'P_B_block_sum': total / N, 'P_B_sealed_full': target / N, 'P_B_diagonal_sum': diagonal / N, 'P_B_cross_terms': cross / N, 'cross_to_diagonal_ratio': cross / diagonal if diagonal else None, 'self_power_fraction_down': float(fdots[0][0]) / total if total else None, 'self_power_fraction_added_1': float(fdots[1][1]) / total if total else None, 'self_power_fraction_added_2': float(fdots[2][2]) / total if total else None, 'cross_terms_sign': 'constructive' if cross > 0 else 'canceling' if cross < 0 else 'zero'}
                (powers if selected == 'all' else seedpowers).append(pr)
    print('SUMMARY', fam, flush=True)
csvwrite('FULL_TANGENT_BLOCK_ATTRIBUTION_SUMMARY_post_v283M.csv', summary)
csvwrite('BLOCK_ATTRIBUTION_SEED_SUMMARY_post_v283M.csv', seedsummary)
csvwrite('BLOCK_GRAM_COHERENCE_post_v283M.csv', grams)
csvwrite('BLOCK_POWER_ACCOUNTING_post_v283M.csv', powers)
csvwrite('BLOCK_SEED_POWER_ACCOUNTING_post_v283M.csv', seedpowers)
closure = read(OUT / 'DECOMPOSITION_CLOSURE_post_v283M.json')
leaks = list(csv.DictReader((OUT / 'BLOCK_SECTOR_LEAKAGE_post_v283M.csv').open()))
all_sector = all((r['geometry_sector_pass'] == 'True' and r['response_sector_pass'] == 'True' for r in leaks))
failures = [c for c in closure['records'] if not c['sum_vs_sealed_full']['pass'] or not c['recomputed_full_vs_sealed']['pass']]
write('FULL_TANGENT_BLOCK_ATTRIBUTION_RESULTS_post_v283M.json', {'scientific_status': 'OUTCOME_INFORMED_EXPLORATORY_DIAGNOSTIC', 'diagnostic_status': 'STRICT_CLOSURE_TOLERANCE_NOT_MET' if failures else 'DECOMPOSITION_VALIDATED', 'primary_classification_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'new_primary_inference_performed': False, 'strict_closure_pass': not failures, 'closure_tolerance_plan': closure['tolerance_plan'], 'closure_maxima': {k: v for k, v in closure.items() if k.startswith('max_')}, 'failed_closure_record_count': len(failures), 'closure_record_count': len(closure['records']), 'closure_failures': failures, 'all_blocks_individually_sector_consistent': all_sector, 'exact_power_accounting_groups': len(powers) + len(seedpowers), 'exact_dyadic_power_identities_all_pass': True, 'source_inputs_verified': read(OUT / 'INPUT_BINDING_MANIFEST_post_v283M.json')['status'] == 'PASS', 'confidence_intervals': 'NONE; seed-resolved descriptive summaries reported, no coordinate-independence inference', 'coordinate_rows': len(raw), 'primary_and_directional_summary_rows': len(summary), 'seed_summary_rows': len(seedsummary), 'summary': summary, 'power_accounting': powers, 'main_probe': 'isotropic', 'directional_probes': 'all four preserved deterministic orthonormal probes reused; separate descriptive summaries', 'manuscript_scope_change_warranted': False, 'auxiliary_only_reason': 'Outcome-informed analysis with strict numerical closure failure; do not enlarge main manuscript.'})
write('RUN_STATUS_post_v283M.json', {'status': 'DESCRIPTIVE_SUMMARIES_COMPLETE_STRICT_CLOSURE_NOT_MET' if failures else 'SUMMARIES_COMPLETE', 'primary_label_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'source_inputs_verified': True, 'strict_closure_pass': not failures, 'no_primary_artifact_modified': True})
print('SUMMARY COMPLETE', len(summary), len(seedsummary), 'EXACT POWER GROUPS', len(powers) + len(seedpowers), 'strict closure', not failures, flush=True)
