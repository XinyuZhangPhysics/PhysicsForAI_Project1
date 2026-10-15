# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import ast, csv, hashlib, json, math, statistics
from inspect import BASE, S1, S2, REG, read, get, hash_file
OUT = BASE / 'outputs'
refs = {}
checks = []

def ref(key, p, locator):
    refs[key] = {'path': str(p), 'locator': locator, 'sha256': hash_file(p)}
    return key

def check(name, ok, detail=None):
    checks.append({'check': name, 'pass': bool(ok), 'detail': detail})

def fact(value, definition, aggregation, stage, source_ids):
    return {'value': value, 'statistic_definition': definition, 'aggregation_rule': aggregation, 'status': stage, 'provenance': [refs[x] for x in source_ids]}
p1 = S1 / 'protocol/PhysicsForAI_Project1_EXP_S1_PREREGISTRATION_FROZEN_v1.md'
p2 = S2 / 'repository_full/outputs/reports/exp_s2/preregistration_freeze/PhysicsForAI_Project1_EXP_S2_PREREGISTRATION_FROZEN_v1.md'
ref('s1_protocol', p1, 'lines 191–259, 397–497, 501–626')
ref('s2_protocol', p2, 'lines 21–38, 69–94')
ref('s1_response', S1 / 'source/response_s1.py', 'lines 53–65, 67–109, 123–164')
ref('s2_response', S2 / 'repository_full/src/science_response.py', 'lines 62–74, 142–191, 210–267')
ref('s1_evaluator', S1 / 'source/evaluate_s1.py', 'lines 36–64, 71–153')
ref('s2_evaluator', S2 / 'repository_full/src/science_adjudicate.py', 'lines 153–238')
ref('s1_construction', S1 / 'source/s1_impl.py', 'lines 10–64')
ref('s2_construction', S2 / 'repository_full/src/s2_impl.py', 'lines 33–120')
ref('s1_representation_source', S1 / 'source/materialize_audit.py', 'lines 64–87')
ref('later_normalized', Path(_p1_path('path_fa9fd8233283')), 'lines 18–38, 52; locator/cross-check only; isotropic assertions not adopted')
ref('later_normalized_method', Path(_p1_path('path_8d6f660f09d5')), 'lines 3395–3445, 3477–3479; explicitly post-hoc')
ref('ledger', Path(_p1_path('path_a6053f7e53dc')), 'E28 lines 1831–1880; E29 lines 1884–1932; locator/cross-check only')
ref('s1_package', S1 / 'PACKAGE_MANIFEST.json', '/files/*/{path,sha256}')
ref('s2_package', S2 / 'EXP_S2_SCIENCE_MANIFEST.sha256', 'sha256 + path records relative to science_full')
package1 = {x['path']: x['sha256'] for x in get(S1 / 'PACKAGE_MANIFEST.json')['files']}
package2 = {line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0] for line in read(S2 / 'EXP_S2_SCIENCE_MANIFEST.sha256').splitlines() if line.strip()}

def assignments(p):
    tree = ast.parse(read(p))
    d = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    d[t.id] = node.value
    return d

class Uncopy(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == 'RESOLUTION_POWER':
            node.id = 'FLOOR'
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'copy' and (not node.args) and (not node.keywords):
            return node.func.value
        return node
a1 = assignments(S1 / 'source/response_s1.py')
a2 = assignments(S2 / 'repository_full/src/science_response.py')
check('Power floor alias values identical', ast.literal_eval(a1['RESOLUTION_POWER']) == ast.literal_eval(a2['FLOOR']) == 1e-20)
for name in ['raw', 'centered', 'Pq', 'qhat', 'q3', 'qc', 'Q', 'K', 'direct_pair', 'projected_pair', 'direct', 'projected', 'P', 'direct_P', 'denominator', 'theta', 'factor_error', 'resolved']:
    check('AST response equality: ' + name, ast.dump(Uncopy().visit(a1[name])) == ast.dump(Uncopy().visit(a2[name])))

class Rename(ast.NodeTransformer):

    def visit_Name(self, node):
        if node.id == 'activation':
            node.id = 'features'
        return node
for name in ['child', 'k', 'mean', 'kc', 'ka', 'km', 'K']:

    def within(p, fn):
        tree = ast.parse(read(p))
        f = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn))
        for n in ast.walk(f):
            if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == name for t in n.targets)):
                return n.value
    n1 = within(S1 / 'source/response_s1.py', 'prepare')
    n2 = within(S2 / 'repository_full/src/science_response.py', 'geometry')
    check('AST geometry equality: ' + name, ast.dump(Rename().visit(n1)) == ast.dump(n2))
rows = []
families = {}
g1_details = []
for family, root, source, protocol, eval_id, seed0 in [('S1', S1 / 'scientific', 's1_response', 's1_protocol', 's1_evaluator', 2026100101), ('S2', S2 / 'science_full', 's2_response', 's2_protocol', 's2_evaluator', 2026110201)]:
    for n in ['AUDIT_GEOMETRY', 'PRIMARY_REGRESSIONS', 'BOOTSTRAP', 'FORMAL_HYPOTHESES', 'FORMAL_LABEL', 'PRIMARY_FREEZE', 'AUDIT_PRIMARY_64K', 'CALIBRATION_FREEZE', 'AUDIT_OPENING_RECORD', 'PROVENANCE', 'TRAINING_INTEGRITY', 'CHECKPOINT_HASHES']:
        ref(f'{family.lower()}_{n.lower()}', root / (n + '.json'), '/' + ('runs/*' if n == 'AUDIT_PRIMARY_64K' else '*'))
    geom = get(root / 'AUDIT_GEOMETRY.json')
    reg = get(root / 'PRIMARY_REGRESSIONS.json')
    boot = get(root / 'BOOTSTRAP.json')
    hyp = get(root / 'FORMAL_HYPOTHESES.json')
    formal = get(root / 'FORMAL_LABEL.json')
    freeze = get(root / 'PRIMARY_FREEZE.json')
    primary = get(root / 'AUDIT_PRIMARY_64K.json')
    cal = get(root / 'CALIBRATION_FREEZE.json')
    opening = get(root / 'AUDIT_OPENING_RECORD.json')
    prov = get(root / 'PROVENANCE.json')
    integrity = get(root / 'TRAINING_INTEGRITY.json')
    seeds = list(range(seed0, seed0 + 8))
    prefix = family.lower()
    check(f'{family} primary freeze binds all six analysis outputs', all((hash_file(root / n) == h for n, h in freeze['files_sha256'].items())))
    check(f'{family} bootstrap indices hash matches frozen metadata', hash_file(root / 'BOOTSTRAP_INDICES.npy') == boot['indices_sha256'])
    check(f'{family} frozen H1–H4 and integrity pass', all((hyp[n] == 'PASS' for n in ['H1', 'H2', 'H3', 'H4', 'resolved_requirement'])) and formal['training_integrity'] == 'PASS' and (integrity['status'] == 'PASS'))
    check(f'{family} chronology calibration < opening < primary', cal['timestamp_asia_shanghai'] < opening['timestamp_asia_shanghai'] < freeze['timestamp_asia_shanghai'])
    if family == 'S1':
        check('S1 calibration/opening bind unchanged source', all((hash_file(S1 / 'source' / n) == h for n, h in cal['analysis_code_sha256'].items())) and cal['analysis_code_sha256'] == opening['analysis_code_sha256'])
        check('S1 provenance binds protocol', prov['protocol_document_sha256'][p1.name] == hash_file(p1))
        check('S1 primary freeze binds evaluator', freeze['evaluator_code_sha256'] == refs['s1_evaluator']['sha256'])
    else:
        ref('s2_pre_science', root / 'PRE_SCIENCE_IMPLEMENTATION_FREEZE.json', '/code_sha256; /timestamp_asia_shanghai; /frozen_preregistration_sha256')
        ref('s2_preregistration_freeze', p2.parent / 'EXP_S2_PREREGISTRATION_FREEZE.json', '/s1_source_sha256; /response; /frozen_preregistration_sha256')
        pre = get(root / 'PRE_SCIENCE_IMPLEMENTATION_FREEZE.json')
        pre_reg = get(p2.parent / 'EXP_S2_PREREGISTRATION_FREEZE.json')
        check('S2 pre-science/calibration/opening/primary code identical', pre['code_sha256'] == cal['code_sha256'] == opening['code_sha256'] == freeze['code_sha256'])
        check('S2 pre-science code precedes calibration', pre['timestamp_asia_shanghai'] < cal['timestamp_asia_shanghai'])
        check('S2 frozen source hashes match', all((hash_file(S2 / 'repository_full/src' / n if n != 'evaluate_s1.py' else S1 / 'source' / n) == h for n, h in freeze['code_sha256'].items())))
        check('S2 protocol hash agrees through preregistration/calibration/opening', pre_reg['frozen_preregistration_sha256'] == cal['frozen_preregistration_sha256'] == opening['frozen_preregistration_sha256'] == hash_file(p2))
        check('S2 inherited S1 response/protocol/evaluator match', pre_reg['s1_source_sha256']['response'] == refs['s1_response']['sha256'] and pre_reg['s1_source_sha256']['preregistration'] == refs['s1_protocol']['sha256'] and (pre_reg['s1_source_sha256']['evaluator'] == refs['s1_evaluator']['sha256']))
    check(f'{family} opening binds calibration freeze', opening['calibration_freeze_sha256'] == hash_file(root / 'CALIBRATION_FREEZE.json'))
    check(f'{family} provenance binds opening/primary/formal/integrity', all((prov[k] == hash_file(root / n) for k, n in [('audit_opening_record_sha256', 'AUDIT_OPENING_RECORD.json'), ('primary_freeze_sha256', 'PRIMARY_FREEZE.json'), ('formal_label_sha256', 'FORMAL_LABEL.json'), ('training_integrity_sha256', 'TRAINING_INTEGRITY.json')])))
    by = {(x['seed'], x['arm']): (i, x) for i, x in enumerate(primary['runs'])}
    bank = S1 / 'preflight/audit_bank_manifest.jsonl' if family == 'S1' else S2 / 'repository_full/outputs/reports/exp_s2/preflight/banks/future_audit_document_ids.jsonl'
    pair = S1 / 'preflight/audit_pair_manifest.jsonl' if family == 'S1' else S2 / 'repository_full/outputs/reports/exp_s2/preflight/banks/future_audit_pair_ids.jsonl'
    ref(prefix + '_bank', bank, 'JSONL document rows')
    ref(prefix + '_pairs', pair, 'JSONL source_domain/target_domain and endpoint rows')
    check(f'{family} endpoint manifest hashes', hash_file(bank) == geom['bank_manifest_sha256'] and hash_file(pair) == geom['pair_manifest_sha256'])
    pairs = [json.loads(x) for x in read(pair).splitlines()]
    counts = {tuple(k): 0 for k in geom['ordered_pair_keys']}
    for x in pairs:
        counts[x['source_domain'], x['target_domain']] += 1
    check(f'{family} 2048 documents; 12*1024 ordered pairs', len(read(bank).splitlines()) == 2048 and len(pairs) == 12288 and (list(counts.values()) == [1024] * 12))
    for arm, v in geom['arms'].items():
        cache = S1 / 'derived' / f'audit_geometry_{arm}.npz' if family == 'S1' else root / f'audit_geometry_{arm}.npz'
        check(f'{family} {arm} geometry cache bytes hash', hash_file(cache) == v['cache_sha256'])
    for g in [1, 2, 4, 8]:
        arm = 'S' + str(g)
        for seed in seeds:
            times = [0, 16000, 64000, 256000] if g == 1 else [64000]
            run_path = root / 'checkpoints' / f'{arm}_seed_{seed}' / 'RUN_RESULT.json'
            run = get(run_path)
            check(f'{family} run metadata frozen experts {arm} {seed}', run['status'] == 'PASS' and all((c['frozen_tensor_hashes'] == run['checkpoints']['0']['frozen_tensor_hashes'] for c in run['checkpoints'].values())))
            for t in times:
                path = root / 'audit_responses' / f'{arm}_seed_{seed}_t{t}.json'
                r = get(path)
                check(f'{family} run identity {arm} {seed} {t}', (r['g'], r['seed'], r['horizon_nonpadding_tokens'], r['membership']) == (g, seed, t, 'audit'))
                check(f'{family} stored projector/factorization diagnostics {arm} {seed} {t}', r['response_projector_max_abs_error'] <= 1e-08 and (r['factorization_abs_error'] is None or r['factorization_abs_error'] <= 1e-08) and (abs(r['direct_response_power'] - r['P_B']) <= 1e-08))
                check(f'{family} geometry binding {arm} {seed} {t}', r['K_C_0'] == geom['arms'][arm]['K_C_0'] and r['geometry_cache_sha256'] == geom['arms'][arm]['cache_sha256'])
                ckpt = root / 'checkpoints' / f'{arm}_seed_{seed}' / f'router_{t}.safetensors'
                ch = hash_file(ckpt)
                recorded = run['checkpoints'][str(t)]['router_checkpoint_sha256' if family == 'S1' else 'sha256']
                check(f'{family} checkpoint bytes and response binding {arm} {seed} {t}', ch == r['router_checkpoint_sha256'] == recorded)
                if family == 'S1':
                    expected = package1[str(path.relative_to(S1))]
                else:
                    expected = package2[str(path.relative_to(root))]
                check(f'{family} response package binding {arm} {seed} {t}', hash_file(path) == expected)
                if g == 1:
                    h1_name = family + '_H1_rows'
                    entry = next((x for x in primary[h1_name] if x['seed'] == seed and x['horizon'] == t))
                    check(f'{family} structural stored zero {seed} {t}', r['Q_C'] == r['K_C_0'] == r['P_B'] == 0 and r['Theta_R'] is None and (r['B_coordinates'] == [0.0] * 12) and (r['B_coordinate_powers'] == [0.0] * 12) and (entry['response_file_sha256'] == hash_file(path)) and entry['all_12_below_floor'])
                    g1_details.append({'family': family, 'seed': seed, 'horizon': t, 'path': str(path), 'sha256': hash_file(path), 'locator': '/B_coordinates; /B_coordinate_powers; /Q_C; /K_C_0; /P_B; /Theta_R', 'max_B_coordinate_power': entry['max_B_coordinate_power']})
                else:
                    idx, entry = by[seed, arm]
                    check(f'{family} primary stored values match response {arm} {seed}', all((entry[k] == r[k] for k in ['Q_C', 'K_C_0', 'P_B', 'Theta_R', 'resolved_positive'])) and entry['response_file_sha256'] == hash_file(path))
                    check(f'{family} resolved response under frozen rule {arm} {seed}', r['resolved_positive'] and all((math.isfinite(r[k]) and r[k] > 1e-20 for k in ['Q_C', 'K_C_0', 'P_B', 'Theta_R'])) and (r['Q_C'] * r['K_C_0'] > 1e-20))
                if t == 64000:
                    loc = f'/runs/{by[seed, arm][0]}' if g > 1 else f"/{family}_H1_rows/{next((i for i, x in enumerate(primary[family + '_H1_rows']) if x['seed'] == seed and x['horizon'] == t))}"
                    row = {'family': family, 'activation': 'GELU' if family == 'S1' else 'SwiGLU', 'g': g, 'arm': arm, 'seed': seed, 'horizon_nonpadding_tokens': t, 'Q_C': r['Q_C'], 'K_C_0': r['K_C_0'], 'P_B': r['P_B'], 'Theta_R': r['Theta_R'], 'resolved_positive': r['resolved_positive'], 'g1_structural_null': g == 1, 'P_B_over_K_C_0': None if g == 1 else r['P_B'] / r['K_C_0'], 'normalization_status': 'undefined_g1' if g == 1 else 'audit_table_arithmetic_from_stored_scalars', 'source_path': str(path), 'source_locator': '/B_coordinates; /B_coordinate_powers; /P_B; /K_C_0; /Theta_R; /Q_C', 'source_sha256': hash_file(path), 'aggregate_path': str(root / 'AUDIT_PRIMARY_64K.json'), 'aggregate_locator': loc, 'aggregate_sha256': hash_file(root / 'AUDIT_PRIMARY_64K.json'), 'geometry_cache_sha256': r['geometry_cache_sha256'], 'router_checkpoint_sha256': r['router_checkpoint_sha256'], 'run_metadata_path': str(run_path), 'run_metadata_sha256': hash_file(run_path), 'response_projector_max_abs_error': r['response_projector_max_abs_error'], 'factorization_abs_error': r['factorization_abs_error'], 'historical_formal_label': formal['label'], 'measurement_status': 'frozen_primary'}
                    for j, v in enumerate(r['B_coordinates']):
                        row[f'B_{j + 1:02}'] = v
                    for j, v in enumerate(r['B_coordinate_powers']):
                        row[f'B_power_{j + 1:02}'] = v
                    rows.append(row)
    summaries = []
    for g in [1, 2, 4, 8]:
        subset = [r for r in rows if r['family'] == family and r['g'] == g]
        stats = {}
        for k in ['P_B', 'Q_C', 'Theta_R', 'P_B_over_K_C_0']:
            vals = [r[k] for r in subset if r[k] is not None]
            stats[k] = {'arithmetic_mean': statistics.mean(vals), 'median': statistics.median(vals), 'min': min(vals), 'max': max(vals), 'geometric_mean': math.exp(statistics.mean((math.log(x) for x in vals))) if all((x > 0 for x in vals)) else 0.0} if vals else None
        summaries.append({'g': g, 'n': 8, 'resolved_positive_count': sum((x['resolved_positive'] for x in subset)), 'K_C_0': geom['arms']['S' + str(g)]['K_C_0'], 'summaries': stats, 'aggregation_rule': 'equal-weight 8 stored seed scalars within arm; arithmetic mean, median, min/max, geometric mean; no pooled fields; not a historical frozen arm summary', 'status': 'current audit descriptive table aggregation explicitly requested in task C', 'source_records': [{'path': x['source_path'], 'locator': '/P_B; /Q_C; /Theta_R; /K_C_0', 'sha256': x['source_sha256']} for x in subset]})
    slopes = {}
    for k in ['beta_Q', 'beta_Theta', 'beta_B']:
        ref(prefix + '_' + k, root / 'PRIMARY_REGRESSIONS.json', '/central_slopes/' + k)
        ref(prefix + '_' + k + '_ci', root / 'BOOTSTRAP.json', '/' + k + '/ci95; /' + k + '/valid_replicates; /replicates; /resampling_unit')
        slopes[k] = fact(reg['central_slopes'][k], {'beta_Q': 'log Q_C slope', 'beta_Theta': 'log Theta_R slope', 'beta_B': 'log P_B power slope (not B-coordinate amplitude slope)'}[k], 'seed fixed-effect OLS over g=2,4,8 positive resolved values; within-seed log centering', 'preregistered_frozen_primary', [prefix + '_' + k, eval_id, protocol])
        slopes[k]['ci95'] = fact(boot[k]['ci95'], '95% paired-seed percentile CI for ' + k, '20000 whole paired-seed triplets; eight seeds with replacement; seed 2026100199; linear 2.5/97.5% quantiles; all replicates valid', 'preregistered_frozen_primary', [prefix + '_' + k + '_ci', eval_id, protocol])
        check(f'{family} aggregate central slope/CI copies frozen analysis {k}', formal['central_slopes'][k] == reg['central_slopes'][k] and formal['ci95'][k] == boot[k]['ci95'] and (boot[k]['valid_replicates'] == 20000))
    ref(prefix + '_beta_K', root / 'PRIMARY_REGRESSIONS.json', '/descriptive_beta_K; /descriptive_K_C_0_by_arm')
    betaK = reg['descriptive_beta_K']
    betaB = reg['central_slopes']['beta_B']
    betaN = betaB - betaK
    ci = [x - betaK for x in boot['beta_B']['ci95']]
    check(f'{family} algebraic frozen slope factorization', abs(betaB - (reg['central_slopes']['beta_Q'] + betaK + reg['central_slopes']['beta_Theta'])) < 1e-12)
    normalized = {'statistic': fact('P_B/K_C^(0) = Q_C*Theta_R', 'geometry-normalized response power; undefined at g=1; not R_C and not Theta_R', 'division of each stored positive P_B by arm-fixed K_C^(0); geometry fixed across seeds', 'later_post_hoc_descriptive', [source, prefix + '_audit_geometry', 'later_normalized', 'later_normalized_method']), 'slope': fact(betaN, 'beta_normalized = beta_B - beta_K = beta_Q + beta_Theta', 'algebraic subtraction of frozen aggregate slopes; no refit, no new bootstrap', 'current audit arithmetic recovery of later descriptive result', [prefix + '_beta_B', prefix + '_beta_K', 'later_normalized']), 'ci95_shifted_from_frozen': fact(ci, 'CI(beta_normalized) = CI(beta_B) - beta_K', 'deterministic constant translation of stored endpoints; exact under fully resolved identical arm sets and seed-independent geometry; no resampling', 'post_hoc_descriptive_algebraic_translation', [prefix + '_beta_B_ci', prefix + '_beta_K', 'later_normalized', 'later_normalized_method']), 'independent_frozen_normalized_result_path': None, 'independent_frozen_normalized_ci': None, 'original_normalized_execution_script': None, 'preregistered_normalized_hypothesis': None, 'formal_frozen_normalized_decision': None, 'descriptive_ci_excludes_zero': ci[0] > 0, 'inference': 'S1 interval includes zero; no resolved positive normalized-growth claim' if family == 'S1' else 'S2 post-hoc translated interval has positive lower endpoint; descriptive, not a new frozen hypothesis PASS'}
    families[family] = {'model': 'Pythia-70M' if family == 'S1' else 'SmolLM2-135M', 'activation': 'GELU' if family == 'S1' else 'SwiGLU', 'input_hidden_dimension': 512 if family == 'S1' else 576, 'feature_intermediate_dimension': 2048 if family == 'S1' else 1536, 'layer_one_indexed': 4 if family == 'S1' else 20, 'paired_seeds': seeds, 'ordered_pair_keys': geom['ordered_pair_keys'], 'primary_horizon': 64000, 'arm_summaries': summaries, 'raw_slopes': slopes, 'geometry_slope': fact(betaK, 'OLS slope of log K_C^(0) against log g', 'g=2,4,8 deterministic arm geometry; no bootstrap significance test', 'preregistered_descriptor_frozen_not_PASS_criterion', [prefix + '_beta_K', protocol, eval_id]), 'geometry_share_of_raw_log_slope': fact(betaK / betaB, 'beta_K/beta_B; algebraic share of raw log slope, not causal variance fraction', 'divide two stored frozen aggregate slopes', 'current audit descriptive arithmetic', [prefix + '_beta_K', prefix + '_beta_B']), 'normalized_response': normalized, 'formal_label': fact(formal['label'], 'all H1,H2,H3,H4 and >=7/8 resolved seeds per nontrivial arm; training integrity PASS', 'frozen conjunction, primary horizon only', 'preregistered_frozen_primary', [prefix + '_formal_label', prefix + '_formal_hypotheses', eval_id, protocol]), 'g1': fact('STRUCTURAL_EXACT_ZERO with stored exact numerical zeros', 'Pi_1=0 -> C=0 and kappa_C=q_C=0 -> B_coordinate=P_B=0 for every router; Theta undefined', 'all 8 seeds * 4 horizons * 12 coordinate powers exactly 0.0; historical H1 finite <=1e-20', 'preregistered_structural_control_frozen_outputs', [protocol, source, prefix + '_audit_geometry', prefix + '_audit_primary_64k']), 'stored_primary_max_projector_error': max((x['response_projector_max_abs_error'] for x in rows if x['family'] == family)), 'stored_primary_max_factorization_error': max((x['factorization_abs_error'] or 0 for x in rows if x['family'] == family))}
for family, root, n in [('S1', S1 / 'scientific', 'EXP_S1_RESULTS.json'), ('S2', S2 / 'science_full', 'EXP_S2_RESULTS.json')]:
    d = get(root / n)
    ref(family.lower() + '_results', root / n, '/central_slopes; /ci95; /formal_label' if family == 'S1' else '/primary/slopes; /primary/ci95; /terminal_status')
    sl = d['central_slopes'] if family == 'S1' else d['primary']['slopes']
    ci = d['ci95'] if family == 'S1' else d['primary']['ci95']
    check(f'{family} final results agree', all((sl[k] == families[family]['raw_slopes'][k]['value'] and ci[k] == families[family]['raw_slopes'][k]['ci95']['value'] for k in sl)))
with (OUT / 'SELECTION_RESPONSE_PRIMARY_TABLE_v281M.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
search_attempts = [{'scope': str(S1) + ' and ' + str(S2), 'method': 'rg text/source/JSON/CSV for 0.4217, 0.5311, geometry-normalized, P_B/K, beta_B-beta_K', 'result': 'No independent normalized statistic or normalized CI found in frozen primary analysis. Source computes Q_C, K_C_0, P_B, Theta_R; bootstrap stores beta_Q, beta_Theta, beta_B only.'}, {'scope': '397 text candidate ResearchRecord/ResearchState/Summary/R6/R7 files under PhysicsForAIProject1 and Codex', 'method': 'static locator regex search; matches logged in work/normalized_search_hits.json', 'result': 'Found explicit post-hoc S2 constant-shift method in v119 record and both-family rounded values in Phase-I retained summary.'}, {'scope': 'Codex 2026-09-2* and 01_authority_and_protocols', 'method': 'targeted rg exact normalized slopes/statistic', 'result': 'Later research notes/blueprints only; no original normalized execution script located. Nonexistent guessed 07_manuscripts_and_figures root excluded and recorded as failed locator attempt.'}]
classification = 'RESTRICTED_RESPONSE_SELECTION_VALIDATED' if all((x['pass'] for x in checks)) else 'PARTIAL_WITH_CONFLICT'
definitions = {'feature_geometry': fact('phi_S1,a(h)=GELU(w_a^T h+b_a); phi_S2,a(h)=SiLU(w_gate,a^T h)*(w_up,a^T h). D=2048 or 1536, m=D/g, I_c={cm,...,(c+1)m-1}. k_n,c=(1/m) sum_(a in I_c) phi_a(h_source,n) phi_a(h_target,n). kappa_raw,n,p,c=k_n,c; kappa_C=C Pi_H kappa_raw; (kappa_C)_n,p,c=k_n,c-(1/g)sum_cprime k_n,cprime.', 'expert-standard child feature Gram, four identical parent copies; no output-projection factor, residuals or full tangent metric in this estimator', 'within-child feature mean before expert projection; fixed untouched-base endpoint representation; CPU float64 feature products after float32 feature evaluation', 'preregistered_frozen_implementation', ['s1_protocol', 's2_protocol', 's1_response', 's2_response']), 'K_C_0': fact('K_C^(0)=(1/N)sum_n ||kappa_C,n||^2=(4/N)sum_n sum_c (k_n,c-mean_c k_n,c)^2; N=12288', 'construction-fixed child geometry power', 'sum 4g expert coordinates then arithmetic pair mean; fixed per arm/bank, shared across seeds and horizons', 'preregistered_frozen', ['s1_protocol', 's2_protocol', 's1_response', 's2_response', 's1_audit_geometry', 's2_audit_geometry']), 'router_field_context': fact('u_n,i=p_i(h_source,n)*p_i(h_target,n); u_tilde=Pi_H u; Pq=(1/N)sum_n sum_i u_tilde_n,i^2/(H-1); qhat=u_tilde/sqrt(Pq); q_C=C qhat; C=J_4 tensor Pi_g.', 'context inherited from Q_C audit; no new Q_C definition or inference', 'Pq bank-wide scalar separately per run/arm/checkpoint/membership', 'preregistered_frozen', ['s1_response', 's2_response', 's2_protocol']), 'B_coordinates': fact('B_exp_ab=(1/1024)sum_(n in P_ab) q_C,n^T kappa_C,n = (1/1024)sum_(n in P_ab) qhat_n^T kappa_C,n, a!=b.', '12 signed ordered-domain feature-space contraction coordinates; vector of group means', 'sum expert coordinates within each pair, then mean 1024 pairs per ordered-domain group; no residual/loss weighting; no across-seed averaging in measurement', 'preregistered_frozen', ['s1_protocol', 's2_protocol', 's1_response', 's2_response', 's1_audit_geometry', 's2_audit_geometry']), 'P_B': fact('P_B=(1/12)sum_(a!=b) B_exp_ab^2', 'squared response power of 12 domain coordinates; not mean pairwise squared contraction, not Q_C', 'square group-mean contraction first, then equal-weight 12-coordinate mean; no square root', 'preregistered_frozen', ['s1_response', 's2_response', 's2_protocol']), 'Theta_R': fact('Theta_R=P_B/(Q_C*K_C^(0)) if Q_C*K_C^(0)>1e-20 and P_B>1e-20, else null.', 'alignment ratio; removes both routing-field and geometry power; differs from geometry-only normalized P_B/K_C^(0)', 'per-run scalar ratio; formal H2 uses log slope upper paired-seed 95% bound <0', 'preregistered_frozen', ['s1_protocol', 's2_protocol', 's1_response', 's2_response', 's1_evaluator', 's2_evaluator']), 'resolved_rule': fact('finite Q_C,K_C^(0),P_B each >1e-20; denominator >1e-20; finite nonnull Theta_R; regression also retains each statistic only if finite >1e-20', 'historical numerical resolution, not a new coordinatewise significance test', 'formal arm requires >=7/8 positive resolved seed powers; all 8/8 actually pass', 'preregistered_frozen', ['s1_response', 's2_response', 's1_evaluator', 's2_evaluator', 's2_protocol'])}
report = {'classification': classification, 'scientific_interpretation_zh': '两族均验证受限特征响应的对称选择，未证明完整响应或效用提升。', 'mode': 'READ_ONLY_STATIC_SCIENTIFIC_AUDIT', 'audit_date_asia_shanghai': '2026-10-11', 'definitions': definitions, 'estimator_comparison': 'MATHEMATICALLY_IDENTICAL_FUNCTIONAL conditional on each family feature map/bank; feature activation, input/feature dimensions, layer and banks differ', 'families': families, 'csv_schema': {'B_01_to_B_12': 'Exact stored /B_coordinates/0 through /B_coordinates/11 in family ordered_pair_keys order', 'B_power_01_to_B_power_12': 'Exact stored /B_coordinate_powers/0 through /B_coordinate_powers/11; not recomputed', 'P_B_over_K_C_0': 'Audit scalar table arithmetic; undefined for g=1', 'null_representation': 'JSON null; CSV empty field'}, 'primary_records': rows, 'g1_all_horizon_evidence': g1_details, 'evidence_sources': refs, 'checks': checks, 'conflicts': [], 'unbound_fields': {'independent_frozen_geometry_normalized_analysis': None, 'original_post_hoc_normalized_execution_script': None, 'preregistered_geometry_normalized_hypothesis': None, 'numerical_identity_B_exp_with_full_B_ts': None, 'numerical_identity_B_exp_with_separately_defined_B_downarrow': None, 'downstream_utility_improvement': None}, 'search_attempts': search_attempts, 'operations': {'historical_module_imports': False, 'historical_module_execution': False, 'training': False, 'checkpoint_forward': False, 'tensor_deserialization': False, 'tensor_statistic_recalculation': False, 'bootstrap_rerun': False, 'slope_refit': False, 'stored_scalar_table_summaries': True, 'stored_frozen_slope_subtraction_and_CI_endpoint_translation': True, 'binary_access': 'streaming SHA256 only, no tensor loading'}, 'authority_file_hashes': json.loads(REG.read_text())}
(OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print('checks', len(checks), 'failed', [x for x in checks if not x['pass']])
print('rows', len(rows), 'registry', len(report['authority_file_hashes']))
for family, d in families.items():
    print(family, 'normalized', d['normalized_response']['slope']['value'], d['normalized_response']['ci95_shifted_from_frozen']['value'], 'share', d['geometry_share_of_raw_log_slope']['value'])
    for s in d['arm_summaries']:
        print('arm', s['g'], 'P_B', s['summaries']['P_B'])
