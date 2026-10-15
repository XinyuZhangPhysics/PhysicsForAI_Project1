# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import importlib.util, types
man = read(OUT / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')
assert man['all_required_hashes_pass']
pr = read(OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.json')
recs = man['primary_records']
refs = pr['evidence_sources']
checks = []

def check(name, yes, detail=None):
    checks.append({'name': name, 'pass': bool(yes), 'detail': detail})
    if not yes:
        write('STAGE123_CHECKS_v283M.json', checks)
        raise RuntimeError(name)

def addbind(p, expected, role):
    actual = sha(p)
    check(role + ' sha256', actual == expected)
    man['entries'].append({'path': str(p), 'expected_sha256': expected, 'actual_sha256': actual, 'bytes': Path(p).stat().st_size, 'role': role, 'pass': True})
S1 = ROOT / '04_prospective_experiments/EXP_S1/scientific_full/package/EXP_S1_RESULTS'
asset = S1 / 'preflight/ASSET_MANIFEST.json'
pkg = read(S1 / 'PACKAGE_MANIFEST.json')
expected = next((x['sha256'] for x in pkg['files'] if x['path'] == 'preflight/ASSET_MANIFEST.json'))
addbind(asset, expected, 's1_asset_manifest')
for fam, rev in [('S1', 'a39f36b100fe8a5377810d56c3f4789b9c53ac42'), ('S2', '28e66ca6931668447a3bac213f23d990ad3b0e2b')]:
    r = refs[fam.lower() + '_provenance']
    addbind(Path(r['path']), r['sha256'], fam + '_provenance')
    v = read(r['path'])
    check(fam + ' revision', v['model_revision'] == rev)
    cfg = read(MODELS[fam] / 'config.json')
    d, D = DIM[fam]
    check(fam + ' config dimensions', (cfg['hidden_size'], cfg['intermediate_size']) == (d, D))
    check(fam + ' activation', cfg['hidden_act'] == ('gelu' if fam == 'S1' else 'silu'))
    hashes = {k: v['sha256'] for k, v in read(asset)['model_files'].items()} if fam == 'S1' else v['model_asset_sha256']
    for n in ['config.json', 'model.safetensors']:
        addbind(MODELS[fam] / n, hashes[n], fam + '_model_' + n)
    bank = [json.loads(s) for s in Path(refs[fam.lower() + '_bank']['path']).read_text().splitlines()]
    pairs = [json.loads(s) for s in Path(refs[fam.lower() + '_pairs']['path']).read_text().splitlines()]
    keys = pr['families'][fam]['ordered_pair_keys']
    ids = {r['train_row_id']: i for i, r in enumerate(bank)}
    src = np.array([ids[r['source_train_row_id']] for r in pairs])
    tgt = np.array([ids[r['target_train_row_id']] for r in pairs])
    group = np.array([keys.index([r['source_domain'], r['target_domain']]) for r in pairs])
    check(fam + ' frozen pair membership', len(bank) == 2048 and len(pairs) == 12288 and (len(ids) == 2048) and all((np.sum(group == i) == 1024 for i in range(12))))
    for famrec in [r for r in recs if r['family'] == fam]:
        run = read(famrec['run_metadata_path'])
        g = famrec['g']
        c = np.load(famrec['bound_geometry_path'])
        w = load_file(famrec['bound_router_path'])['router.weight']
        check(f"{fam} {g} {famrec['seed']} dimensions/membership", c['h'].shape == (2048, d) and w.shape == (4 * g, d) and (run['g'] == g) and (run['seed'] == famrec['seed']) and (run['arm'] == famrec['arm']) and (run['H'] == 4 * g) and (run['m'] == D // g) and (run['checkpoints']['64000']['horizon_nonpadding_tokens'] == 64000) and all((np.array_equal(c[k], a) for k, a in [('src', src), ('tgt', tgt), ('group', group)])))
write('FULL_TANGENT_SOURCE_MANIFEST_v283M.json', man)
(OUT / 'FULL_TANGENT_SOURCE_HASHES_v283M.sha256').write_text(''.join((f"{x['actual_sha256']}  {x['path']}\n" for x in man['entries'])))
print('STAGE1 COMPLETE', len(man['entries']), flush=True)
bind = []
for fam in ['S1', 'S2']:
    source = refs[fam.lower() + '_construction']['path']
    spec = importlib.util.spec_from_file_location('construct_' + fam, source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = state(fam)
    d, D = DIM[fam]
    if fam == 'S1':
        dense = types.SimpleNamespace(dense_h_to_4h=torch.nn.Linear(d, D), dense_4h_to_h=torch.nn.Linear(D, d), act=torch.nn.GELU(approximate='none'))
        dense.dense_h_to_4h.weight.data.copy_(s['dense_h_to_4h.weight'])
        dense.dense_h_to_4h.bias.data.copy_(s['dense_h_to_4h.bias'])
        dense.dense_4h_to_h.weight.data.copy_(s['dense_4h_to_h.weight'])
        dense.dense_4h_to_h.bias.data.copy_(s['dense_4h_to_h.bias'])
        cls = mod.SoftRouterMLP
    else:
        dense = types.SimpleNamespace(gate_proj=torch.nn.Linear(d, D, bias=False), up_proj=torch.nn.Linear(d, D, bias=False), down_proj=torch.nn.Linear(D, d, bias=False), act_fn=F.silu)
        for n in ['gate_proj', 'up_proj', 'down_proj']:
            getattr(dense, n).weight.data.copy_(s[n + '.weight'])
        cls = mod.SoftRouterSwiGLU
    for g in [1, 2, 4, 8]:
        moe = cls(dense, g, torch.zeros(4, d))
        m = D // g
        entries = []
        for c in range(g):
            params = dict(moe.experts[c].named_parameters())
            for p in range(4):
                other = dict(moe.experts[p * g + c].named_parameters())
                check(f'{fam} g{g} child{c} parent{p} parameter equality', all((torch.equal(v, other[n]) and v.data_ptr() != other[n].data_ptr() if p > 0 else torch.equal(v, other[n]) for n, v in params.items())))
            for name, t in params.items():
                common_bias = fam == 'S1' and name == 'down.bias'
                srcnames = {'up.weight': 'dense_h_to_4h.weight', 'up.bias': 'dense_h_to_4h.bias', 'down.weight': 'dense_4h_to_h.weight', 'down.bias': 'dense_4h_to_h.bias'}
                srcn = srcnames[name] if fam == 'S1' else name
                rule = 'full tensor, public output term (experts[0].down.bias)' if common_bias else 'columns [c*m:(c+1)*m] multiplied by g' if 'down' in name else 'rows [c*m:(c+1)*m]'
                entries.append({'child': c, 'parameter_name': f'experts[p*g+{c}].{name}', 'tensor_source': srcn, 'slice_rule': rule, 'shape': list(t.shape), 'dtype': str(t.dtype), 'included': not common_bias, 'reason': 'S1 forward excludes bias from every child F.linear and adds experts[0].down.bias once as a common output parameter; other copied down.bias tensors are unused' if common_bias else 'independently stored and used child-specific parameter'})
        bind.append({'family': fam, 'g': g, 'd': d, 'D': D, 'm': m, 'parents_identical_verified': True, 'parameters': entries, 'excluded': ['router.weight', 'base attention/norm/other layers', 'shared common output bias (S1 only)'], 'source_sha256': refs[fam.lower() + '_construction']['sha256']})
write('EXPERT_BLOCK_BINDING_v283M.json', {'utc_before_science': stamp(), 'records': bind, 'S1_bias_rationale': 'Forward child branches call F.linear(...,e.down.weight,None); the public bias is added outside every child branch. It is excluded by the frozen shared/common-parameter rule.'})
(OUT / 'EXPERT_BLOCK_BINDING_v283M.md').write_text('# Declared child-expert block\n\nBound before any new full-tangent outcomes.\n\nS1: up.weight [m,512], up.bias [m], down.weight [512,m], with copied output columns scaled by g. The source calls child output linear with bias=None and adds experts[0].down.bias once outside the mixture; this common output bias and unused duplicate biases are excluded.\n\nS2: gate_proj.weight [m,576], up_proj.weight [m,576], down_proj.weight [576,m], output columns scaled by g. No biases exist.\n\nAll four parent copies have independent storage and identical tensors for each child. Router, norms and all unrelated/shared base parameters are excluded. Per-child slices and dtypes are in the JSON.\n')
write('IMPLEMENTATION_PLAN_FREEZE_v283M.json', {'utc': stamp(), 'subset': [{'family': f, 'g': g, 'pair_index': 0 if g == 2 else 11264, 'child': c} for f in ['S1', 'S2'] for g in [2, 8] for c in [0, g - 1]], 'reference': 'all output-coordinate scalar gradient inner products summed; no J or full NTK matrix stored', 'metric': 'I/m, isotropic output trace 1/d', 'analytic_dtype': 'float64', 'historical_device': 'mps float32 with CPU float64 aggregates', 'numeric_tolerance_semantics': 'absolute and aggregate relative errors separately reported; both limits enforced for mandatory aggregate cross-checks', 'bootstrap': {'generator': 'PCG64', 'seed': 2026101201, 'replicates': 20000, 'unit': '8 paired seeds, retain all g and coordinates', 'RMS_normalization': 'normalize separately inside each family x g in every resampled dataset'}, 'sign_floor': 'within-cell restricted floor=max(1e-10,max historical B reproduction discrepancy); full floor=max(1e-10,conservative cross-check B error bound)', 'epsilon_B_propagation': 'max absolute kernel cross-check discrepancy times maximum per-domain mean L1 norm of q_C; extra factor 4 for conservative extrapolation', 'secondary': 'shared sequential PCG64(2026101202) stream, S1 Gaussian [512,4] QR then S2 [576,4] QR, first four Q columns'})
print('STAGE2 COMPLETE', flush=True)
(OUT / 'historical_reproduction').mkdir(exist_ok=True)
compare = []
for fam in ['S1', 'S2']:
    s = state(fam)
    base = next((r for r in recs if r['family'] == fam))
    cache = np.load(base['bound_geometry_path'])
    h = cache['h']
    src = cache['src']
    tgt = cache['tgt']
    group = cache['group']
    phi = features(h, s, fam, 'mps', torch.float32).cpu().double().numpy()
    d, D = DIM[fam]
    for g in [1, 2, 4, 8]:
        rs = [r for r in recs if r['family'] == fam and r['g'] == g]
        old = np.load(rs[0]['bound_geometry_path'])
        m = D // g
        child = np.empty((len(src), g))
        for c in range(g):
            sl = slice(c * m, (c + 1) * m)
            for a in range(0, len(src), 1024):
                b = a + 1024
                child[a:b, c] = (phi[src[a:b], sl] * phi[tgt[a:b], sl]).mean(axis=1)
        raw, (ka, kc, km) = center_child(child)
        kerr = float(np.max(np.abs(kc - old['kappa_C'])))
        check(fam + f' g{g} historical feature geometry', np.allclose(kc, old['kappa_C'], rtol=5e-06, atol=1e-08), {'max_abs': kerr})
        j4 = np.ones((4, 4)) / 4
        pg = np.eye(g) - np.ones((g, g)) / g
        C = np.kron(j4, pg)
        check(fam + f' g{g} C projector', np.max(np.abs(raw @ C - kc)) <= 1e-10)
        for r in rs:
            w = load_file(r['bound_router_path'])['router.weight']
            prob = qc_from_h(h, w, g)
            u = prob[src] * prob[tgt]
            centered = u - u.mean(axis=1, keepdims=True)
            Pq = float(np.mean(np.square(centered).sum(axis=1) / (4 * g - 1)))
            qhat = centered / math.sqrt(Pq)
            _, qc, _ = project(qhat, g)
            B = coords(qc, kc, group)
            P = float(np.mean(B ** 2))
            hist = read(r['source_path'])
            B0 = np.array(hist['B_coordinates'])
            be = float(np.max(np.abs(B - B0)))
            pe = abs(P - hist['P_B'])
            br = float(np.linalg.norm(B - B0) / max(np.linalg.norm(B0), 1e-300))
            pre = pe / max(abs(hist['P_B']), 1e-300)
            exact = g == 1 and np.array_equal(B, np.zeros(12)) and (P == 0)
            ok = exact if g == 1 else be <= 1e-08 and br <= 5e-06 and (pe <= 1e-08) and (pre <= 5e-06)
            compare.append({'family': fam, 'g': g, 'seed': r['seed'], 'geometry_max_abs_error': kerr, 'B_max_abs_error': be, 'B_aggregate_relative_error': br, 'P_B_abs_error': pe, 'P_B_relative_error': pre, 'Pq_abs_error': abs(Pq - hist['Pq_normalizer']), 'Q_C_abs_error': abs(float(np.mean(np.sum(qc ** 2, axis=1))) - hist['Q_C']), 'pass': ok})
            np.savez_compressed(OUT / 'historical_reproduction' / f"{fam}_g{g}_seed{r['seed']}.npz", q_C=qc, qhat=qhat, B_restricted=B, kappa_restricted=kc, Pq=Pq, group=group, src=src, tgt=tgt)
            check(f"{fam} g{g} seed{r['seed']} restricted reproduction", ok, compare[-1])
        print('STAGE3', fam, g, 'max B error', max((x['B_max_abs_error'] for x in compare if x['family'] == fam and x['g'] == g)), flush=True)
csvwrite('HISTORICAL_REPRODUCTION_COMPARISON_v283M.csv', compare)
write('STAGE123_CHECKS_v283M.json', checks)
write('HISTORICAL_REPRODUCTION_v283M.json', {'pass': True, 'records': 64, 'comparison': compare, 'max_B_error': max((x['B_max_abs_error'] for x in compare)), 'max_P_B_error': max((x['P_B_abs_error'] for x in compare))})
write('RUN_STATUS.json', {'status': 'HISTORICAL_REPRODUCTION_PASSED', 'stage': 3, 'utc': stamp(), 'new_full_tangent_outcomes_computed': False})
print('STAGE3 ALL PASS', flush=True)
