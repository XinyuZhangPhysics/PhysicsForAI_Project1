# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import json, csv, hashlib, sys, math, datetime
from fractions import Fraction
import numpy as np, torch
from torch.nn import functional as F
BASE = Path.cwd()
SEALED = BASE / 'outputs/v283M_r1'
OUT = BASE / 'outputs/block_attribution_post_v283M_r1'
sys.path.insert(0, str(SEALED / 'source'))
from common import DIM, state
from kernels import prepared

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = []
crosschecks = []

def verify(name, ok, detail=None):
    checks.append({'name': name, 'pass': bool(ok), 'detail': detail})
    if not ok:
        raise AssertionError(name)
torch.set_num_threads(1)
records = read(SEALED / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')['primary_records']
subset = read(SEALED / 'IMPLEMENTATION_PLAN_FREEZE_v283M.json')['subset']
for item in subset:
    fam, g, c, idx = (item['family'], item['g'], item['child'], item['pair_index'])
    r = next((r for r in records if r['family'] == fam and r['g'] == g))
    cache = np.load(r['bound_geometry_path'])
    x = torch.from_numpy(cache['h'][cache['src'][idx]]).double()
    y = torch.from_numpy(cache['h'][cache['tgt'][idx]]).double()
    s = state(fam)
    d, D = DIM[fam]
    m = D // g
    sl = slice(c * m, (c + 1) * m)
    if fam == 'S1':
        up = s['dense_h_to_4h.weight'][sl].double().clone().requires_grad_()
        bias = s['dense_h_to_4h.bias'][sl].double().clone().requires_grad_()
        down = (s['dense_4h_to_h.weight'][:, sl].double() * g).clone().requires_grad_()
        params = (down, up, bias)
        names = ['down.weight', 'up.weight', 'up.bias']

        def f(h):
            return F.linear(F.gelu(F.linear(h, up, bias), approximate='none'), down)
    else:
        gate = s['gate_proj.weight'][sl].double().clone().requires_grad_()
        up = s['up_proj.weight'][sl].double().clone().requires_grad_()
        down = (s['down_proj.weight'][:, sl].double() * g).clone().requires_grad_()
        params = (down, up, gate)
        names = ['down_proj.weight', 'up_proj.weight', 'gate_proj.weight']

        def f(h):
            return F.linear(F.silu(F.linear(h, gate)) * F.linear(h, up), down)
    fx, fy = (f(x), f(y))
    probes = np.load(SEALED / f'{fam}_DIRECTIONAL_PROBES_v283M.npy')
    for probeidx in [-1, 0, 1, 2, 3]:
        label = 'isotropic' if probeidx == -1 else f'directional_{probeidx}'
        z = np.load(OUT / 'geometry' / f'{fam}_g{g}_{label}.npz')
        if probeidx == -1:
            terms = [[], [], []]
            for j in range(d):
                gx = torch.autograd.grad(fx[j], params, retain_graph=True)
                gy = torch.autograd.grad(fy[j], params, retain_graph=True)
                for k, (a, b) in enumerate(zip(gx, gy)):
                    terms[k].append(float((a * b).sum()))
            values = [math.fsum(t) / (d * m) for t in terms]
        else:
            v = torch.from_numpy(probes[:, probeidx])
            gx = torch.autograd.grad(v @ fx, params, retain_graph=True)
            gy = torch.autograd.grad(v @ fy, params, retain_graph=True)
            values = [float((a * b).sum()) / m for a, b in zip(gx, gy)]
        for j, (name, ref) in enumerate(zip(names, values)):
            actual = float(z[f'block_{j}'][idx, c])
            error = abs(actual - ref)
            relative = error / max(abs(ref), 1e-300)
            crosschecks.append({**item, 'probe': label, 'block': name, 'analytic': actual, 'scalar_gradient_reference': ref, 'absolute_error': error, 'relative_error': relative})
            verify(f'{fam} g{g} c{c} {label} {name} independent gradient', error <= 1e-12 and relative <= 1e-10)
    print('GRADIENT CHECK', fam, g, c, flush=True)
raw = list(csv.DictReader((OUT / 'FULL_TANGENT_BLOCK_ATTRIBUTION_COORDS_post_v283M.csv').open()))
by = {(r['family'], int(r['g']), r['probe'], int(r['seed']), int(r['coordinate_index']), r['block']): r for r in raw}
verify('complete block-coordinate ordering', len(raw) == len(by) == 11520)
for rec in records:
    fam, g, seed = (rec['family'], rec['g'], rec['seed'])
    names = ['down.weight', 'up.weight', 'up.bias'] if fam == 'S1' else ['down_proj.weight', 'up_proj.weight', 'gate_proj.weight']
    for probe in ['isotropic'] + [f'directional_{i}' for i in range(4)]:
        z = np.load(OUT / 'pair_response' / f'{fam}_g{g}_{probe}_seed{seed}.npz')
        group = z['group']
        for j, block in enumerate(names):
            B = np.array([z[f'block_{j}_pair'][group == i].mean() for i in range(12)])
            stored = np.array([float(by[fam, g, probe, seed, i, block]['B_block']) for i in range(12)])
            verify(f'{fam} g{g} seed{seed} {probe} {block} coordinate replay', np.array_equal(B, stored))
        if g == 1:
            verify(f'{fam} seed{seed} {probe} g1 exact zeros', all((float(by[fam, g, probe, seed, i, b]['B_block']) == 0 for i in range(12) for b in names)))
source = read(SEALED / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')
verify('external sources remain unchanged', all((sha(e['path']) == e['expected_sha256'] for e in source['entries'])))
freeze = read(SEALED / 'PRIMARY_CLASSIFICATION_FREEZE_v283M.json')
verify('primary classification/result unchanged', freeze['classification'] == 'PARTIAL_CHANNEL_AGREEMENT' and sha(SEALED / 'FULL_TANGENT_RESULTS_v283M.json') == freeze['results_sha256'])
verify('original archive hash unchanged', sha(BASE / 'outputs/PhysicsForAI_Project1_FullTangentBenchmark_v283M_r1.tar.gz') == read(OUT / 'INPUT_BINDING_MANIFEST_post_v283M.json')['archive_sha256'])
verify('primary preregistration unchanged', sha(SEALED / 'FULL_TANGENT_PREREGISTRATION_v283M.md') == '0b5b04b7aa2d29b6630095cad95ac91d37ce3fc8e4be609d2bc3b15802a132d7')
closure = read(OUT / 'DECOMPOSITION_CLOSURE_post_v283M.json')
verify('no primary or directional full replay discrepancies', all((c['recomputed_full_vs_sealed']['max_abs_error'] == 0 for c in closure['records'])))
verify('strict closure failures retained, not silently upgraded', not closure['strict_all_pass'] and read(OUT / 'FULL_TANGENT_BLOCK_ATTRIBUTION_RESULTS_post_v283M.json')['diagnostic_status'] == 'STRICT_CLOSURE_TOLERANCE_NOT_MET')
(OUT / 'INDEPENDENT_BLOCK_IMPLEMENTATION_CHECK_post_v283M.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pass': True, 'checks': crosschecks, 'block_kernel_reference_count': len(crosschecks), 'max_absolute_error': max((v['absolute_error'] for v in crosschecks)), 'max_relative_error': max((v['relative_error'] for v in crosschecks)), 'comparison_tolerances': 'original mandated float64 independent implementation limits: abs<=1e-12, rel<=1e-10; strict decomposition closure limits remain separate and failed'}, indent=2) + '\n')
(OUT / 'FINAL_ATTRIBUTION_VERIFICATION_post_v283M.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'implementation_source_coordinate_checks_pass': True, 'checks': checks, 'checks_passed': len(checks), 'strict_decomposition_closure_pass': False, 'primary_classification_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'scientific_validation_not_upgraded': True}, indent=2) + '\n')
print('INDEPENDENT CHECKS', len(checks), 'PASS; STRICT CLOSURE RETAINS FAILURE')
