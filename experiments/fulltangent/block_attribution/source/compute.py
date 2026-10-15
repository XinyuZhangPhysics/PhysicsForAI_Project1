# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import sys, json, csv, hashlib, datetime, math
import numpy as np, torch
BASE = Path.cwd()
SEALED = BASE / 'outputs/v283M_r1'
OUT = BASE / 'outputs/block_attribution_post_v283M_r1'
sys.path.insert(0, str(SEALED / 'source'))
from common import DIM, state, project, center_child, coords
from kernels import prepared, kernel

def read(p):
    return json.loads(Path(p).read_text())

def write(n, x):
    (OUT / n).write_text(json.dumps(x, indent=2, allow_nan=False, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def csvwrite(n, rows):
    with (OUT / n).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def closure(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    dif = a - b
    absmax = float(np.max(np.abs(dif)))
    rel = float(np.linalg.norm(dif.ravel()) / max(np.linalg.norm(b.ravel()), 1e-300))
    symmetric = float(np.max(np.abs(dif) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)))
    return {'max_abs_error': absmax, 'aggregate_relative_error': rel, 'max_pointwise_relative_error': symmetric, 'pass': absmax <= PLAN['absolute_closure_tolerance'] and rel <= PLAN['aggregate_relative_closure_tolerance']}
assert read(OUT / 'RUN_STATUS_post_v283M.json')['status'] == 'SEALED_INPUTS_VERIFIED'
PLAN = read(OUT / 'DECOMPOSITION_NUMERICS_PLAN_post_v283M.json')
records = read(SEALED / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')['primary_records']
floors = read(SEALED / 'NUMERICAL_FLOORS_FREEZE_v283M.json')['cells']
familyinfo = read(SEALED / 'SELECTION_RESPONSE_PROVENANCE_v281M.json')['families']
with (SEALED / 'FULL_TANGENT_COORDINATES_v283M.csv').open() as f:
    primary = list(csv.DictReader(f))
with (SEALED / 'SECONDARY_DIRECTIONAL_COORDINATES_v283M.csv').open() as f:
    direction = list(csv.DictReader(f))
pby = {(r['family'], int(r['g']), int(r['seed']), int(r['coordinate_index'])): float(r['B_full']) for r in primary}
dby = {(r['family'], int(r['g']), int(r['seed']), int(r['coordinate_index']), int(r['probe'])): float(r['B_directional_full']) for r in direction}
(OUT / 'geometry').mkdir(exist_ok=True)
(OUT / 'pair_response').mkdir(exist_ok=True)
allcoords = []
closures = []
leaks = []
torch.set_num_threads(1)
for fam in ['S1', 'S2']:
    rec = next((r for r in records if r['family'] == fam))
    basecache = np.load(rec['bound_geometry_path'])
    src, tgt, group = (basecache['src'], basecache['tgt'], basecache['group'])
    p = prepared(basecache['h'], state(fam), fam)
    d, D = DIM[fam]
    probes = np.load(SEALED / f'{fam}_DIRECTIONAL_PROBES_v283M.npy')
    names = ['down.weight', 'up.weight', 'up.bias'] if fam == 'S1' else ['down_proj.weight', 'up_proj.weight', 'gate_proj.weight']
    h = p['h']
    dot = np.einsum('nd,nd->n', h[src], h[tgt])
    for g in [1, 2, 4, 8]:
        rs = [r for r in records if r['family'] == fam and r['g'] == g]
        m = D // g
        fl = next((z for z in floors if z['family'] == fam and z['g'] == g))
        fgeo = np.load(SEALED / 'full_geometry' / f'{fam}_g{g}.npz')
        for probeidx in [-1, 0, 1, 2, 3]:
            label = 'isotropic' if probeidx == -1 else f'directional_{probeidx}'
            probe = None if probeidx == -1 else probes[:, probeidx]
            Kfull, Kdown, Kinput = kernel(p, fam, g, src, tgt, probe)
            targetK = fgeo['full_child'] if probeidx == -1 else Kfull
            reproduction = closure(Kfull, targetK)
            norm = np.sum((g * p['V']) ** 2, axis=0) / d if probe is None else (probe @ (g * p['V'])) ** 2
            blocks = [Kdown, np.empty_like(Kdown), np.empty_like(Kdown)]
            for c in range(g):
                sl = slice(c * m, (c + 1) * m)
                for lo in range(0, len(src), 256):
                    hi = lo + 256
                    ix = src[lo:hi]
                    iy = tgt[lo:hi]
                    a = p['a'][ix, sl] * p['a'][iy, sl]
                    aterm = np.sum(a * norm[sl], axis=1) / m
                    if fam == 'S1':
                        blocks[1][lo:hi, c] = dot[lo:hi] * aterm
                        blocks[2][lo:hi, c] = aterm
                    else:
                        b = p['b'][ix, sl] * p['b'][iy, sl]
                        blocks[1][lo:hi, c] = dot[lo:hi] * np.sum(b * norm[sl], axis=1) / m
                        blocks[2][lo:hi, c] = dot[lo:hi] * aterm
            Ksum = blocks[0] + blocks[1] + blocks[2]
            kclose = closure(Ksum, targetK)
            path = OUT / 'geometry' / f'{fam}_g{g}_{label}.npz'
            np.savez_compressed(path, **{f'block_{i}': v for i, v in enumerate(blocks)}, full=targetK, source=src, target=tgt, group=group)
            gh = sha(path)
            projected = [center_child(v)[1] for v in blocks]
            fullproj = center_child(targetK)[1]
            centered_target = fullproj[1]
            Csum = sum((v[1] for v in projected))
            geometryclose = closure(Csum, centered_target)
            geometrypowers = []
            for block, v in zip(names, projected):
                power = {s: float(np.mean(np.sum(a * a, axis=1))) for s, a in zip(['A', 'C', 'M'], v)}
                ratio = {s: power[s] / power['C'] if power['C'] > 0 else None for s in ['A', 'M']}
                geometrypowers.append((power, ratio))
            Bcases = []
            Bfullcases = []
            for r in rs:
                seed = r['seed']
                qpath = SEALED / 'historical_reproduction' / f'{fam}_g{g}_seed{seed}.npz'
                qdata = np.load(qpath)
                qC = qdata['q_C']
                qhat = qdata['qhat']
                qproj = project(qhat, g)
                Br = qdata['B_restricted']
                Bs = [coords(qC, v[1], group) for v in projected]
                newfull = coords(qC, centered_target, group)
                saved = np.array([pby[fam, g, seed, i] if probeidx == -1 else dby[fam, g, seed, i, probeidx] for i in range(12)])
                bclose = closure(sum(Bs), saved)
                reproduceB = closure(newfull, saved)
                Bcases.append(np.stack(Bs, axis=0))
                Bfullcases.append(saved)
                pa = {'full_pair': np.sum(qC * centered_target, axis=1), 'group': group}
                pa.update({f'block_{i}_pair': np.sum(qC * v[1], axis=1) for i, v in enumerate(projected)})
                np.savez_compressed(OUT / 'pair_response' / f'{fam}_g{g}_{label}_seed{seed}.npz', **pa)
                for j, (name, v) in enumerate(zip(names, projected)):
                    BA = coords(qproj[0], v[0], group)
                    BM = coords(qproj[2], v[2], group)
                    BC = Bs[j]
                    power, ratio = geometrypowers[j]
                    Bdirect = coords(qhat, np.broadcast_to(blocks[j][:, None, :], (len(src), 4, g)).reshape(len(src), 4 * g) - np.broadcast_to(blocks[j][:, None, :], (len(src), 4, g)).reshape(len(src), 4 * g).mean(axis=1, keepdims=True), group)
                    leak = {'family': fam, 'g': g, 'probe': label, 'seed': seed, 'block': name, 'K_A': power['A'], 'K_C': power['C'], 'K_M': power['M'], 'rho_A': ratio['A'], 'rho_M': ratio['M'], 'B_A_max_abs': float(np.max(np.abs(BA))), 'B_M_max_abs': float(np.max(np.abs(BM))), 'C_projection_response_max_abs_error': float(np.max(np.abs(Bdirect - BC))), 'geometry_sector_pass': bool(power['A'] == 0 and power['C'] == 0 and (power['M'] == 0) if g == 1 else max(ratio.values()) <= 1e-10), 'response_sector_pass': bool(np.max(np.abs(BA)) <= fl['full_sign_floor'] and np.max(np.abs(BM)) <= fl['full_sign_floor'])}
                    leaks.append(leak)
                    for i in range(12):
                        allcoords.append({'family': fam, 'activation': 'GELU' if fam == 'S1' else 'SwiGLU', 'g': g, 'arm': r['arm'], 'seed': seed, 'probe': label, 'coordinate_index': i, 'domain_pair': '->'.join(familyinfo[fam]['ordered_pair_keys'][i]), 'block': name, 'B_block': float(BC[i]), 'B_full': float(saved[i]), 'B_restricted': float(Br[i]), 'B_A_block': float(BA[i]), 'B_M_block': float(BM[i]), 'B_direct_block': float(Bdirect[i]), 'full_sign_floor': fl['full_sign_floor'], 'restricted_sign_floor': fl['restricted_sign_floor'], 'kernel_cache_sha256': gh, 'q_C_cache_sha256': sha(qpath), 'router_checkpoint_sha256': r['router_checkpoint_sha256'], 'frozen_full_coordinate_source_sha256': sha(SEALED / ('FULL_TANGENT_COORDINATES_v283M.csv' if probeidx == -1 else 'SECONDARY_DIRECTIONAL_COORDINATES_v283M.csv'))})
                closures.append({'family': fam, 'g': g, 'probe': label, 'seed': seed, 'kind': 'response', 'sum_vs_sealed_full': bclose, 'recomputed_full_vs_sealed': reproduceB})
            closures.append({'family': fam, 'g': g, 'probe': label, 'kind': 'kernel', 'sum_vs_sealed_full': kclose, 'recomputed_full_vs_sealed': reproduction, 'projected_C_sum_vs_full': geometryclose})
            print(fam, g, label, 'K closure', kclose['max_abs_error'], kclose['aggregate_relative_error'], 'B closure', max((z['sum_vs_sealed_full']['max_abs_error'] for z in closures if z['family'] == fam and z['g'] == g and (z['probe'] == label) and (z['kind'] == 'response'))), flush=True)
csvwrite('FULL_TANGENT_BLOCK_ATTRIBUTION_COORDS_post_v283M.csv', allcoords)
csvwrite('BLOCK_SECTOR_LEAKAGE_post_v283M.csv', leaks)
write('DECOMPOSITION_CLOSURE_post_v283M.json', {'tolerance_plan': PLAN, 'records': closures, 'strict_all_pass': all((c['sum_vs_sealed_full']['pass'] and c['recomputed_full_vs_sealed']['pass'] for c in closures)), 'max_kernel_abs_error': max((c['sum_vs_sealed_full']['max_abs_error'] for c in closures if c['kind'] == 'kernel')), 'max_response_abs_error': max((c['sum_vs_sealed_full']['max_abs_error'] for c in closures if c['kind'] == 'response')), 'max_kernel_aggregate_relative_error': max((c['sum_vs_sealed_full']['aggregate_relative_error'] for c in closures if c['kind'] == 'kernel')), 'max_response_aggregate_relative_error': max((c['sum_vs_sealed_full']['aggregate_relative_error'] for c in closures if c['kind'] == 'response'))})
write('RUN_STATUS_post_v283M.json', {'status': 'BLOCK_COORDINATES_COMPUTED', 'strict_closure_pass': all((c['sum_vs_sealed_full']['pass'] and c['recomputed_full_vs_sealed']['pass'] for c in closures)), 'primary_label_unchanged': 'PARTIAL_CHANNEL_AGREEMENT', 'coordinate_rows': len(allcoords)})
print('BLOCK COMPUTATION COMPLETE', len(allcoords), flush=True)
