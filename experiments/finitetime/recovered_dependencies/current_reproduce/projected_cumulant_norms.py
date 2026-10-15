# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Exact projection of full log-map cumulants; never logarithms of truncated maps."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys, json, math, ast
from pathlib import Path
R = Path(__file__).resolve().parents[1]
B = R.parents[1]
A = B / '2026-10-12/g-j/outputs/v285M_archive_completion'
P = B / '2026-10-09/g-ju/outputs/P2P_NV1_FRESH_P2P_COMPLETE'
H = B / '2026-10-08/jie/outputs'
O = R / 'outputs/v286M_finite_time_mechanism'
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, torch, pandas as pd
torch.set_num_threads(4)
DT = torch.float64
D = 576
T = 32
p = R / 'work/finite_time_v286.py'
nodes = [n for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ['c', 'ct', 'gram']]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'))
arms = json.loads((H / 'P2H_PREDICTIONS.json').read_text())['predictions']
with np.load(P / 'P2P_POOLED_SUBSPACE.npz') as z:
    QM = torch.from_numpy(z['Q_M'].copy()).double()
    QU = torch.from_numpy(z['Q_U'].copy()).double()
assert torch.allclose(QM.T @ QM, torch.eye(24, dtype=DT), atol=1e-12) and torch.allclose(QU.T @ QU, torch.eye(24, dtype=DT), atol=1e-12)
rows = []
mats = {}
for j in range(8):
    caps = []
    for n in range(1, 33):
        with np.load(A / f'FINITE_TIME_RECOVERED_MAP_FACTORS/stream{j}_step{n:02}.npz') as z:
            caps.append(tuple((torch.from_numpy(z[k].copy()).double() for k in ['h', 'phi', 'g'])))
    G = gram(caps)
    GF = G.reshape(T * D, T * D)
    for i in range(T):
        GF[i * D:(i + 1) * D, i * D:(i + 1) * D] = 0
        GF[i * D:(i + 1) * D, (i + 1) * D:] *= -1
    eig = []
    for h, p, g in caps:
        eig.append(torch.linalg.eigh(h.T @ (g @ g.T * (p @ p.T)) @ h / 16))
    CQ = torch.stack([torch.stack([c(cap, QM[:, i].reshape(D, 1536)) for i in range(24)], 1) for cap in caps])
    CM = torch.cat([CQ, torch.zeros(T, D, 24, dtype=DT)], 2)
    UU = torch.cat([torch.zeros(D, 24, dtype=DT), QU], 1)
    w = torch.arange(31, -33, -2, dtype=DT)[:, None, None]
    for k, ar in enumerate(arms):
        em, eu = (ar['eta_M'], ar['eta_U'])
        am = 1 - em * ar['mu']
        au = 1 - eu * ar['lambda_']
        da, db = (am - 1, au - 1)
        km, ku = (math.log(am), math.log(au))
        Bs = []
        Ds = []
        Es = []
        Fs = []
        for ev, V in eig:
            sv = ev.clamp_min(0).sqrt()
            bn = torch.zeros(D, dtype=DT)
            dn = torch.full((D,), em, dtype=DT)
            en = torch.full((D,), eu, dtype=DT)
            fn = torch.full((D,), db, dtype=DT)
            bb = torch.zeros(D, dtype=DT)
            dd = bb.clone()
            ee = bb.clone()
            ff = bb.clone()
            for n in range(1, 13):
                sign = (-1) ** (n + 1) / n
                bb += sign * bn
                dd += sign * dn
                ee += sign * en
                ff += sign * fn
                bn, dn, en, fn = (da * bn + em * en, da * dn + em * fn, eu * (da ** n + sv * sv * bn) + db * en, eu * sv * sv * dn + db * fn)
            Bs.append(V * bb @ V.T)
            Ds.append(V * dd @ V.T)
            Es.append(V * ee @ V.T)
            Fs.append(V * (ff - ku) @ V.T)
        BB, DD, EE, FF = map(torch.stack, [Bs, Ds, Es, Fs])
        ti = torch.bmm(BB, CM) + DD @ UU
        vi = torch.bmm(EE, CM) + FF @ UU
        one = torch.cat([torch.einsum('ndi,ndj->ij', CQ, ti), QU.T @ vi.sum(0)], 0)
        one[:24, :24] += 32 * km * torch.eye(24, dtype=DT)
        one[24:, 24:] += 32 * ku * torch.eye(24, dtype=DT)
        zd = (GF @ ti.reshape(T * D, 48)).reshape(T, D, 48)
        vd = 2 * vi.cumsum(0) - vi - vi.sum(0)
        rr = 0.5 * (w * (km - ku) * (DD @ UU) + torch.bmm(BB, zd) + torch.bmm(DD, vd))
        uu = 0.5 * (w * (ku - km) * torch.bmm(EE, CM) + torch.bmm(EE, zd) + torch.bmm(FF, vd)).sum(0)
        two = torch.cat([torch.einsum('ndi,ndj->ij', CQ, rr), QU.T @ uu], 0)
        mats[f'stream{j}_arm{k}_Omega1'] = one.numpy()
        mats[f'stream{j}_arm{k}_Omega2'] = two.numpy()
        for sector, sel in [('full_state', slice(None)), ('expert_private', slice(0, 24)), ('router_private', slice(24, 48))]:
            rows.append(dict(population='primary' if j < 6 else 'replication', stream=j, arm=k, sector=sector, first_cumulant_projected_frobenius=float(torch.linalg.vector_norm(one[sel])), first_cumulant_projected_spectral=float(torch.linalg.svdvals(one[sel])[0]), commutator_projected_frobenius=float(torch.linalg.vector_norm(two[sel])), commutator_projected_spectral=float(torch.linalg.svdvals(two[sel])[0])))
    print('projected cumulant norms stream', j, 'complete', flush=True)
    pd.DataFrame(rows).to_csv(O / 'FINITE_TIME_CUMULANT_PROJECTED_NORMS.csv', index=False, float_format='%.17g')
    np.savez_compressed(O / 'FINITE_TIME_PROJECTED_CUMULANTS.npz', **mats)
print('ALL PROJECTED NORMS COMPLETE', flush=True)
