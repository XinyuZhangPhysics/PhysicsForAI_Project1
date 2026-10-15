# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(R / 'work/mplconfig')
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, json
from scipy.linalg import logm, expm
rng = np.random.default_rng(286)
checks = []
for dM, dU in [(5, 3), (9, 4)]:
    C = rng.normal(size=(dU, dM)) * 0.03
    x = rng.normal(size=dM)
    em = 0.01
    eu = 0.04
    am = 0.998
    au = 0.993
    L = np.block([[am * np.eye(dM), em * C.T], [eu * C, au * np.eye(dU)]])
    p = np.zeros(17)
    q = p.copy()
    for n in range(32):
        pn = am * p + em * q
        qn = au * q
        qn[0] += eu * am ** n
        qn[1:] += eu * p[:-1]
        p, q = (pn, qn)
    v = C @ x
    M = am ** 32 * x
    U = np.zeros(dU)
    for n in range(16):
        M += p[n] * C.T @ v
        U += q[n] * v
        v = C @ C.T @ v
    exact = np.linalg.matrix_power(L, 32) @ np.r_[x, np.zeros(dU)]
    err = np.linalg.norm(np.r_[M, U] - exact) / np.linalg.norm(exact)
    assert err < 2e-14
    checks.append(dict(check='finite-degree mean polynomial vs dense L^32', relative_error=err))
N = 7
dm = 5
du = 3
em = 0.01
eu = 0.04
am = 0.998
au = 0.993
km = np.log(am)
ku = np.log(au)
Cs = rng.normal(size=(N, du, dm)) * 0.03
x = rng.normal(size=dm)
u = rng.normal(size=du)
R = rng.normal(size=(N, du)) * 0.01
G = np.einsum('iad,jbd->iajb', Cs, Cs).reshape(N * du, N * du)
cm = np.einsum('iad,d->ia', Cs, x) + (G @ R.reshape(-1)).reshape(N, du)
B = []
D = []
E = []
F = []
K = []
maxlog = 0
for C in Cs:
    L = np.block([[am * np.eye(dm), em * C.T], [eu * C, au * np.eye(du)]])
    KK = logm(L)
    K.append(KK)
    ev, V = np.linalg.eigh(C @ C.T)
    sv = np.sqrt(np.maximum(ev, 0))
    bn = np.zeros(du)
    dn = np.full(du, em)
    en = np.full(du, eu)
    fn = np.full(du, au - 1)
    bb = np.zeros(du)
    dd = bb.copy()
    ee = bb.copy()
    ff = bb.copy()
    for n in range(1, 13):
        sign = (-1) ** (n + 1) / n
        bb += sign * bn
        dd += sign * dn
        ee += sign * en
        ff += sign * fn
        bn, dn, en, fn = ((am - 1) * bn + em * en, (am - 1) * dn + em * fn, eu * ((am - 1) ** n + sv ** 2 * bn) + (au - 1) * en, eu * sv ** 2 * dn + (au - 1) * fn)
    B.append(V * bb @ V.T)
    D.append(V * dd @ V.T)
    E.append(V * ee @ V.T)
    F.append(V * (ff - ku) @ V.T)
    calc = np.block([[km * np.eye(dm) + C.T @ B[-1] @ C, C.T @ D[-1]], [E[-1] @ C, ku * np.eye(du) + F[-1]]])
    err = np.linalg.norm(calc - KK)
    maxlog = max(maxlog, err)
    assert err < 5e-14
B, D, E, F = map(np.array, [B, D, E, F])
ti = np.einsum('nij,nj->ni', B, cm) + np.einsum('nij,j->ni', D, u)
vi = np.einsum('nij,nj->ni', E, cm) + np.einsum('nij,j->ni', F, u)
GA = G.reshape(N, du, N, du) * np.sign(np.arange(N)[:, None] - np.arange(N)[None, :])[:, None, :, None]
zd = (GA.reshape(N * du, N * du) @ ti.reshape(-1)).reshape(N, du)
vd = 2 * vi.cumsum(0) - vi - vi.sum(0)
w = (N - 1 - 2 * np.arange(N))[:, None]
r = 0.5 * (w * (km - ku) * np.einsum('nij,j->ni', D, u) + np.einsum('nij,nj->ni', B, zd) + np.einsum('nij,nj->ni', D, vd))
vv = 0.5 * (w * (ku - km) * np.einsum('nij,nj->ni', E, cm) + np.einsum('nij,nj->ni', E, zd) + np.einsum('nij,nj->ni', F, vd)).sum(0)
expected = sum((K[m] @ K[n] - K[n] @ K[m] for m in range(N) for n in range(m)), np.zeros_like(K[0])) / 2 @ np.r_[x + np.einsum('iad,ia->d', Cs, R), u]
actual = np.r_[np.einsum('iad,ia->d', Cs, r), vv]
err = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
assert err < 2e-10
checks.append(dict(check='log series coefficients vs scipy dense logm', max_absolute_frobenius_error=maxlog))
checks.append(dict(check='condensed Omega2 vs explicit 21 ordered commutator pairs', relative_error=err))
(Path(__file__).resolve().parents[1] / 'outputs/v286M_finite_time_mechanism/ALGEBRA_VERIFICATION.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2))
print(json.dumps(checks))
