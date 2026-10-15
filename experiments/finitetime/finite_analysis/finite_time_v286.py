# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""No-model, no-training v286M full-coordinate operator analysis."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v286M_finite_time_mechanism'
WORK = ROOT / 'work'
os.environ.update(MPLCONFIGDIR=str(WORK / 'mplconfig'), PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
sys.path.append(_p1_path('path_d4fe3b336f94'))
import numpy as np, torch, pandas as pd, json, hashlib, time, math
from scipy.linalg import logm, expm
BASE = ROOT.parents[1]
ARCH = BASE / '2026-10-12/g-j/outputs/v285M_archive_completion'
P = BASE / '2026-10-09/g-ju/outputs/P2P_NV1_FRESH_P2P_COMPLETE'
H = BASE / '2026-10-08/jie/outputs'
torch.set_num_threads(6)
torch.use_deterministic_algorithms(True)
DT = torch.float64
D = 576
DM = D * 1536
T = 32
START = time.time()
SOURCES = {}

def log(s):
    print(f'{time.time() - START:.1f}s {s}', flush=True)

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def use(p, role):
    p = Path(p)
    SOURCES[str(p)] = dict(source_path=str(p), sha256=sha(p), size_bytes=p.stat().st_size, role=role)
    return p
sealpaths = [p for p in ARCH.rglob('*') if p.is_file()] + [p for p in (BASE / '2026-10-12/jie-2/outputs/v284M_r1').rglob('*') if p.is_file()]
seal = {str(p): sha(p) for p in sealpaths}
(OUT / 'SOURCE_SEAL_BEFORE.json').write_text(json.dumps(seal, indent=2))
log(f'sealed {len(seal)} v284M/v285M files')
old = pd.read_csv(use(ARCH / 'FINITE_TIME_THREE_WAY_CASES.csv', 'sealed baseline per-case errors'), float_precision='round_trip')
with np.load(use(ARCH / 'FINITE_TIME_STREAM_BOOTSTRAP_DRAWS.npz', 'original whole-stream bootstrap')) as z:
    BOOT = {k: z[k].copy() for k in ['primary', 'replication']}
assert json.loads((OUT / 'BASELINE_VERIFICATION.json').read_text())['status'] == 'PASS'
arms = json.loads(use(H / 'P2H_PREDICTIONS.json', 'original 15 arm parameters').read_text())['predictions']
cp = {(r['seed'], r['arm'], r['update']): r for r in json.loads(use(H / 'P2H_CHECKPOINT_INDEX.json', 'original target index').read_text())['checkpoints']}
seeds = torch.stack(torch.load(use(H.parent / 'work/p2h_private_seeds.pt', 'original initial expert states'), weights_only=True)).double()
with np.load(use(P / 'P2P_POOLED_SUBSPACE.npz', 'declared 24 expert/24 router projection basis')) as z:
    QM = torch.from_numpy(z['Q_M'].copy()).double()
    QU = torch.from_numpy(z['Q_U'].copy()).double()
views = [v for v in json.loads(use(P / 'P2P_BATCH_VIEW_SEAL.json', 'fixed independent reference view identities').read_text())['views'] if v['group'] == 'primary']
assert len(views) == 32
refs = []
for v in views:
    x = torch.load(use(P / f"captures/fresh_P2P/capture_{v['name']}.pt", 'independent reference h/phi/g'), weights_only=True)
    refs.append(tuple((x[k].double() for k in ['h', 'phi', 'g'])))

def c(cap, m):
    h, p, g = cap
    return -0.25 * ((g * (p @ m.T)).sum(1) @ h)

def ct(cap, u):
    h, p, g = cap
    return -0.25 * g.T @ ((h @ u)[:, None] * p)

def gram(caps):
    N = len(caps)
    G = torch.empty(N, D, N, D, dtype=DT)
    for i, (h, p, g) in enumerate(caps):
        for k in range(i, N):
            hh, pp, gg = caps[k]
            v = h.T @ (g @ gg.T * (p @ pp.T)) @ hh / 16
            G[i, :, k, :] = v
            G[k, :, i, :] = v.T
    return G
log('reference cross-Gram')
GR = gram(refs)
GRmean = GR.mean((0, 2))
refproj = []
for cap in refs:
    refproj.append(torch.stack([c(cap, QM[:, i].reshape(D, 1536)) @ QU for i in range(24)], dim=1))
refproj = torch.stack(refproj)
refsv = torch.linalg.eigvalsh(GRmean).clamp_min(0).sqrt()
log('reference kernels ready')

def coefficients(a):
    em, eu = (a['eta_M'], a['eta_U'])
    am = 1 - em * a['mu']
    au = 1 - eu * a['lambda_']
    p = np.zeros(17)
    q = p.copy()
    for n in range(T):
        pn = am * p + em * q
        qn = au * q
        qn[0] += eu * am ** n
        qn[1:] += eu * p[:-1]
        p, q = (pn, qn)
    return (am, au, p, q)
CO = [coefficients(a) for a in arms]

def mean_errors(G, C0, CD, seednorm, dmnorm, dmdot, unorm, Utarget, anchor=None):
    if G.ndim == 2:
        G = G[None]
        C0 = C0[None]
        CD = CD[None]
    V = [C0]
    for k in range(15):
        V.append(torch.bmm(G, V[-1][:, :, None])[:, :, 0])
    V = torch.stack(V, 1)
    rs = []
    for k, a in enumerate(arms):
        am, au, p, q = CO[k]
        pv = torch.einsum('l,bld->bd', torch.from_numpy(p[:16]), V)
        uv = torch.einsum('l,bld->bd', torch.from_numpy(q[:16]), V)
        bnorm = (am ** 32 - 1) ** 2 * seednorm + dmnorm[k] ** 2 - 2 * (am ** 32 - 1) * dmdot[k]
        cb = (am ** 32 - 1) * C0 - CD[:, k, :]
        e2 = bnorm + 2 * (pv * cb).sum(1) + (pv * torch.bmm(G, pv[:, :, None])[:, :, 0]).sum(1)
        if anchor is not None:
            ps, baseerr2, ce, gcross, gseq = anchor
            crossvec = torch.einsum('bij,j->bi', gcross, ps[k])
            e2 = baseerr2[k] + (pv * torch.bmm(G, pv[:, :, None])[:, :, 0]).sum(1) + ps[k] @ (gseq @ ps[k]) - 2 * (pv * crossvec).sum(1) + 2 * (pv * ce[:, k]).sum(1) - 2 * (ps[k] * ce_seq[k]).sum()
        assert float(e2.min()) >= -1e-22, (k, e2.min())
        emerr = e2.clamp_min(0).sqrt()
        euerr = torch.linalg.vector_norm(uv - Utarget[k], dim=1)
        rs.append(torch.stack([torch.sqrt(emerr ** 2 + euerr ** 2) / math.hypot(dmnorm[k], unorm[k]), emerr / dmnorm[k], euerr / unorm[k]], 1))
    return torch.stack(rs, 1).numpy()

def aggregate(rows, group, fields):
    df = pd.DataFrame(rows)
    out = []
    for key, x in df.groupby(group, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        pop = x.population.iloc[0]
        dr = BOOT[pop]
        ns = 6 if pop == 'primary' else 2
        for f in fields:
            vals = x.sort_values(['stream', 'arm'])[f].to_numpy().reshape(ns, 15)
            bt = np.median(vals[dr].reshape(len(dr), -1), axis=1)
            lo, hi = np.quantile(bt, [0.025, 0.975])
            out.append({**dict(zip(group, key)), 'statistic': f, 'median': float(np.median(vals)), 'ci_low': lo, 'ci_high': hi, 'clusters': ns, 'arms_per_cluster': 15, 'bootstrap_draws': len(dr)})
    return pd.DataFrame(out)
shiftrows = []
streamrows = []
samplerows = []
cumrows = []
projarr = {}
valid = []
for j in range(8):
    log(f'stream {j} load sequence')
    caps = []
    for n in range(1, 33):
        with np.load(use(ARCH / f'FINITE_TIME_RECOVERED_MAP_FACTORS/stream{j}_step{n:02}.npz', 'realized sequence h/phi/g')) as z:
            caps.append(tuple((torch.from_numpy(z[k].copy()).double() for k in ['h', 'phi', 'g'])))
    GS = gram(caps)
    GF = GS.reshape(T * D, T * D)
    Gmean = GS.mean((0, 2))
    log(f'stream {j} cross-Gram ready')
    seed = seeds[j]
    sn = float(torch.linalg.vector_norm(seed))
    cs = torch.stack([c(cap, seed) for cap in caps])
    cr = torch.stack([c(cap, seed) for cap in refs])
    dm = []
    uu = []
    for k in range(15):
        r = cp[j, k, 32]
        with np.load(use(r['path'], 'observed update-32 target M/U')) as z:
            dm.append(torch.from_numpy(z['M'].copy()).double() - seed)
            uu.append(torch.from_numpy(z['U'].copy()).double())
    dm = torch.stack(dm)
    uu = torch.stack(uu)
    dn = torch.linalg.vector_norm(dm.reshape(15, -1), dim=1).numpy()
    un = torch.linalg.vector_norm(uu, dim=1).numpy()
    sd = (dm * seed).sum((1, 2)).numpy()
    cds = torch.stack([torch.stack([c(cap, m) for m in dm]) for cap in caps])
    cdr = torch.stack([torch.stack([c(cap, m) for m in dm]) for cap in refs])
    sm = mean_errors(Gmean, cs.mean(0), cds.mean(0), sn ** 2, dn, sd, un, uu)
    rf = mean_errors(GRmean, cr.mean(0), cdr.mean(0), sn ** 2, dn, sd, un, uu)
    pop = 'primary' if j < 6 else 'replication'
    seq_ps = []
    seq_res = []
    for kind, bank, cm, gmean, ee in [('seqmean', caps, cs.mean(0), Gmean, sm), ('refmean', refs, cr.mean(0), GRmean, rf)]:
        for k, a in enumerate(arms):
            am, au, p, q = CO[k]
            vv = cm.clone()
            rr = torch.zeros(D, dtype=DT)
            u = rr.clone()
            for n in range(16):
                rr += p[n] * vv
                u += q[n] * vv
                vv = gmean @ vv
            delta = (am ** 32 - 1) * seed + sum((ct(cap, rr) for cap in bank)) / 32
            eM = float(torch.linalg.vector_norm(delta - dm[k]))
            eU = float(torch.linalg.vector_norm(u - uu[k]))
            ex = np.array([math.hypot(eM, eU) / math.hypot(dn[k], un[k]), eM / dn[k], eU / un[k]])
            archived = old[(old.stream == j) & (old.arm == k)].set_index('sector')[kind + '_original_error'].reindex(['full_state', 'expert_private', 'router_private']).to_numpy()
            assert np.allclose(ex, archived, rtol=2e-08, atol=2e-12), (j, k, kind, ex, archived)
            if kind == 'seqmean':
                seq_ps.append(rr)
                seq_res.append(delta - dm[k])
            valid.append(dict(stream=j, arm=k, comparator=kind, max_relative_norm_identity_error=float(np.max(abs(ex - ee[0, k]) / ex)), archived_match=True))
    seq_ps = torch.stack(seq_ps)
    seq_res = torch.stack(seq_res)
    seq_err2 = (seq_res ** 2).sum((1, 2))
    ceseq = torch.stack([torch.stack([c(cap, m) for m in seq_res]) for cap in caps])
    ce_seq = ceseq.mean(0)
    ordered = []
    orderedR = []
    for k, a in enumerate(arms):
        em, eu = (a['eta_M'], a['eta_U'])
        am, au, _, _ = CO[k]
        m = seed.clone()
        u = torch.zeros(D, dtype=DT)
        r = torch.zeros(T, D, dtype=DT)
        for i, cap in enumerate(caps):
            m, u = (am * m + em * ct(cap, u), au * u + eu * c(cap, m))
        ordered.append((m - seed, u))
        rr = torch.zeros(T, D, dtype=DT)
        ur = torch.zeros(D, dtype=DT)
        for i in range(T):
            cm = am ** i * cs[i] + GF[i * D:(i + 1) * D] @ rr.reshape(-1)
            rr *= am
            rr[i] += em * ur
            ur = au * ur + eu * cm
        orderedR.append((am ** 32, rr, ur))
        assert torch.allclose(ur, u, rtol=1e-10, atol=1e-15)
        ex = math.hypot(float(torch.linalg.vector_norm(m - seed - dm[k])), float(torch.linalg.vector_norm(u - uu[k]))) / math.hypot(dn[k], un[k])
        ar = old[(old.stream == j) & (old.arm == k) & (old.sector == 'full_state')].sequence_original_error.iloc[0]
        assert np.isclose(ex, ar, rtol=1e-09, atol=1e-12)
    log(f'stream {j} all original comparators reproduced')
    idx = [i for i, v in enumerate(views) if v['stream'] == j]
    other = [i for i in range(32) if i not in idx]
    means = {'global_reference': (GRmean, cr.mean(0), cdr.mean(0)), 'same_stream_reference': (GR[idx][:, :, idx, :].mean((0, 2)), cr[idx].mean(0), cdr[idx].mean(0)), 'leave_one_stream_out_reference': (GR[other][:, :, other, :].mean((0, 2)), cr[other].mean(0), cdr[other].mean(0)), 'same_sequence': (Gmean, cs.mean(0), cds.mean(0))}
    me = {}
    for name, (g, c0, cd) in means.items():
        bank = refs if name == 'global_reference' else [refs[i] for i in idx] if name == 'same_stream_reference' else [refs[i] for i in other] if name == 'leave_one_stream_out_reference' else caps
        ee = []
        for k in range(15):
            am, au, p, q = CO[k]
            vv = c0.clone()
            rr = torch.zeros(D, dtype=DT)
            u = rr.clone()
            for n in range(16):
                rr += p[n] * vv
                u += q[n] * vv
                vv = g @ vv
            emerr = float(torch.linalg.vector_norm((am ** 32 - 1) * seed + sum((ct(cap, rr) for cap in bank)) / len(bank) - dm[k]))
            euerr = float(torch.linalg.vector_norm(u - uu[k]))
            ee.append([math.hypot(emerr, euerr) / math.hypot(dn[k], un[k]), emerr / dn[k], euerr / un[k]])
        me[name] = np.array(ee)
    controls = {}
    for m, N in [(1, 32), (2, 256), (4, 1024), (8, 256), (16, 256), (32, 1)]:
        rng = np.random.default_rng(np.random.SeedSequence([20261012, 286, j, m]))
        ix = np.arange(32)[:, None] if m == 1 else np.array([rng.choice(32, m, replace=False) for _ in range(N)])
        if m == 32:
            ix = np.arange(32)[None]
        allerrors = []
        for st in range(0, len(ix), 16):
            sel = ix[st:st + 16]
            gg = torch.stack([GS[x][:, :, x, :].mean((0, 2)) for x in sel])
            cc = torch.stack([cs[x].mean(0) for x in sel])
            cd = torch.stack([cds[x].mean(0) for x in sel])
            ce = torch.stack([ceseq[x].mean(0) for x in sel])
            gcross = torch.stack([GS[x].mean((0, 2)) for x in sel])
            allerrors.append(mean_errors(gg, cc, cd, sn ** 2, dn, sd, un, uu, (seq_ps, seq_err2, ce, gcross, Gmean)))
        errs = np.concatenate(allerrors)
        if m == 32:
            assert np.allclose(errs[0], me['same_sequence'], rtol=2e-05, atol=2e-10)
            errs[0] = me['same_sequence']
        controls[m] = errs
        np.savez_compressed(WORK / f'subsets_stream{j}_m{m}.npz', indices=ix, errors=errs)
        for k in range(15):
            for s, sector in enumerate(['full_state', 'expert_private', 'router_private']):
                lo, hi = np.quantile(errs[:, k, s], [0.025, 0.975])
                samplerows.append(dict(population=pop, stream=j, arm=k, sector=sector, m=m, subset_draws=len(ix), subset_median=float(np.median(errs[:, k, s])), subset_q025=lo, subset_q975=hi))
        log(f'stream {j} sample m={m} draws={len(ix)}')
    for k in range(15):
        for s, sector in enumerate(['full_state', 'expert_private', 'router_private']):
            noise = controls[4][:, k, s]
            ss = me['same_stream_reference'][k, s]
            pct = (np.count_nonzero(noise < ss) + 0.5 * np.count_nonzero(noise == ss)) / len(noise)
            for name, err in me.items():
                streamrows.append(dict(population=pop, stream=j, arm=k, sector=sector, comparator=name, closure_error=float(err[k, s]), four_subset_median=float(np.median(noise)), four_subset_q025=float(np.quantile(noise, 0.025)), four_subset_q975=float(np.quantile(noise, 0.975)), same_stream_reference_percentile=pct, same_stream_outside_95_sampling_control=bool(pct < 0.025 or pct > 0.975)))
    cross = torch.zeros(D, D, dtype=DT)
    for i, (h, p, g) in enumerate(caps):
        for hh, pp, gg in refs:
            cross += h.T @ (g @ gg.T * (p @ pp.T)) @ hh / (16 * 1024)
    GD = Gmean + GRmean - cross - cross.T
    GD = (GD + GD.T) / 2
    sv = torch.linalg.eigvalsh(GD).clamp_min(0).sqrt()
    cdelta = cs.mean(0) - cr.mean(0)
    seqproj = torch.stack([torch.stack([c(cap, QM[:, i].reshape(D, 1536)) @ QU for i in range(24)], 1) for cap in caps])
    dp = seqproj.mean(0) - refproj.mean(0)
    for k, a in enumerate(arms):
        em, eu = (a['eta_M'], a['eta_U'])
        shiftu = eu * cdelta
        vobs = uu[k]
        fullalign = float(shiftu @ vobs) / (float(torch.linalg.vector_norm(shiftu)) * math.hypot(dn[k], un[k]))
        routeralign = float(shiftu @ vobs) / (float(torch.linalg.vector_norm(shiftu)) * un[k])
        projectu = QU.T @ shiftu
        mat = torch.zeros(48, 48, dtype=DT)
        mat[:24, 24:] = em * dp.T
        mat[24:, :24] = eu * dp
        us, sigs, vs = torch.linalg.svd(mat)
        projarr[f'stream{j}_arm{k}_mean_shift'] = mat.numpy()
        projarr[f'stream{j}_arm{k}_left'] = us[:, :6].numpy()
        projarr[f'stream{j}_arm{k}_right'] = vs[:6].T.numpy()
        projarr[f'stream{j}_arm{k}_singular'] = sigs[:6].numpy()
        shiftrows.append(dict(population=pop, stream=j, arm=k, mean_shift_frobenius=math.sqrt(em ** 2 + eu ** 2) * float(torch.sqrt(torch.trace(GD).clamp_min(0))), mean_shift_spectral=max(em, eu) * float(sv[-1]), action_full_norm=float(torch.linalg.vector_norm(shiftu)), action_expert_norm=0.0, action_router_norm=float(torch.linalg.vector_norm(shiftu)), action_expert_projected_norm=0.0, action_router_projected_norm=float(torch.linalg.vector_norm(projectu)), alignment_observed_full=fullalign, alignment_observed_router=routeralign, projected_frobenius=float(torch.linalg.vector_norm(mat)), projected_top_singular=float(sigs[0]), projected_action_capture_fraction=float(torch.linalg.vector_norm(projectu) / torch.linalg.vector_norm(shiftu))))
    log(f'stream {j} mean-shift complete')

    def normrep(x):
        al, r, u = x
        flat = r.reshape(-1)
        val = al ** 2 * sn ** 2 + 2 * al * (r * cs).sum() + flat @ (GF @ flat) + u @ u
        return math.sqrt(max(float(val), 0.0))

    def add(x, y, fac=1.0):
        return (x[0] + fac * y[0], x[1] + fac * y[1], x[2] + fac * y[2])

    def scale(x, v):
        return (x[0] * v, x[1] * v, x[2] * v)

    def real(x):
        al, r, u = x
        return (al * seed + sum((ct(cap, r[i]) for i, cap in enumerate(caps))), u)
    eig = [torch.linalg.eigh(GS[i, :, i, :]) for i in range(T)]
    sigmax = max((float(e[0][-1].clamp_min(0).sqrt()) for e in eig))
    Ganti = GF.clone()
    for i in range(T):
        Ganti[i * D:(i + 1) * D, i * D:(i + 1) * D] = 0
        Ganti[i * D:(i + 1) * D, (i + 1) * D:] *= -1
    for k, a in enumerate(arms):
        em, eu = (a['eta_M'], a['eta_U'])
        am, au, _, _ = CO[k]
        da, db = (am - 1, au - 1)
        km, ku = (math.log(am), math.log(au))
        svals = [e[0].clamp_min(0).sqrt() for e in eig]
        eta = max(em / eu, eu / em) ** 0.5
        normA = max(float(torch.linalg.svdvals(torch.tensor([[da, em * sigmax], [eu * sigmax, db]], dtype=DT))[0]), abs(da), abs(db))
        mineig = min((float((am + au) / 2 - torch.sqrt(torch.tensor(((am - au) / 2) ** 2) + em * eu * s * s).max()) for s in svals))
        assert normA < 1 and mineig > 0, (j, k, normA, mineig)
        rem = normA ** 13 / (13 * (1 - normA))
        assert rem < 1e-18
        Bs = []
        Ds = []
        Es = []
        Fs = []
        lnnorm = 0.0
        projectKs = []
        for i, (ev, V) in enumerate(eig):
            s = ev.clamp_min(0).sqrt()
            AA = torch.zeros(D, 2, 2, dtype=DT)
            AA[:, 0, 0] = da
            AA[:, 1, 1] = db
            AA[:, 0, 1] = em * s
            AA[:, 1, 0] = eu * s
            power = AA.clone()
            kk = AA.clone()
            b = torch.zeros(D, dtype=DT)
            bn = torch.zeros(D, dtype=DT)
            dn0 = torch.full((D,), em, dtype=DT)
            en = torch.full((D,), eu, dtype=DT)
            fn = torch.full((D,), db, dtype=DT)
            DD = torch.zeros(D, dtype=DT)
            EE = DD.clone()
            FF = DD.clone()
            BB = DD.clone()
            for n in range(1, 13):
                sign = (-1) ** (n + 1) / n
                BB += sign * bn
                DD += sign * dn0
                EE += sign * en
                FF += sign * fn
                bn, dn0, en, fn = (da * bn + em * en, da * dn0 + em * fn, eu * (da ** n + s * s * bn) + db * en, eu * s * s * dn0 + db * fn)
            FF -= ku
            Bs.append(V * BB @ V.T)
            Ds.append(V * DD @ V.T)
            Es.append(V * EE @ V.T)
            Fs.append(V * FF @ V.T)
            kl = logm(np.array([[am, em * float(s[-1])], [eu * float(s[-1]), au]]))
            lnnorm = max(lnnorm, float(np.linalg.norm(kl, 2)), abs(km), abs(ku))
            assert np.linalg.norm(expm(kl) - np.array([[am, em * float(s[-1])], [eu * float(s[-1]), au]])) < 1e-12
        Bm, Dm, Em, Fm = map(torch.stack, [Bs, Ds, Es, Fs])
        w = torch.arange(31, -33, -2, dtype=DT)[:, None]

        def batches(m, v):
            return torch.bmm(m, v[:, :, None])[:, :, 0]

        def omegas(x, second=False):
            al, r, u = x
            cm = al * cs + (GF @ r.reshape(-1)).reshape(T, D)
            ti = batches(Bm, cm) + torch.einsum('nij,j->ni', Dm, u)
            vi = batches(Em, cm) + torch.einsum('nij,j->ni', Fm, u)
            one = (32 * km * al, 32 * km * r + ti, 32 * ku * u + vi.sum(0))
            if not second:
                return one
            zd = (Ganti @ ti.reshape(-1)).reshape(T, D)
            vd = 2 * vi.cumsum(0) - vi - vi.sum(0)
            two = (0.0, 0.5 * (w * (km - ku) * torch.einsum('nij,j->ni', Dm, u) + batches(Bm, zd) + batches(Dm, vd)), 0.5 * (w * (ku - km) * batches(Em, cm) + batches(Em, zd) + batches(Fm, vd)).sum(0))
            return add(one, two)
        zero = (1.0, torch.zeros(T, D, dtype=DT), torch.zeros(D, dtype=DT))
        om1 = omegas(zero)
        om2 = add(omegas(zero, True), om1, -1.0)

        def exponent(second):
            xx = zero
            term = zero
            bound = 32 * lnnorm + (496 * lnnorm ** 2 if second else 0.0)
            for n in range(1, 19):
                term = scale(omegas(term, second), 1 / n)
                xx = add(xx, term)
            tail = math.exp(bound) * bound ** 19 / math.factorial(19) * sn
            assert tail < 1e-16
            return (xx, tail)
        first, tail1 = exponent(False)
        second, tail2 = exponent(True)
        fm, fu = real(first)
        smat, su = real(second)
        om, ou = ordered[k]
        baseM = seed + om
        d1 = [fm - baseM, fu - ou]
        d2 = [smat - baseM, su - ou]
        resM, resU = (om - (real((am ** 32, torch.zeros(T, D, dtype=DT), torch.zeros(D, dtype=DT)))[0] - seed), ou)
        p, q = CO[k][2:]
        v = cs.mean(0).clone()
        rr = torch.zeros(D, dtype=DT)
        usq = rr.clone()
        for n in range(16):
            rr += p[n] * v
            usq += q[n] * v
            v = Gmean @ v
        msq = am ** 32 * seed + sum((ct(cap, rr) for cap in caps)) / 32
        residual = [baseM - msq, ou - usq]
        corr = [smat - fm, su - fu]
        act1 = real(om1)
        act2 = real(om2)
        for sector, take in [('full_state', [0, 1]), ('expert_private', [0]), ('router_private', [1])]:

            def nn(v):
                return math.sqrt(sum((float(torch.sum(v[t] ** 2)) for t in take)))
            nr = nn(residual)
            nc = nn(corr)
            dot = sum((float((residual[t] * corr[t]).sum()) for t in take))
            errcorr = nn([residual[t] - corr[t] for t in range(2)])
            nfirst = nn(d1)
            nsecond = nn(d2)
            cumrows.append(dict(population=pop, stream=j, arm=k, sector=sector, map_norm_L_minus_I_upper=normA, map_min_eigenvalue_lower=mineig, real_log_stable=True, log_series_terms=12, log_series_operator_remainder_bound=rem, first_cumulant_spectral_upper_bound=32 * lnnorm, commutator_spectral_upper_bound=496 * lnnorm ** 2, first_cumulant_action_norm=nn(act1), commutator_action_norm=nn(act2), ordered_vs_seqmean_residual_norm=nr, second_order_action_correction_norm=nc, second_order_residual_alignment=dot / max(nr * nc, 1e-300), second_order_fraction_residual_explained=1 - (errcorr / max(nr, 1e-300)) ** 2, first_cumulant_vs_ordered_action_error=nfirst, second_cumulant_vs_ordered_action_error=nsecond, second_cumulant_fraction_first_error_reduced=1 - (nsecond / max(nfirst, 1e-300)) ** 2, exponential_action_remainder_bound=max(tail1, tail2)))
        log(f'stream {j} log cumulants arm {k}')
    pd.DataFrame(shiftrows).to_csv(OUT / 'FINITE_TIME_MEAN_SHIFT_CASES.csv', index=False, float_format='%.17g')
    pd.DataFrame(streamrows).to_csv(OUT / 'FINITE_TIME_STREAM_MEANS.csv', index=False, float_format='%.17g')
    pd.DataFrame(samplerows).to_csv(OUT / 'FINITE_TIME_SAMPLE_SIZE_CASES.csv', index=False, float_format='%.17g')
    pd.DataFrame(cumrows).to_csv(OUT / 'FINITE_TIME_CUMULANTS.csv', index=False, float_format='%.17g')
    np.savez_compressed(OUT / 'FINITE_TIME_PROJECTED_SINGULAR_DIRECTIONS.npz', **projarr)
    del caps, GS, GF, Ganti, dm, uu, cds, ordered, orderedR
aggregate(shiftrows, ['population'], [k for k in shiftrows[0] if k not in ['population', 'stream', 'arm']]).to_csv(OUT / 'FINITE_TIME_MEAN_SHIFT_AGGREGATES.csv', index=False, float_format='%.17g')
aggregate(samplerows, ['population', 'sector', 'm'], ['subset_median', 'subset_q025', 'subset_q975']).to_csv(OUT / 'FINITE_TIME_SAMPLE_SIZE.csv', index=False, float_format='%.17g')
aggregate(streamrows, ['population', 'sector', 'comparator'], ['closure_error', 'four_subset_median', 'same_stream_reference_percentile']).to_csv(OUT / 'FINITE_TIME_STREAM_MEAN_AGGREGATES.csv', index=False, float_format='%.17g')
aggregate(cumrows, ['population', 'sector'], ['first_cumulant_action_norm', 'commutator_action_norm', 'ordered_vs_seqmean_residual_norm', 'second_order_fraction_residual_explained', 'second_cumulant_fraction_first_error_reduced', 'first_cumulant_vs_ordered_action_error', 'second_cumulant_vs_ordered_action_error']).to_csv(OUT / 'FINITE_TIME_CUMULANT_AGGREGATES.csv', index=False, float_format='%.17g')
pd.DataFrame(valid).to_csv(OUT / 'NUMERICAL_VERIFICATION.csv', index=False, float_format='%.17g')
pd.DataFrame(SOURCES.values()).to_csv(OUT / 'ANALYSIS_SOURCE_MANIFEST.csv', index=False)
changed = [p for p, h in seal.items() if sha(p) != h]
assert not changed
(OUT / 'SOURCE_INTEGRITY_VERIFICATION.json').write_text(json.dumps(dict(status='PASS', sealed_archived_files=len(seal), changed_files=changed, no_training=True, no_source_overwrite=True, elapsed_seconds=time.time() - START), indent=2))
log('ALL NUMERICAL ANALYSIS COMPLETE')
