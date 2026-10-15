# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Response algebra: direct/cached full transfer, permutation Reynolds estimator,
finite-difference probe, permutation bootstrap, C1-C5 metrics and the node decision.

Formulas are transcribed from the frozen preregistration (Sections 8, 10-17) and the
theory note (Theorem U4.2).  Nothing here reads a reference number.

The estimator and the cached full transfer are written **generically in the number of
experts and in the number of token sets** so that the production 64-expert / 8-set problem
and the mandatory synthetic H<=5 exact-enumeration calibration (preregistration Section 19)
run through the *same* code path.
"""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from u4_common import json_dump, log
from u4_model import N_EXPERTS, N_PARAMS_PER_EXPERT, manual_next_token_ce
DOMAINS = ['RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaStackExchange', 'RedPajamaC4']
TEST_KEYS = [f'test|{d}' for d in DOMAINS]
VAL_KEYS = [f'validation|{d}' for d in DOMAINS]
SET_KEYS = TEST_KEYS + VAL_KEYS
OFFDIAG = [(i, j) for i in range(4) for j in range(4) if i != j]

def load_permutations(path) -> Tuple[np.ndarray, List[np.ndarray]]:
    import json
    with open(path, 'r', encoding='utf-8') as fh:
        obj = json.load(fh)
    perms = np.asarray(obj['permutations'], dtype=np.int64)
    return (perms, [np.argsort(p) for p in perms])

def validate_permutations(perms, n_experts, want_compact_sha=None) -> Dict:
    ok_valid = all((sorted((int(x) for x in p)) == list(range(n_experts)) for p in perms))
    uniq = len({tuple((int(x) for x in p)) for p in perms}) == len(perms)
    dig = compact_permutation_sha256(perms)
    out = {'num_permutations': len(perms), 'num_experts': int(n_experts), 'all_valid': bool(ok_valid), 'all_unique': bool(uniq), 'compact_sha256': dig, 'compact_sha256_matches_preregistration': want_compact_sha is None or dig == want_compact_sha, 'compact_serialisation': "sha256(json.dumps(permutations, separators=(',',':')))"}
    out['pass'] = bool(ok_valid and uniq and out['compact_sha256_matches_preregistration'])
    return out

def compact_permutation_sha256(perms) -> str:
    import json
    from u4_common import sha256_bytes
    return sha256_bytes(json.dumps([[int(x) for x in p] for p in perms], separators=(',', ':')).encode('utf-8'))

def all_permutations(H: int) -> np.ndarray:
    import itertools
    return np.asarray(list(itertools.permutations(range(H))), dtype=np.int64)

def active_tokens(route_idx, n_experts: int):
    """For each expert label c: (token rows) where c is in the routed support."""
    idx = route_idx.numpy() if hasattr(route_idx, 'numpy') else np.asarray(route_idx)
    out = []
    for c in range(n_experts):
        rows = np.nonzero(idx == c)[0].astype(np.int64)
        out.append(rows)
    return out

def slot_of(route_idx, c: int):
    idx = route_idx.numpy() if hasattr(route_idx, 'numpy') else np.asarray(route_idx)
    rows, slot = np.nonzero(idx == c)
    return (rows.astype(np.int64), slot.astype(np.int64))

def estimator_matrices(moe, sets: Sequence[Dict], perms: np.ndarray, n_experts: int, n_params: int | Sequence[int] | None=None):
    """T^{(pi)}_{ab} = sum_j <g^{(pi)}_{j,a}, g^{(pi)}_{j,b}> for every ordered pair (a,b).

    g^{(pi)}_{j,a} = (1/N_a) sum_r a_{pi^{-1}(j)}(r) J_j(z_r)^T d_r .
    Returns an array of shape (n_pi, len(sets), len(sets)) in float64.
    """
    n_sets = len(sets)
    n_pi = perms.shape[0]
    inv = [np.argsort(p) for p in perms]
    z = [s['z'] for s in sets]
    d = [s['d'] for s in sets]
    w = [s['route_w'] for s in sets]
    act = [slot_of(s['route_idx'], c) for s in sets for c in []]
    act = [[slot_of(sets[si]['route_idx'], c) for c in range(n_experts)] for si in range(n_sets)]
    N = [float(s['N']) for s in sets]
    T = np.zeros((n_pi, n_sets, n_sets), dtype=np.float64)
    for pi_i in range(n_pi):
        for j in range(n_experts):
            c = int(inv[pi_i][j])
            U = None
            for si in range(n_sets):
                rows, slot = act[si][c]
                if rows.size == 0:
                    continue
                dadj = d[si][rows] * w[si][rows, slot].unsqueeze(-1)
                g = moe.expert_grad_scaled(j, z[si][rows], dadj)
                gn = g.numpy().astype(np.float64)
                if U is None:
                    U = np.zeros((n_sets, gn.size), dtype=np.float64)
                U[si] = gn
            if U is None:
                continue
            T[pi_i] += U @ U.T
    return T

def cached_full_from_sets(moe, sets: Sequence[Dict], n_experts: int):
    """Cached (Jacobian) full transfer: the identity-label permutation, i.e. T^full."""
    n_sets = len(sets)
    z = [s['z'] for s in sets]
    d = [s['d'] for s in sets]
    w = [s['route_w'] for s in sets]
    N = [float(s['N']) for s in sets]
    act = [[slot_of(sets[si]['route_idx'], c) for c in range(n_experts)] for si in range(n_sets)]
    G = np.zeros((n_sets, n_sets), dtype=np.float64)
    for j in range(n_experts):
        U = None
        for si in range(n_sets):
            rows, slot = act[si][j]
            if rows.size == 0:
                continue
            dadj = d[si][rows] * w[si][rows, slot].unsqueeze(-1)
            g = moe.expert_grad_scaled(j, z[si][rows], dadj)
            gn = g.numpy().astype(np.float64)
            if U is None:
                U = np.zeros((n_sets, gn.size), dtype=np.float64)
            U[si] = gn
        if U is not None:
            G += U @ U.T
    return G

def flatten_blocks(blocks: Sequence, per_expert: int=3) -> List[np.ndarray]:
    """Concatenate the per-expert parameter-gradient blocks in fixed order."""
    out = []
    for j in range(len(blocks) // per_expert):
        chunk = blocks[j * per_expert:(j + 1) * per_expert]
        out.append(np.concatenate([np.asarray(c).reshape(-1) for c in chunk]))
    return out

def direct_full_matrix(blocks_target: Sequence[np.ndarray], blocks_source: Sequence[np.ndarray]):
    G = np.zeros((len(blocks_target), len(blocks_source)), dtype=np.float64)
    for i, a in enumerate(blocks_target):
        for j, b in enumerate(blocks_source):
            G[i, j] = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    return G

def probe_from_direction(moe, sets: Sequence[Dict], direction: Sequence, eps: float):
    """[L_t(theta + eps d) - L_t(theta - eps d)] / (2 eps) for every set in `sets`."""
    import torch
    out = []
    for s in sets:
        z, w, idx = (s['z'], s['route_w'], s['route_idx'])
        Ls = []
        for sign in (+1.0, -1.0):
            deltas = [[eps * sign * direction[j][k] for k in range(3)] for j in range(len(direction))]
            m = moe.moe(z, w, idx, deltas)
            logits = moe.head(s['h_att'] + m)
            Ls.append(float(manual_next_token_ce(logits, s['ids'])))
        out.append((Ls[0] - Ls[1]) / (2.0 * eps))
    return out

def permutation_bootstrap(T_pi, T_full, D_primary, G_norm_src, n_rep, seed, target_idx=None, source_idx=None, offdiag=None):
    """Nonparametric bootstrap over the frozen permutation matrices (Section 11)."""
    Tpi = T_pi
    if target_idx is not None:
        Tpi = Tpi[:, target_idx][:, :, source_idx]
    nt, ns = (Tpi.shape[1], Tpi.shape[2])
    if offdiag is None:
        offdiag = [(i, j) for i in range(nt) for j in range(ns) if i != j]
    D = D_primary
    Pfull = np.zeros((nt, ns), dtype=np.float64)
    for i in range(nt):
        for j in range(ns):
            Pfull[i, j] = -T_full[i, j] / G_norm_src[j]
    mae_full = float(np.mean([abs(D[i, j] - Pfull[i, j]) for i, j in offdiag]))
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    n_pi = Tpi.shape[0]
    I_arr = np.zeros(n_rep)
    SB_arr = np.zeros(n_rep)
    n_off = len(offdiag)
    Pr_off = np.zeros((n_rep, n_off))
    for b in range(n_rep):
        sel = rng.integers(0, n_pi, n_pi)
        Tr = Tpi[sel].mean(axis=0)
        errs_r, errs_b = ([], [])
        for k, (i, j) in enumerate(offdiag):
            pr = -Tr[i, j] / G_norm_src[j]
            Pr_off[b, k] = pr
            errs_r.append(abs(D[i, j] - pr))
            errs_b.append(abs(Pfull[i, j] - pr))
        I_arr[b] = 1.0 - mae_full / float(np.mean(errs_r))
        SB_arr[b] = float(np.sqrt(np.mean(np.square(errs_b))))
    return {'I_MAE': I_arr, 'S_B': SB_arr, 'P_route_replicates': Pr_off, 'P_full': Pfull, 'mae_full': mae_full, 'seed': int(seed), 'n_replicates': int(n_rep), 'offdiag': [tuple((int(x) for x in ij)) for ij in offdiag]}

def percentile_ci(arr, lo=2.5, hi=97.5):
    return (float(np.percentile(arr, lo)), float(np.percentile(arr, hi)))

def compute_metrics(T_full, T_route, D, D_primary, boot, G_norm_src, radii, cfg):
    md = {}
    Pfull = boot['P_full']
    Proute = np.zeros_like(T_full)
    for i in range(T_route.shape[0]):
        for j in range(T_route.shape[1]):
            Proute[i, j] = -T_route[i, j] / G_norm_src[j]
    PB = Pfull - Proute
    off = OFFDIAG
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    d_off = np.array([D_primary[i, j] for i, j in off])
    ef = np.array([D_primary[i, j] - Pfull[i, j] for i, j in off])
    er = np.array([D_primary[i, j] - Proute[i, j] for i, j in off])
    E_full = rms(ef)
    R_full = E_full / max(rms(d_off), 1e-12)
    MAE_full = float(np.mean(np.abs(ef)))
    MAE_route = float(np.mean(np.abs(er)))
    I_MAE = 1.0 - MAE_full / MAE_route
    S_B = rms(np.array([PB[i, j] for i, j in off]))
    sd = boot['P_route_replicates'].std(axis=0, ddof=1)
    E_MC = rms(sd)
    Q_B = S_B / max(E_full, E_MC, 1e-12)
    ci_I = percentile_ci(boot['I_MAE'])
    ci_SB = percentile_ci(boot['S_B'])
    D_RMS = rms(d_off)
    dscale = []
    for i, j in off:
        vals = [D[rho][i, j] for rho in radii]
        den = max(abs(D_primary[i, j]), D_RMS, 1e-12)
        dscale.append((max(vals) - min(vals)) / den)
    dscale = np.array(dscale)
    locality = bool(np.median(dscale) <= cfg['thresholds']['locality_median'] and dscale.max() <= cfg['thresholds']['locality_max'])
    md.update({'rho_primary': cfg['radii']['primary'], 'radii': list(radii), 'RMS_D_probe': rms(d_off), 'E_full': E_full, 'R_full': R_full, 'MAE_full': MAE_full, 'MAE_route': MAE_route, 'I_MAE': I_MAE, 'I_MAE_CI95': list(ci_I), 'S_B': S_B, 'S_B_CI95': list(ci_SB), 'E_MC': E_MC, 'E_MC_over_S_B': E_MC / S_B, 'Q_B': Q_B, 'median_delta_scale': float(np.median(dscale)), 'max_delta_scale': float(dscale.max()), 'D_RMS': D_RMS, 'locality_resolved': locality, 'max_abs_D_probe_offdiag': float(np.abs(d_off).max()), 'T_full': T_full, 'T_route_hat': T_route, 'B_hat': T_full - T_route, 'P_full': Pfull, 'P_route': Proute, 'P_B': PB, 'D_probe_primary': D_primary, 'delta_scale_rows': [(int(i), int(j), float(v)) for (i, j), v in zip(off, dscale)]})
    thr = cfg['thresholds']
    C = {}
    C['C1'] = {'R_full': R_full, 'threshold': thr['C1_R_full_max'], 'pass': bool(R_full <= thr['C1_R_full_max'])}
    C['C2'] = {'I_MAE': I_MAE, 'CI_low': ci_I[0], 'threshold': thr['C2_I_MAE_min'], 'CI_low_threshold': thr['C2_CI_low_min'], 'pass': bool(I_MAE >= thr['C2_I_MAE_min'] and ci_I[0] > thr['C2_CI_low_min'])}
    C['C3'] = {'Q_B': Q_B, 'S_B': S_B, 'S_B_CI_low': ci_SB[0], 'threshold': thr['C3_Q_B_min'], 'pass': bool(Q_B >= thr['C3_Q_B_min'] and ci_SB[0] > 0.0)}
    C['C4'] = {'E_MC_over_S_B': E_MC / S_B, 'threshold': thr['C4_E_MC_over_S_B_max'], 'pass': bool(E_MC / S_B <= thr['C4_E_MC_over_S_B_max'])}
    C['C5'] = {'median_delta_scale': float(np.median(dscale)), 'max_delta_scale': float(dscale.max()), 'thresholds': [thr['locality_median'], thr['locality_max']], 'pass': locality}
    md['gates'] = C
    return md

def decide(gates: Dict, validity_ok: bool, blocked: str | None=None):
    if blocked:
        return blocked
    if not validity_ok:
        return 'INVALID'
    if all((g['pass'] for g in gates.values())):
        return 'CONFIRMED'
    if not gates['C4']['pass'] and all((gates[k]['pass'] for k in ('C1', 'C2', 'C3', 'C5'))):
        return 'INCONCLUSIVE_MC'
    return 'NOT_CONFIRMED'
