# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Frozen Formal-A v159 statistics and null-control helpers."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import numpy as np
RIDGE = 0.001
RANK = 1
LAYERS = (7, 15)
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
NULL_SEED = 'R2A-NULL-20261005'
CHECKPOINTS = ((50, 41, 'step10000-tokens41B', 'e5642d2d7b37c138775ddf59f685f91788e173b7'), (150, 146, 'step35000-tokens146B', '608f483a04570e8dca8d7be3cc0da6ee54dce350'), (500, 503, 'step120000-tokens503B', 'cecae3d36575569db9b2ccf1eef095af7820f22a'), (1000, 1006, 'step240000-tokens1006B', '3e60f41bb1fc064fa0123db72fa7ff282d6a9c84'), (1750, 1740, 'step415000-tokens1740B', '6b019278dd3beb6005268dd72a1285363f3ad171'), (2500, 2495, 'step595000-tokens2495B', '8d2d59277df98763e2de2d3569667fff7e8a0546'), (3250, 3250, 'step775000-tokens3250B', '874127f5abce55c12439fadcd6dc39021bb482ad'), (4000, 4005, 'step955000-tokens4005B', 'a08635184887c71ad6e3d0a6617327a516d66c7c'), (4750, 4760, 'step1135000-tokens4760B', 'eb7c4702989e446075157383a9ce61bdccc645ba'), (5033, 5033, 'step1200000-tokens5033B', 'b99743a400844e532bec7ce4575d97a3a39dcd4d'))

def checkpoint_grid():
    return [dict(index=i, old_design_anchor_tokens_B=a, exact_tokens_B=t, branch_label=b, revision_commit_sha=s) for i, (a, t, b, s) in enumerate(CHECKPOINTS)]

def _center(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('Expected 2D matrix')
    return x - x.mean(0, keepdims=True) - x.mean(1, keepdims=True) + x.mean()

def pooled_whitener(rows, domain_ids):
    """Return symmetric A-only inverse square root; denominator is N - 7."""
    x = np.asarray(rows, dtype=np.float64)
    d = np.asarray(domain_ids)
    if x.ndim != 2 or x.shape[1] != 64 or len(x) != len(d):
        raise ValueError('Expected aligned document x 64 rows and domain ids')
    n = len(x)
    if n not in (224, 448):
        raise ValueError('Expected 448 full-A rows or 224 half-A rows')
    means = {s: x[d == s].mean(0) for s in np.unique(d)}
    if len(means) != 7 or any(((d == s).sum() != n // 7 for s in means)):
        raise ValueError('Expected seven balanced domains')
    residual = np.stack([row - means[s] for row, s in zip(x, d)])
    covariance = residual.T @ residual / (n - 7)
    covariance = (covariance + covariance.T) / 2
    eigval, eigvec = np.linalg.eigh(covariance)
    scale = max(1.0, float(eigval.max()))
    if eigval.min() < -1e-10 * scale:
        raise ArithmeticError('Whitening covariance has a materially negative eigenvalue')
    eigval = np.maximum(eigval, 0)
    ridge = RIDGE * float(np.trace(covariance)) / 64
    regularized = eigval + ridge
    if regularized.min() <= 1e-12 * regularized.max():
        raise ArithmeticError('Whitening ridge failed eigenvalue guard')
    w = eigvec * regularized ** (-0.5) @ eigvec.T
    return (w, dict(n_documents=n, covariance_denominator=n - 7, trace=float(np.trace(covariance)), ridge_coefficient=RIDGE, ridge_value=ridge, min_eigenvalue=float(eigval.min()), max_eigenvalue=float(eigval.max()), min_ridged_eigenvalue=float(regularized.min()), max_ridged_eigenvalue=float(regularized.max())))

def domain_means(values, domain_ids):
    x = np.asarray(values, dtype=np.float64)
    d = np.asarray(domain_ids)
    return np.stack([x[d == s].mean(0) for s in DOMAINS], axis=1)

def fit_joint_mode(router_rows, expert_rows, domain_ids):
    """Full/half A fit. Output vector order is [gain 64, router bias 64]."""
    u, m, d = (np.asarray(x) for x in (router_rows, expert_rows, domain_ids))
    if u.shape != m.shape or u.shape != (len(d), 64):
        raise ValueError('Expected aligned 64-channel router/expert rows')
    wu, udiag = pooled_whitener(u, d)
    wm, mdiag = pooled_whitener(m, d)
    yc = np.vstack((wm @ _center(domain_means(m, d)), wu @ _center(domain_means(u, d))))
    left, singular, _ = np.linalg.svd(yc, full_matrices=False)
    a = left[:, 0]
    if a[np.argmax(np.abs(a))] < 0:
        a = -a
    gain = wm.T @ a[:64]
    bias = wu.T @ a[64:]
    gain -= gain.mean()
    bias -= bias.mean()
    norm = np.sqrt(gain @ gain + bias @ bias)
    if not np.isfinite(norm) or norm == 0:
        raise ArithmeticError('Invalid mapped joint direction')
    gain, bias = (gain / norm, bias / norm)
    return (np.r_[gain, bias], dict(sigma1=float(singular[0]), sigma2=float(singular[1]), sigma1_sigma2=float(singular[0] / singular[1]) if singular[1] else None, gain_norm=float(np.linalg.norm(gain)), router_norm=float(np.linalg.norm(bias)), signed_pivot_component=float(a[np.argmax(np.abs(a))]), router_whitening=udiag, expert_whitening=mdiag))

def split_half_overlap(router_rows, expert_rows, domain_ids, half_ids):
    u, m, d, h = map(np.asarray, (router_rows, expert_rows, domain_ids, half_ids))
    if u.shape != (448, 64) or m.shape != u.shape or d.shape != (448,) or (h.shape != (448,)):
        raise ValueError('Split-half replay requires 448 aligned document rows')
    modes = []
    for part in (0, 1):
        mask = h == part
        if int(mask.sum()) != 224 or any((int(np.sum((d == s) & mask)) != 32 for s in np.unique(d))):
            raise ValueError('Each split half must contain 32 documents per domain')
        mode, _ = fit_joint_mode(u[mask], m[mask], d[mask])
        modes.append(mode)
    return (float(abs(modes[0] @ modes[1])), modes)

def layer_seed(layer):
    return int.from_bytes(hashlib.sha256(f'{NULL_SEED}|layer|{layer}'.encode()).digest()[:8], 'big')

def matched_controls(candidate, layer, n=256):
    v = np.asarray(candidate, dtype=np.float64)
    if v.shape != (128,) or n != 256:
        raise ValueError('Frozen controls require a 128-D candidate and exactly 256 draws')
    rg, rb = (np.linalg.norm(v[:64]), np.linalg.norm(v[64:]))
    rng = np.random.Generator(np.random.PCG64(layer_seed(layer)))
    out = []
    for _ in range(n):
        blocks = []
        for target in (rg, rb):
            z = rng.standard_normal(64)
            z -= z.mean()
            norm = np.linalg.norm(z)
            if norm < 1e-12:
                raise ArithmeticError('Projected control block norm below 1e-12')
            blocks.append(z * (target / norm))
        out.append(np.r_[blocks[0], blocks[1]])
    return np.asarray(out)

def empirical_percentile(candidate_visibility, null_visibilities):
    p = np.asarray(null_visibilities, dtype=np.float64)
    if p.shape != (256,) or not np.isfinite(p).all() or (not np.isfinite(candidate_visibility)):
        raise ValueError('Need 256 finite nulls and one finite candidate visibility')
    return float(np.mean(p <= candidate_visibility))

def outcome(split_overlap, percentile):
    if not np.isfinite(split_overlap) or not np.isfinite(percentile):
        raise ValueError('Nonfinite scientific gate')
    if split_overlap < 0.7:
        return 'NOT_REPRODUCIBLE'
    if percentile <= 0.25:
        return 'QUALIFIED_LATENT_CANDIDATE'
    return 'REPRODUCIBLE_BUT_NOT_LOW_VISIBLE'
