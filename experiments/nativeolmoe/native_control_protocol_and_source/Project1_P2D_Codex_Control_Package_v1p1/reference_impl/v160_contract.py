# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Prospective v160 firewall and direct transcription of frozen scientific formulas."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from types import MappingProxyType
import numpy as np
CONTRACT = MappingProxyType(dict(checkpoint='e5642d2d7b37c138775ddf59f685f91788e173b7', layers=(7, 15), top_k=8, experts=64, rank=1, ridge=0.001, nulls=256, split_threshold=0.7, visibility_threshold=0.25, seed='R2A-NULL-20261005', primary='clean_fixed_support_FP32_soft', bank='v156_A'))
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
FORBIDDEN = {'quantization', 'pruning', 'surrogate', 'weight_change', 'approximate_router', 'drop_documents'}
MATERIAL = ('device', 'compute_dtype', 'kernel', 'loading', 'torch', 'transformers', 'checkpoint')

def guard(options):
    for k, v in options.items():
        if k in FORBIDDEN or (k in CONTRACT and v != CONTRACT[k]):
            raise ValueError('Frozen scientific contract: ' + k)
        if k not in CONTRACT and k not in MATERIAL and (k not in ('batch_size', 'chunk_size', 'threads', 'offload', 'cache')):
            raise ValueError('Unknown option: ' + k)

def fingerprint(config):
    return hashlib.sha256(json.dumps({k: config[k] for k in MATERIAL}, sort_keys=True).encode()).hexdigest()

def require_certified(cert, config):
    if cert.get('BACKEND_EQUIVALENCE_CERTIFIED') != 'YES' or cert.get('configuration_sha256') != fingerprint(config):
        raise RuntimeError('Formal A firewall: absent certification or material configuration change')

def subset(rows):
    return [next((r for r in rows if r['domain'] == d and r['half'] == h)) for d in DOMAINS for h in (0, 1)]

def route_mismatch(new, ref):
    a, b = (np.asarray(new), np.asarray(ref))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 8 or (not a.size):
        raise ValueError('Aligned nonempty token x 8 support required')
    sets = np.any(np.sort(a, axis=1) != np.sort(b, axis=1), axis=1)
    slots = np.any(a != b, axis=1)
    return dict(set_mismatch_fraction=float(sets.mean()), order_only_fraction=float((slots & ~sets).mean()))

def errors(new, ref):
    a, b = (np.asarray(new, dtype=np.float64), np.asarray(ref, dtype=np.float64))
    if a.shape != b.shape or not a.size or (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
        raise ValueError('Aligned finite nonempty endpoints required')
    if a.ndim == 1:
        a, b = (a[:, None], b[:, None])
    size = np.linalg.norm(b, axis=1)
    nz = size[size > 0]
    floor = 1e-06 * np.median(nz) if len(nz) else 0.0
    delta = np.linalg.norm(a - b, axis=1)
    den = np.maximum(size, floor)
    e = np.divide(delta, den, out=np.where(delta == 0, 0.0, np.inf), where=den > 0)
    return dict(median=float(np.median(e)), q95=float(np.quantile(e, 0.95)), maximum=float(e.max()), reference_floor=float(floor), zero_reference_absolute_pass=bool(np.all(delta[size == 0] <= floor)), count=len(e))

def equivalent(e, channel):
    med, q95 = {'router': (0.005, 0.02), 'expert': (0.01, 0.05), 'soft': (0.01, 0.03)}[channel]
    return e['median'] <= med and e['q95'] <= q95 and e['zero_reference_absolute_pass']

def certification_pass(semantic, route, router, expert, soft, ordering, gates):
    return bool(semantic and route and equivalent(router, 'router') and equivalent(expert, 'expert') and equivalent(soft, 'soft') and ordering and gates and all((v == 'PASS' for v in gates.values())))

def same_order(ref, new):
    a, b = (np.asarray(ref), np.asarray(new))
    return all((np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]) for i in range(len(a)) for j in range(i) if abs(a[i] - a[j]) >= 1e-06 * max(abs(a[i]), abs(a[j]))))

def near_boundary(r, q):
    return abs(r - 0.7) < 0.02 or abs(q - 0.25) <= 4 / 256

def center(x):
    return x - x.mean(0, keepdims=True) - x.mean(1, keepdims=True) + x.mean()

def whitening(x, d):
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d)
    n = len(x)
    if x.shape != (n, 64) or n not in (448, 224) or any((np.sum(d == s) != n // 7 for s in DOMAINS)):
        raise ValueError('Frozen balanced A design required')
    means = np.stack([x[d == s].mean(0) for s in DOMAINS], axis=1)
    res = np.stack([row - means[:, DOMAINS.index(s)] for row, s in zip(x, d)])
    res -= res.mean(1, keepdims=True)
    cov = res.T @ res / (n - 7)
    cov = (cov + cov.T) / 2
    val, vec = np.linalg.eigh(cov)
    if val.min() < -1e-10 * max(1.0, float(val.max())):
        raise ArithmeticError('Negative covariance')
    val = np.maximum(val, 0)
    ridge = 0.001 * np.trace(cov) / 64
    reg = val + ridge
    if reg.min() <= 1e-12 * reg.max():
        raise ArithmeticError('Frozen eigenguard')
    w = vec * reg ** (-0.5) @ vec.T
    return (w, means, dict(n_documents=n, covariance_denominator=n - 7, ridge_coefficient=0.001, ridge_value=float(ridge), trace=float(np.trace(cov)), min_eigenvalue=float(val.min()), max_eigenvalue=float(val.max()), min_ridged_eigenvalue=float(reg.min()), max_ridged_eigenvalue=float(reg.max()), H_E_applied=True), cov)

def fit(u, m, d):
    wu, um, ud, uc = whitening(u, d)
    wm, mm, md, mc = whitening(m, d)
    y = np.vstack((wm @ center(mm), wu @ center(um)))
    left, s, _ = np.linalg.svd(y, full_matrices=False)
    a = left[:, 0]
    if a[np.argmax(abs(a))] < 0:
        a = -a
    g, b = (wm.T @ a[:64], wu.T @ a[64:])
    g -= g.mean()
    b -= b.mean()
    v = np.r_[g, b]
    v /= np.linalg.norm(v)
    return dict(a=a, v=v, W_U=wu, W_M=wm, cov_U=uc, cov_M=mc, summary=dict(sigma1=float(s[0]), sigma2=float(s[1]), sigma1_sigma2=float(s[0] / s[1]), gain_norm=float(np.linalg.norm(v[:64])), router_norm=float(np.linalg.norm(v[64:])), sign_pivot=int(np.argmax(abs(a))), router_whitening=ud, expert_whitening=md))

def split(u, m, d, h):
    h = np.asarray(h)
    d = np.asarray(d)
    halves = [fit(u[h == i], m[h == i], d[h == i]) for i in (0, 1)]
    return (float(abs(halves[0]['a'] @ halves[1]['a'])), halves)

def exact_soft_gram(local):
    """Same frozen FP32 tangent, algebraically contracted; FP64 sufficient reduction.

    Per token only the eight selected expert vectors are materialized. No approximation,
    output sketch, control subsampling, or expert subsampling is used.
    """
    import torch
    from project1_r2.hard_routing_diagnostic import fixed_support_reweighted_routes
    routes = fixed_support_reweighted_routes(local, torch.zeros(64, device=local.hidden.device))
    t, h = local.hidden.shape
    outputs = torch.zeros((t, 8, h), dtype=torch.float32, device=local.hidden.device)
    for item in local.contributions:
        outputs[item.tokens, item.slots] = item.outputs.float()
    z = outputs * routes.weights[:, :, None]
    dot = torch.bmm(z, z.transpose(1, 2)).cpu().double().numpy()
    ix = routes.indices.cpu().numpy()
    p = routes.probabilities.cpu().double().numpy()
    if local.normalized:
        p = np.zeros_like(p)
        np.put_along_axis(p, ix, routes.weights.cpu().double().numpy(), axis=1)
    a = np.zeros((64, 64), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            np.add.at(a, (ix[:, i], ix[:, j]), dot[:, i, j])
    cross = np.zeros((64, 64), dtype=np.float64)
    for i in range(8):
        np.add.at(cross, ix[:, i], dot[:, i, :].sum(1)[:, None] * p)
    q = p.T @ (p * dot.sum((1, 2))[:, None])
    gram = np.block([[a, a - cross], [a - cross.T, a - cross - cross.T + q]]) / (t * h)
    return (gram + gram.T) / 2
