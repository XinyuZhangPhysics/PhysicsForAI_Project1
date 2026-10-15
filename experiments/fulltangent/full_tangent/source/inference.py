# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *

def rank_rows(v):
    a = np.asarray(v)
    idx = np.argsort(a, axis=-1, kind='stable')
    s = np.take_along_axis(a, idx, axis=-1)
    n = s.shape[-1]
    positions = np.broadcast_to(np.arange(n), s.shape)
    begin = np.concatenate([np.ones(s.shape[:-1] + (1,), bool), s[..., 1:] != s[..., :-1]], axis=-1)
    end = np.concatenate([s[..., :-1] != s[..., 1:], np.ones(s.shape[:-1] + (1,), bool)], axis=-1)
    lo = np.maximum.accumulate(np.where(begin, positions, 0), axis=-1)
    hi = np.minimum.accumulate(np.where(end, positions, n - 1)[..., ::-1], axis=-1)[..., ::-1]
    rank = (lo + hi) / 2 + 1
    out = np.empty_like(rank, dtype=float)
    np.put_along_axis(out, idx, rank, axis=-1)
    return out

def correlations(x, y):
    dot = np.sum(x * y, axis=-1)
    cos = dot / np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    pear = np.sum(xc * yc, axis=-1) / np.sqrt(np.sum(xc * xc, axis=-1) * np.sum(yc * yc, axis=-1))
    rx = rank_rows(x)
    ry = rank_rows(y)
    rx -= rx.mean(axis=-1, keepdims=True)
    ry -= ry.mean(axis=-1, keepdims=True)
    sp = np.sum(rx * ry, axis=-1) / np.sqrt(np.sum(rx * rx, axis=-1) * np.sum(ry * ry, axis=-1))
    return {'cosine': cos, 'pearson': pear, 'spearman': sp}

def all_metrics(x, y, fr, ff):
    xf = x.reshape(x.shape[:-3] + (-1,))
    yf = y.reshape(y.shape[:-3] + (-1,))
    r = correlations(xf, yf)
    valid = (np.abs(x) > fr) & (np.abs(y) > ff)
    same = (np.sign(x) == np.sign(y)) & valid
    r['sign_agreement'] = np.sum(same, axis=(-3, -2, -1)) / np.sum(valid, axis=(-3, -2, -1))
    xn = x / np.sqrt(np.mean(x * x, axis=(-3, -1), keepdims=True))
    yn = y / np.sqrt(np.mean(y * y, axis=(-3, -1), keepdims=True))
    r['scale_free_slope'] = np.sum(xn * yn, axis=(-3, -2, -1)) / np.sum(xn * xn, axis=(-3, -2, -1))
    return r

def bootstrap(x, y, fr, ff, indices):
    result = {k: [] for k in ['cosine', 'pearson', 'spearman', 'sign_agreement', 'scale_free_slope']}
    for lo in range(0, len(indices), 500):
        v = all_metrics(x[indices[lo:lo + 500]], y[indices[lo:lo + 500]], fr, ff)
        for k in result:
            result[k].append(v[k])
    return {k: np.concatenate(v) for k, v in result.items()}
