# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import numpy as np
from . import StopPilot

def double_center(matrix):
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError('Expected a matrix.')
    return x - x.mean(axis=0, keepdims=True) - x.mean(axis=1, keepdims=True) + x.mean()

def subspace_overlap(first, second):
    if first.shape[1] != second.shape[1]:
        raise ValueError('Subspace ranks differ.')
    if np.linalg.matrix_rank(first) < first.shape[1] or np.linalg.matrix_rank(second) < second.shape[1]:
        raise StopPilot('Rank-deficient subspace; canonical correlation undefined.')
    qa, _ = np.linalg.qr(first)
    qb, _ = np.linalg.qr(second)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)

def canonical_correlation(first, second, rank):
    va = np.linalg.svd(first, full_matrices=False)[2][:rank].T
    vb = np.linalg.svd(second, full_matrices=False)[2][:rank].T
    if np.linalg.matrix_rank(first) < rank or np.linalg.matrix_rank(second) < rank:
        raise StopPilot('Internal rank too small for frozen CCA rank.')
    return subspace_overlap(va, vb)

def domain_matrix(rows, domains, field, half=None):
    columns = []
    for domain in domains:
        values = [np.asarray(r[field]) for r in rows if r['domain'] == domain and (half is None or r['half'] == half)]
        if not values:
            raise StopPilot('Missing domain/half sufficient statistics.')
        columns.append(np.mean(values, axis=0))
    return np.stack(columns, axis=1)

def internal_report(rows, domains, rank):
    matrices = {}
    result = {}
    for field in ('logit_mean', 'expert_internal_amplitude'):
        raw = domain_matrix(rows, domains, field)
        full = double_center(raw)
        halves = [double_center(domain_matrix(rows, domains, field, half)) for half in (0, 1)]
        norms = [np.linalg.norm(h) for h in halves]
        discrepancy = abs(norms[0] - norms[1]) / max(np.mean(norms), 1e-30)
        vectors = [np.linalg.svd(h, full_matrices=False)[0][:, :rank] for h in halves]
        if any((np.linalg.matrix_rank(h) < rank for h in halves)):
            raise StopPilot('Internal split-half rank is below frozen rank.')
        correlations = subspace_overlap(*vectors)
        result[field] = dict(raw_expert_by_domain_matrix=raw.tolist(), doubly_centered_matrix=full.tolist(), internal_amplitude=float(np.linalg.norm(full)), split_half_relative_discrepancy=float(discrepancy), split_half_canonical_correlations=correlations.tolist(), engineering_amplitude_ok=bool(discrepancy <= 0.2), engineering_subspace_ok=bool(correlations.mean() >= 0.7))
        matrices[field] = (full, halves)
    router, expert = (matrices['logit_mean'], matrices['expert_internal_amplitude'])
    correlation = canonical_correlation(router[0], expert[0], rank)
    joint_halves = []
    for h in (0, 1):
        r, e = (router[1][h], expert[1][h])
        joint_halves.append(np.vstack([e / max(np.linalg.norm(e), 1e-30), r / max(np.linalg.norm(r), 1e-30)]))
    vectors = [np.linalg.svd(y, full_matrices=False)[0][:, :rank] for y in joint_halves]
    candidate_cc = subspace_overlap(*vectors)
    result.update(router_expert_internal_canonical_correlations=correlation.tolist(), candidate_subspace_split_half_canonical_correlations=candidate_cc.tolist(), candidate_subspace_engineering_ok=bool(candidate_cc.mean() >= 0.7))
    return (result, router[0], expert[0])

def document_response_report(sketches, controls, domains):
    """Document-unit SE, domain stratification, uncertainty of RMS response."""
    result = {}
    for name, control in controls.items():
        means, variances = ([], [])
        for domain in sorted(set(domains)):
            powers = np.array([np.linalg.norm(sketch @ control) ** 2 for sketch, d in zip(sketches, domains) if d == domain])
            if len(powers) < 2:
                raise StopPilot('At least two documents per domain required for SE.')
            means.append(powers.mean())
            variances.append(powers.var(ddof=1) / len(powers))
        power = np.mean(means)
        power_se = np.sqrt(sum(variances)) / len(means)
        norm = np.sqrt(power)
        relative_se = 0.5 * power_se / max(power, 1e-30) if power > 0 else float('inf')
        result[name] = dict(response_rms=float(norm), relative_standard_error=float(relative_se), document_count=len(sketches), domains=len(means), engineering_snr_ok=bool(relative_se <= 0.2))
    return result
