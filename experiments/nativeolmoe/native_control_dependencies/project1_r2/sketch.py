# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import numpy as np
import torch

def probe_seed(seed, document_id, layer, probe):
    text = f'{seed}:{document_id}:{layer}:{probe}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')

def output_probes(count, shape, seed, document_id, layer, device, family='rademacher'):
    probes = []
    for i in range(count):
        rng = np.random.default_rng(probe_seed(seed, document_id, layer, i))
        if family == 'rademacher':
            array = (2 * rng.integers(0, 2, size=shape, dtype=np.int8) - 1).astype(np.float32)
        elif family == 'gaussian':
            array = rng.normal(size=shape).astype(np.float32)
        else:
            raise ValueError('Unknown probe family.')
        probes.append(torch.from_numpy(array).to(device))
    return torch.stack(probes)

def local_sketch(local, count, seed, document_id, layer, family='rademacher'):
    """Matrix-free random-output projection: at most one probe and one JVP resident.

    The output dimension is normalized by tokens*hidden. Gram statistics weight
    documents equally, never average signed sketches across documents.
    """
    rows = []
    ne = local.n_experts
    for probe in range(count):
        rng = np.random.default_rng(probe_seed(seed, document_id, layer, probe))
        shape = tuple(local.hidden.shape)
        raw = (2 * rng.integers(0, 2, size=shape, dtype=np.int8) - 1).astype(np.float32) if family == 'rademacher' else rng.normal(size=shape).astype(np.float32)
        q = torch.from_numpy(raw).to(local.hidden.device)
        gain_row = torch.zeros(ne, device=q.device)
        bias_row = torch.zeros(ne, device=q.device)
        scalar_output = torch.zeros(len(q), device=q.device)
        for item in local.contributions:
            dot = (q[item.tokens] * item.outputs.float()).sum(-1)
            w = local.routes.weights[item.tokens, item.slots]
            gain_row[item.expert] = (dot * w.to(local.hidden.dtype).float()).sum()
            bias_row[item.expert] = (dot * w).sum()
            scalar_output.index_add_(0, item.tokens, dot * w)
        probabilities = local.routes.probabilities
        if local.normalized:
            probabilities = torch.zeros_like(probabilities)
            probabilities.scatter_(1, local.routes.indices, local.routes.weights)
        bias_row -= (probabilities * scalar_output[:, None]).sum(0)
        rows.append(torch.cat([gain_row, bias_row]).cpu().numpy())
    return np.asarray(rows) / np.sqrt(count * np.prod(shape))

def block_gram(sketches, block):
    sl = slice(0, 64) if block == 'expert_gain' else slice(64, 128)
    return sum((s[:, sl].T @ s[:, sl] for s in sketches)) / len(sketches)

def domain_balanced_sketches(sketches, domains):
    """Scale rows so mean Gram = equal domains, equal documents within domain."""
    from collections import Counter
    counts = Counter(domains)
    if len(sketches) != len(domains):
        raise ValueError('Sketch/document counts differ.')
    return [s * np.sqrt(len(sketches) / (len(counts) * counts[d])) for s, d in zip(sketches, domains)]

def spectral_report(sketches, rank):
    result = {}
    for block in ('expert_gain', 'router_bias'):
        gram = block_gram(sketches, block)
        eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        result[block] = dict(trace_estimate=float(np.trace(gram)), leading_eigenvalues=eigenvalues[:rank].tolist(), leading_singular_values=np.sqrt(np.maximum(eigenvalues[:rank], 0)).tolist())
    return result

def compare_sketches(first, second, rank, relative_tolerance, overlap_minimum):
    from .metrics import subspace_overlap
    result = {}
    for block in ('expert_gain', 'router_bias'):
        a, b = (block_gram(first, block), block_gram(second, block))
        trace_change = abs(np.trace(b) - np.trace(a)) / max(abs(np.trace(b)), 1e-30)
        ea, va = np.linalg.eigh(a)
        eb, vb = np.linalg.eigh(b)
        overlap = subspace_overlap(va[:, -rank:], vb[:, -rank:])
        spectrum_change = np.linalg.norm(ea[-rank:] - eb[-rank:]) / max(np.linalg.norm(eb[-rank:]), 1e-30)
        result[block] = dict(relative_trace_change=float(trace_change), relative_primary_norm_change=float(abs(np.sqrt(np.trace(a)) - np.sqrt(np.trace(b))) / max(np.sqrt(np.trace(b)), 1e-30)), relative_leading_eigenvalue_change=float(spectrum_change), subspace_canonical_correlations=overlap.tolist(), engineering_stable=bool(trace_change <= relative_tolerance and spectrum_change <= relative_tolerance and (overlap.mean() >= overlap_minimum)))
    return result
