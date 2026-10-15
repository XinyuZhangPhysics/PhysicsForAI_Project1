# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""One frozen primary endpoint; no reserve or post-Q3 model operations."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import numpy as np
from common import *

def main():
    freeze = frozen()
    assert not (BUNDLE / 'endpoint.json').exists(), 'ENDPOINT_ALREADY_COMPUTED'
    assert not (BUNDLE / 'technical_stop_trigger.json').exists(), 'TECHNICAL_STOP_LOCK'
    tables = []
    for arm in ARMS:
        folder = BUNDLE / 'arms' / arm
        r = read(folder / 'completion_receipt.json')
        assert sha(folder / 'completion_receipt.json') == (folder / 'completion_receipt.sha256').read_text().split()[0]
        assert r['arm'] == arm and r['status'] == 'COMPLETE' and (r['nll_table_sha256'] == sha(folder / 'heldout_nll.json'))
        rows = read(folder / 'heldout_nll.json')
        bydomain = {d: sorted([x for x in rows if x['domain'] == d], key=lambda x: x['document_id']) for d in DOMAINS}
        assert len(rows) == 448 and all((len(v) == 64 for v in bydomain.values()))
        tables.append(bydomain)
    for t in tables[1:]:
        assert all(([x['document_id'] for x in t[d]] == [x['document_id'] for x in tables[0][d]] for d in DOMAINS))
    losses = np.asarray([[[x['nll'] for x in t[d]] for d in DOMAINS] for t in tables], dtype=np.float64)
    assert np.isfinite(losses).all()
    L = losses.mean(axis=(1, 2))
    U = L[0] - L
    D = float(U[1] - U[2:].mean())
    per_doc_D = losses[2:].mean(0) - losses[1]
    per_doc_U = losses[0] - losses[1]
    seed_value = int.from_bytes(hashlib.sha256(b'PROJECT1-Q3-BOOTSTRAP-v173').digest()[:8], 'little')
    rng = np.random.Generator(np.random.PCG64(seed_value))
    boot = np.empty((10000, 2), dtype=np.float64)
    for rep in range(10000):
        idx = rng.integers(0, 64, size=(7, 64))
        ds = np.arange(7)[:, None]
        boot[rep] = [per_doc_D[ds, idx].mean(), per_doc_U[ds, idx].mean()]
    intervals = np.quantile(boot, [0.025, 0.975], axis=0, method='linear')
    low, high = intervals[:, 0]
    label = 'BENEFICIAL_RELATIVE_TO_RANDOM' if low > 0 else 'HARMFUL_RELATIVE_TO_RANDOM' if high < 0 else 'NULL_OR_UNRESOLVED'
    np.save(BUNDLE / 'bootstrap_replicates.npy', boot, allow_pickle=False)
    result = dict(Q3_COMPLETE=True, **HISTORY, Q3_RESERVE_USED_FOR_SCIENCE=False, Q3_HELDOUT_MODEL_ACCESS_BEFORE_ARM_SEAL=False, POST_Q3_EXPERIMENTS_NOT_EXECUTED=True, primary_label=label, D=D, D_CI95=[float(low), float(high)], candidate_utility=float(U[1]), candidate_vs_baseline_CI95=intervals[:, 1].tolist(), candidate_rank=1 + int(np.count_nonzero(U[2:] > U[1])), rank_tie_rule='1 + number of random utilities strictly larger', lower_inclusive_matched_null_percentile=float(np.mean(U[2:] <= U[1])), upper_inclusive_matched_null_percentile=float(np.mean(U[2:] >= U[1])), random_utilities={a: float(u) for a, u in zip(ARMS[2:], U[2:])}, replicates=10000, seed_string='PROJECT1-Q3-BOOTSTRAP-v173', seed_integer=seed_value, seed_conversion='first 8 SHA256 UTF8 bytes interpreted unsigned little endian', generator='NumPy PCG64', percentile_method='linear', conditional_inference='conditional on the frozen 39-control ensemble', scientific_freeze_sha256=sha(BUNDLE / 'Q3_SCIENTIFIC_FREEZE_v176.json'))
    write(BUNDLE / 'endpoint.json', result)
    write(BUNDLE / 'per_arm_summary.json', [dict(arm=a, L=float(l), U=float(u), final_alpha=read(BUNDLE / 'arms' / a / 'final_alpha.json')) for a, l, u in zip(ARMS, L, U)])
    event('ENDPOINT_SEALED', endpoint_sha256=sha(BUNDLE / 'endpoint.json'), bootstrap_sha256=sha(BUNDLE / 'bootstrap_replicates.npy'), post_Q3_experiments=False)
    print('Q3 endpoint complete; ' + label, flush=True)
if __name__ == '__main__':
    main()
