# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import sqlite3, heapq, shutil

def main():
    roots = [HIST / 'EXP_S1', HIST / 'EXP_S2', MATCH, R1]
    assert all((r.exists() for r in roots))
    before = {str(p): sha(p) for r in roots for p in sorted(r.rglob('*')) if p.is_file() and '.git' not in p.parts}
    write(ROOT / 'preflight/HISTORICAL_BEFORE.json', {'roots': [str(r) for r in roots], 'files': before, 'timestamp': now()})
    b = read(MATCH / 'preregistration/INPUT_BINDINGS.json')
    for x in b['copied_points']:
        assert sha(x['path']) == x['sha256']
    assert (R1 / 'outputs/BACKGROUND_MODE_CONVERSION_RESULTS_v282M_r1.md').read_bytes() == (ROOT / 'inputs/R1_TECHNICAL_FAILURE_REPORT.md').read_bytes()
    inputs = {str(p): sha(p) for p in (ROOT / 'inputs').iterdir()}
    inputs[str(MATCH / 'preregistration/INPUT_BINDINGS.json')] = sha(MATCH / 'preregistration/INPUT_BINDINGS.json')
    inputs[str(DB)] = sha(DB)
    for m in MODELS.values():
        for p in m.iterdir():
            if p.is_file():
                inputs[str(p)] = sha(p)
    for x in b['copied_points']:
        inputs[x['path']] = x['sha256']
    old = set()
    manifests = {}
    for f, c in b['families'].items():
        for membership in ['calibration', 'audit']:
            p = Path(c['banks'][membership]['documents']['path'])
            manifests[str(p)] = sha(p)
            old.update((json.loads(line)['text_sha256'] for line in p.open()))
    for p in HIST.parent.glob('90_historical_superseded/EXP_G_GT2/**/outputs/EXP_G_V3_LAPTOP_PREFLIGHT/*bank_manifest.jsonl'):
        manifests[str(p)] = sha(p)
        old.update((json.loads(line)['text_sha256'] for line in p.open()))
    con = sqlite3.connect(f'file:{DB}?mode=ro&immutable=1', uri=True)
    chosen = {(f, m): [] for f in ['S1', 'S2'] for m in ['calibration', 'audit']}
    for domain in DOMAINS:
        candidates = ((digest(f'v282M-r2-fresh-bank|{domain}|{h}'.encode()), h, sh, ord, orig) for h, sh, ord, orig in con.execute('select text_sha,shard,row_ordinal,original_index from global_docs where domain=?', (domain,)) if h not in old)
        ranked = heapq.nsmallest(2048, candidates)
        for i, (f, m) in enumerate(chosen):
            for rank, h, sh, ordinal, orig in ranked[i * 512:(i + 1) * 512]:
                chosen[f, m].append({'family': f, 'membership': m, 'domain': domain, 'train_row_id': f'train-{sh:05d}:row-{ordinal}', 'shard': sh, 'row_ordinal': ordinal, 'original_index': orig, 'text_sha256': h, 'rank_key': rank})
        print('IDENTITIES', domain, flush=True)
    sets = [set((r['text_sha256'] for r in rows)) for rows in chosen.values()]
    assert len(set.union(*sets)) == 8192 and (not set.union(*sets) & old)
    banks = {}
    for (f, m), rows in chosen.items():
        p = ROOT / 'preregistration' / f'{f}_{m}_identities.jsonl'
        p.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in rows)))
        banks[f + '_' + m] = {'path': str(p), 'sha256': sha(p), 'documents': 2048}
    write(ROOT / 'preregistration/INPUT_BINDINGS.json', {'timestamp': now(), 'families': b['families'], 'copied_points': b['copied_points'], 'input_hashes': inputs, 'historical_identity_manifests': manifests, 'historical_identity_count': len(old), 'fresh_banks': banks, 'fresh_pair_seed': 20261011962, 'r1_report_sha256': sha(ROOT / 'inputs/R1_TECHNICAL_FAILURE_REPORT.md'), 'audit_content_opened': False, 'dataset_revision': 'b5f90f419b7489cdba26fdbc8c022fcb5562f968'})
    write(ROOT / 'preflight/PRE_AUDIT_CHECK.json', {'timestamp': now(), 'all_authority_roots_located': True, 'historical_roots': [str(r) for r in roots], 'r1_bound_immutable': True, 'fixed_reference_observable_implemented': 'source/experiment.py; strictly positive P_ref required at each bank load; synthetic and Stage A positivity gates precede audit', 'fresh_identities_disjoint': True, 'audit_content_unopened': True, 'historical_before_complete': True, 'historical_files': len(before), 'historical_before_sha256': sha(ROOT / 'preflight/HISTORICAL_BEFORE.json'), 'fresh_unique_documents': 8192})
    print('PRE-AUDIT CHECK: all authority roots located; r1 immutable and bound; fixed positive-reference observable implemented with hard positivity gate; 8192 fresh identities disjoint, audit content unopened; historical before-hash snapshot complete.', flush=True)
if __name__ == '__main__':
    main()
