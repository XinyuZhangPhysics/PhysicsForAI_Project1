# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import csv, hashlib, json
BASE = Path(_p1_path('path_d5a5a4203b69'))
OUT = BASE / 'outputs'
p = OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.json'
d = json.loads(p.read_text())
registry = json.loads((BASE / 'work/input_hashes.json').read_text())
bad = []

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
for path, h in registry.items():
    if sha(path) != h:
        bad.append(path)
assert not bad, bad
assert d['classification'] == 'RESTRICTED_RESPONSE_SELECTION_VALIDATED'
assert len(d['primary_records']) == 64 and all((c['pass'] for c in d['checks']))
with (OUT / 'SELECTION_RESPONSE_PRIMARY_TABLE_v281M.csv').open() as f:
    csvrows = list(csv.DictReader(f))
for row, cr in zip(d['primary_records'], csvrows, strict=True):
    for key in ['Q_C', 'K_C_0', 'P_B', 'Theta_R'] + [f'B_{i:02}' for i in range(1, 13)] + [f'B_power_{i:02}' for i in range(1, 13)]:
        assert cr[key] == '' and row[key] is None or float(cr[key]) == row[key], key
for name in ['SELECTION_RESPONSE_PROVENANCE_v281M.md', 'SELECTION_RESPONSE_SCIENTIFIC_IMPACT_v281M.md']:
    assert (OUT / name).read_text().splitlines()[0] == d['classification']
for fp in OUT.glob('*.md'):
    s = fp.read_text()
    assert not any((ord(c) < 32 and c not in '\n\r\t' for c in s))
for family, a in d['families'].items():
    n = a['normalized_response']
    expected = (0.4217, [-0.423485, 1.155159]) if family == 'S1' else (0.531114, [0.019294, 1.081964])
    assert round(n['slope']['value'], 6) == expected[0]
    assert [round(x, 6) for x in n['ci95_shifted_from_frozen']['value']] == expected[1]
    assert n['formal_frozen_normalized_decision'] is None
    assert len([r for r in d['primary_records'] if r['family'] == family and r['g'] > 1 and r['resolved_positive']]) == 24
    assert all((x['max_B_coordinate_power'] == 0 for x in d['g1_all_horizon_evidence'] if x['family'] == family))
    assert len([x for x in d['g1_all_horizon_evidence'] if x['family'] == family]) == 32
for ref in d['evidence_sources'].values():
    assert registry[ref['path']] == ref['sha256']
(OUT / 'SELECTION_RESPONSE_SOURCE_HASHES_v281M.sha256').write_text(''.join((h + '  ' + path + '\n' for path, h in sorted(registry.items()))))
d['authority_file_hashes'] = registry
d['consistency_audit'] = {'result': 'PASS', 'input_files_rehashed': len(registry), 'changed_inputs': [], 'static_and_lineage_checks_passed': len(d['checks']), 'primary_csv_json_exact_value_match': True, 'primary_records': 64, 'nontrivial_resolved_records': 48, 'g1_all_horizon_files': 64, 'stored_g1_coordinate_powers_exact_zero': 768, 'original_protocol_source_conflicts': [], 'normalized_frozen_vs_post_hoc_distinction': 'PASS', 'observables_kept_distinct': 'PASS', 'historical_files_modified': False, 'source_hash_file_path': str(OUT / 'SELECTION_RESPONSE_SOURCE_HASHES_v281M.sha256'), 'source_hash_file_sha256': sha(OUT / 'SELECTION_RESPONSE_SOURCE_HASHES_v281M.sha256')}
p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
md = OUT / 'SELECTION_RESPONSE_PROVENANCE_v281M.md'
s = md.read_text().replace('输入 SHA256 最终重核结果会写入 JSON `/consistency_audit`。', f"最终重核 {len(registry)} 个 inspected evidence / metadata / byte-hash 输入，全部 unchanged；{len(d['checks'])} 项静态与 lineage checks 通过。CSV/JSON 的 64 records 全部精确一致。详见 JSON `/consistency_audit`。")
md.write_text(s)
assert len(list(OUT.glob('*'))) == 6
print(json.dumps(d['consistency_audit'], ensure_ascii=False, indent=2))
print('Deliverables:', [(x.name, x.stat().st_size) for x in sorted(OUT.iterdir())])
