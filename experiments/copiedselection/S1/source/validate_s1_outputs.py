# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
WORK = ROOT / 'work'
OLD = Path(_p1_path('path_44e1748c77d6'))
DOMAINS = ('RedPajamaArXiv', 'RedPajamaGithub', 'RedPajamaWikipedia', 'RedPajamaStackExchange')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_lines(path):
    return [json.loads(line) for line in path.open()]

def sequence_hash(ids, mask):
    return hashlib.sha256(json.dumps({'input_ids': ids, 'attention_mask': mask}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assets = json.loads((OUT / 'ASSET_MANIFEST.json').read_text())
assert assets['dataset_train_shards_verified'] == 48
assert assets['model_revision'] == 'a39f36b100fe8a5377810d56c3f4789b9c53ac42'
for name, record in assets['model_files'].items():
    assert sha(WORK / 'model' / name) == record['sha256']
data = json.loads((OUT / 'DATA_MANIFEST_HASHES.json').read_text())
for key, name in [('calibration', 'calibration_bank_manifest.jsonl'), ('audit', 'audit_bank_manifest.jsonl'), ('calibration_pairs', 'calibration_pair_manifest.jsonl'), ('audit_pairs', 'audit_pair_manifest.jsonl'), ('training_pool_compressed', 'training_pool_manifest.jsonl.zst'), ('benchmark_source', 'benchmark_source_manifest.jsonl')]:
    assert sha(OUT / name) == data['manifest_sha256'][key]
old = set()
for name in ('calibration', 'audit'):
    old.update((r['text_sha256'] for r in read_lines(OLD / f'{name}_bank_manifest.jsonl')))
cal = read_lines(OUT / 'calibration_bank_manifest.jsonl')
aud = read_lines(OUT / 'audit_bank_manifest.jsonl')
assert len(cal) == len(aud) == 2048
calh = {r['text_sha256'] for r in cal}
audh = {r['text_sha256'] for r in aud}
assert len(calh) == len(audh) == 2048 and (not calh & audh) and (not old & (calh | audh))
for rows in (cal, aud):
    assert Counter((r['domain'] for r in rows)) == {d: 512 for d in DOMAINS}
    for d in DOMAINS:
        subset = [r for r in rows if r['domain'] == d]
        assert [r['rank_key'] for r in subset] == sorted((r['rank_key'] for r in subset))
        for r in subset:
            assert r['rank_key'] == hashlib.sha256(f"EXP-S1-v1-bank|{d}|{r['text_sha256']}".encode()).hexdigest()
for name, rows in (('calibration', cal), ('audit', aud)):
    pairs = read_lines(OUT / f'{name}_pair_manifest.jsonl')
    assert len(pairs) == 12288
    assert Counter(((r['source_domain'], r['target_domain']) for r in pairs)) == {(a, b): 1024 for a in DOMAINS for b in DOMAINS if a != b}
    h = {r['text_sha256'] for r in rows}
    assert all((r['source_text_sha256'] in h and r['target_text_sha256'] in h for r in pairs))
gate = torch.load(WORK / 's1_calibration_gate_256.pt', map_location='cpu', weights_only=True)
assert tuple(gate['input_ids'].shape) == (256, 256)
expected = [r['tokenized_sequence_sha256'] for d in DOMAINS for r in [x for x in cal if x['domain'] == d][:64]]
actual = [sequence_hash(ids, mask) for ids, mask in zip(gate['input_ids'].tolist(), gate['attention_mask'].tolist())]
assert actual == expected
train = json.loads((OUT / 'TRAINING_MANIFEST_HASHES.json').read_text())
for seed in range(2026100101, 2026100109):
    rows = read_lines(OUT / f'TRAIN_SEED_{seed}_MANIFEST.jsonl')
    tensor = torch.load(WORK / f'train_seed_{seed}.pt', map_location='cpu', weights_only=True)
    assert len(rows) == 1000 and tuple(tensor['input_ids'].shape) == (1000, 256)
    assert int(tensor['attention_mask'].sum()) == 256000
    assert sha(OUT / f'TRAIN_SEED_{seed}_MANIFEST.jsonl') == train['seeds'][str(seed)]['manifest_sha256']
    assert sha(WORK / f'train_seed_{seed}.pt') == train['seeds'][str(seed)]['tensor_sha256']
    for i, r in enumerate(rows):
        assert r['sequence_index'] == i and r['domain'] == DOMAINS[i % 4]
        assert r['sequence_sha256'] == sequence_hash(tensor['input_ids'][i].tolist(), tensor['attention_mask'][i].tolist())
        assert all((doc['text_sha256'] not in calh | audh | old for doc in r['source_documents']))
benchmark = torch.load(WORK / 's1_benchmark_100k.pt', map_location='cpu', weights_only=True)
assert int(benchmark['attention_mask'].sum()) == 100000
assert all((r['text_sha256'] not in calh | audh | old for r in read_lines(OUT / 'benchmark_source_manifest.jsonl')))
numerical = json.loads((OUT / 'NUMERICAL_GATE.json').read_text())
sector = json.loads((OUT / 'SECTOR_CONTROLS.json').read_text())
floor = json.loads((OUT / 'ZERO_FLOOR_RULE.json').read_text())
assert sector['calibration_sequences'] == 2048
assert sector['pair_manifest_sha256'] == sha(OUT / 'calibration_pair_manifest.jsonl')
assert sector['representation_tensor_sha256'] == sha(WORK / 's1_calibration_representation.pt')
smoke = json.loads((OUT / 'IMMUTABILITY_GATE.json').read_text())
runtime = json.loads((OUT / 'RUNTIME_BENCHMARK.json').read_text())
assert numerical['pass'] and sector['pass'] and floor['strictly_at_least_100x_largest_zero_power']
assert smoke['status'] == 'PASS' and runtime['status'] == 'PASS'
result = {'status': 'PASS', 'model_files_verified': len(assets['model_files']), 'dataset_train_shards_sha256_verified': 48, 'fresh_calibration_docs': 2048, 'fresh_audit_docs': 2048, 'old_and_new_holdout_overlap': 0, 'pair_rows_per_bank': 12288, 'calibration_gate_sequences_verified': 256, 'calibration_sector_sequences_verified': 2048, 'training_seeds_verified': 8, 'training_tokens_per_seed': 256000, 'benchmark_tokens_verified': 100000, 'numerical_gate': True, 'sector_gate': True, 'immutability_gate': True, 'runtime_gate': True}
(OUT / 'PREFLIGHT_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
