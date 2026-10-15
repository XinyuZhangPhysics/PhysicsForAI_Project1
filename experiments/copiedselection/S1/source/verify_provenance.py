# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
WORK = ROOT / 'work'
SRC = WORK / 'scientific/PhysicsForAI_EXP_S1_Scientific'
TARGET = OUT / 'exp_s1'
TARGET.mkdir(parents=True, exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
pre = json.loads((OUT / 'EXP_S1_PREFLIGHT.json').read_text())
assert pre['status'] == 'EXP_S1_PREFLIGHT_PASS'
for name in ('EXP_S1_PREFLIGHT.json', 'EXP_S1_PREFLIGHT_REPORT.md'):
    assert sha(SRC / name) == sha(OUT / name)
for name in ('PhysicsForAI_Project1_EXP_S1_IMPLEMENTATION_SPEC_v1.yaml', 'PhysicsForAI_Project1_EXP_S1_PREREGISTRATION_FROZEN_v1.md'):
    assert sha(SRC / name) == pre['source_documents_sha256'][name]
assert '0.02' in (SRC / 'PhysicsForAI_Project1_EXP_S1_ROUTER_INIT_CLARIFICATION_v1.md').read_text()
assets = json.loads((OUT / 'ASSET_MANIFEST.json').read_text())
data = json.loads((OUT / 'DATA_MANIFEST_HASHES.json').read_text())
train = json.loads((OUT / 'TRAINING_MANIFEST_HASHES.json').read_text())
assert sha(OUT / 'ASSET_MANIFEST.json') == pre['asset_manifest_sha256']
assert sha(OUT / 'DATA_MANIFEST_HASHES.json') == pre['data_manifest_hashes_sha256']
assert sha(OUT / 'TRAINING_MANIFEST_HASHES.json') == pre['training_manifest_hashes_sha256']
assert sha(OUT / 'ROUTER_INITIALIZATION.json') == pre['router_initialization_manifest_sha256']
assert assets['model_revision'] == pre['model_revision']
assert assets['dataset_revision'] == pre['dataset_revision']
for name, record in assets['model_files'].items():
    assert sha(WORK / 'model' / name) == record['sha256']
dataset = Path(_p1_path('path_7d5464ba027c'))
for name, record in assets['dataset_train_files'].items():
    assert sha(dataset / name) == record['sha256']
for key, name in [('calibration', 'calibration_bank_manifest.jsonl'), ('audit', 'audit_bank_manifest.jsonl'), ('calibration_pairs', 'calibration_pair_manifest.jsonl'), ('audit_pairs', 'audit_pair_manifest.jsonl'), ('training_pool_compressed', 'training_pool_manifest.jsonl.zst'), ('benchmark_source', 'benchmark_source_manifest.jsonl')]:
    assert sha(OUT / name) == data['manifest_sha256'][key]
for seed in range(2026100101, 2026100109):
    item = train['seeds'][str(seed)]
    assert sha(OUT / f'TRAIN_SEED_{seed}_MANIFEST.jsonl') == item['manifest_sha256']
    assert sha(WORK / f'train_seed_{seed}.pt') == item['tensor_sha256']
for name, expected in pre['code_hashes'].items():
    assert sha(WORK / name) == expected
assert pre['scientific_runs_started'] == 0 and pre['audit_scientific_response_observables_opened'] == 0
result = {'status': 'EXP_S1_PROVENANCE_VERIFIED', 'preflight_json_sha256': sha(OUT / 'EXP_S1_PREFLIGHT.json'), 'scientific_archive_sha256': sha(Path(_p1_path('path_b2b55db8f49e'))), 'source_handoff_sha256': sha(SRC / 'PhysicsForAI_Project1_EXP_S1_SCIENTIFIC_CODEX_HANDOFF.md'), 'router_init_clarification_sha256': sha(SRC / 'PhysicsForAI_Project1_EXP_S1_ROUTER_INIT_CLARIFICATION_v1.md'), 'model_files_sha256_verified': len(assets['model_files']), 'dataset_train_shards_sha256_verified': len(assets['dataset_train_files']), 'training_seed_streams_verified': 8, 'preflight_code_files_verified': len(pre['code_hashes']), 'audit_scientific_responses_opened': False}
(TARGET / 'PROVENANCE_PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
