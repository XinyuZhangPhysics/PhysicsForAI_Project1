# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'work/model'
OLD = Path(_p1_path('path_5c4748c9a3ef'))
DATA = Path(_p1_path('path_7d5464ba027c'))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
expected = {'model_type': 'gpt_neox', 'num_hidden_layers': 6, 'hidden_size': 512, 'intermediate_size': 2048, 'num_attention_heads': 8, 'hidden_act': 'gelu', 'max_position_embeddings': 2048}
config = json.loads((MODEL / 'config.json').read_text())
assert all((config.get(k) == v for k, v in expected.items()))
model_files = {}
for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
    p = MODEL / name
    assert p.is_file()
    model_files[name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
old = json.loads(OLD.read_text())
assert old['dataset_revision'] == 'b5f90f419b7489cdba26fdbc8c022fcb5562f968'
dataset_files = {}
for name, record in old['dataset_train_files'].items():
    p = DATA / name
    assert p.is_file() and p.stat().st_size == record['bytes']
    actual = sha(p)
    assert actual == record['sha256'], name
    dataset_files[name] = {'sha256': actual, 'bytes': p.stat().st_size}
    print('verified', name, flush=True)
result = {'status': 'ASSETS_FROZEN', 'model_id': 'EleutherAI/pythia-70m', 'model_revision': 'a39f36b100fe8a5377810d56c3f4789b9c53ac42', 'model_config_verified': expected, 'model_files': model_files, 'dataset_id': 'DKYoon/SlimPajama-6B', 'dataset_revision': 'b5f90f419b7489cdba26fdbc8c022fcb5562f968', 'dataset_train_files': dataset_files, 'prior_asset_manifest_sha256': sha(OLD), 'dataset_train_shards_verified': len(dataset_files)}
(ROOT / 'outputs/ASSET_MANIFEST.json').write_text(json.dumps(result, indent=2) + '\n')
print('ASSETS_FROZEN', len(dataset_files), flush=True)
