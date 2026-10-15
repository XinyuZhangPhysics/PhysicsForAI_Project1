# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from pathlib import Path
import hashlib, json, datetime, os
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs'
PKG = ROOT / 'work/Project1_P2H_FormationBridge_Codex_Package_v1'
MODEL = Path(_p1_path('path_1a97f575bc3c'))
DATA = Path(_p1_path('path_7d5464ba027c'))
HISTORY = Path(_p1_path('path_a27a23695440'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def main():
    authority = {}
    for line in (PKG / 'SHA256SUMS.txt').read_text().splitlines():
        expected, n = line.split(maxsplit=1)
        n = n.lstrip('*')
        actual = sha(PKG / n)
        assert actual == expected, n
        authority[n] = actual
    assert authority['Project1_P2G_MinimalJointMoE_Formation_Protocol_v1_FROZEN.md'] == 'ff252991f9373763472c151757edced32e9d0336a67f56eb66f70dbbcc8f6ca1'
    assert authority['P2G_PROTOCOL_FREEZE.json'] == '46eef38ee01cdf9b37e92bed2ecb20ca5a8bb11f0c9de5de9e5e7e9b5771e10b'
    ledger = json.loads((MODEL / 'MODEL_ASSET.json').read_text())
    models = []
    for p, h in ledger['file_hashes'].items():
        f = MODEL / Path(p).name
        a = sha(f)
        assert a == h, f
        models.append(dict(path=str(f), sha256=a, size=f.stat().st_size))
    treepath = DATA.parents[1] / 'trees' / f'{DATA.name}.json'
    tree = json.loads(treepath.read_text())
    inventory = []
    for rel, record in tree['files'].items():
        if rel.endswith('.parquet'):
            f = DATA / rel
            assert f.exists(), f
            inventory.append(dict(path=str(f), resolved_path=str(f.resolve()), size=f.stat().st_size, expected_sha256=record['lfs_sha256'], size_verified=f.stat().st_size == record['size']))
    shard = next((x for x in inventory if '/train-00000-' in x['path']))
    shard['sha256'] = sha(shard['path'])
    assert shard['sha256'] == shard['expected_sha256']
    sources = [dict(path=str(HISTORY / n), sha256=sha(HISTORY / n)) for n in ['j1_torch_backend.py', 'j1c_runner.py', 'materialize_j1_preflight.py']]
    result = dict(status='EXACT_LOCAL_PRIMARY_ASSETS_VERIFIED', timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), authority_sha256=authority, search_roots=[_p1_path('path_796298a2b5da'), _p1_path('path_27e4e1a80a62'), _p1_path('path_4f415242142d')], search_method='rg --files --hidden --no-ignore; canonical pinned paths, cache tree and historical metadata inspected', search_inventory_sha256=sha(ROOT / 'work/local_file_inventory.txt'), search_errors=[], model_repo=ledger['repo'], model_commit=ledger['revision'], model_dir=str(MODEL), model_files=models, dataset_repo='DKYoon/SlimPajama-6B', dataset_revision=DATA.name, dataset_tree_path=str(treepath), dataset_tree_sha256=sha(treepath), dataset_cached_shards=inventory, data_files_selected=[shard], historical_sources=sources, target_index=20, historical_actual_index=19, adaptation='Reuse g=1 SwiGLU full-copy forward mechanics; constrain all expert private weights to c_i M and router logits to c_i U^T h; follow current zero-based index 20', other_local_assets=[dict(repo='EleutherAI/pythia-70m', commit='a39f36b100fe8a5377810d56c3f4789b9c53ac42', path=_p1_path('path_82160b73a452'), role='inventory_only'), dict(repo='allenai/OLMoE-1B-7B-0924', commit='6d84c48581ece794365f2b8e9cfb043c68ade9c5', path=_p1_path('path_8ffb5e529ad6'), role='inventory_only')], MODEL_FILES_REUSED=len(models), MODEL_FILES_NEWLY_DOWNLOADED=0, DATA_FILES_REUSED=1, DATA_FILES_NEWLY_DOWNLOADED=0, BYTES_NEWLY_DOWNLOADED=0, large_asset_copies=0, training_started=False)
    (OUT / 'P2H_ASSET_REUSE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print('AUDIT PASSED', len(models), 'model files;', len(inventory), 'cached data shards; 1 selected and hash verified; 0 downloaded', flush=True)
if __name__ == '__main__':
    main()
