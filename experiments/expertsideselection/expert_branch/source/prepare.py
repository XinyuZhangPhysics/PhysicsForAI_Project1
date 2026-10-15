# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from core import *
import shutil, subprocess
assert not (OUT / 'EXPERT_SIDE_SELECTION_RESULTS_v284M.json').exists()
entries = {}

def bind(p, expected=None, role='source'):
    p = Path(p)
    h = sha(p)
    if expected is not None:
        assert h == expected, (str(p), 'hash mismatch')
    entries[str(p)] = {'path': str(p), 'sha256': h, 'bytes': p.stat().st_size, 'role': role}
    return h
old = read(OLD / 'FULL_TANGENT_SOURCE_MANIFEST_v283M.json')
for x in old['entries']:
    if any((k in x['role'] for k in ['model', 'construction', 'provenance', 'bank', 'pairs'])):
        bind(x['path'], x['actual_sha256'], x['role'])
for n in ['EXPERT_BLOCK_BINDING_v283M.json', 'FULL_TANGENT_IMPLEMENTATION_CHECK_v283M.json', 'FULL_TANGENT_PREREGISTRATION_v283M.md', 'FULL_TANGENT_RESULTS_v283M.json', 'OUTPUT_HASHES_v283M.sha256']:
    bind(OLD / n, role='sealed_full_tangent')
for rel in ['common.py', 'kernels.py']:
    bind(BASE / 'work/source' / rel, role='copied_sealed_implementation')
for p in (R2 / 'source').glob('*.py'):
    bind(p, role='r2_source')
b = read(R2 / 'preregistration/INPUT_BINDINGS.json')
bind(R2 / 'preregistration/INPUT_BINDINGS.json', role='r2_bindings')
for x in b['input_hashes'] if isinstance(b['input_hashes'], list) else []:
    pass
points = []
for f in ['S1', 'S2']:
    for g in [1, 2, 4, 8]:
        for seed in range(2026100101 if f == 'S1' else 2026110201, 2026100109 if f == 'S1' else 2026110209):
            ref = next((x for x in b['copied_points'] if (x['family'], x['g'], x['seed']) == (f, 2 if g == 1 else g, seed)))
            p = Path(ref['path']) if g != 1 else Path(str(ref['path']).replace(f'S2_seed_{seed}', f'S1_seed_{seed}'))
            h = bind(p, ref['sha256'] if g != 1 else None, 'copied_router_0')
            w = load_file(str(p))['router.weight'].double().numpy()
            d = DIM[f][0]
            assert w.shape == (4 * g, d)
            assert np.array_equal(w, w.reshape(4, g, d)[:, 0].repeat(g, 0))
            if g == 1:
                refw = load_file(ref['path'])['router.weight'].double().numpy().reshape(4, 2, d)[:, 0]
                assert np.array_equal(w, refw)
            points.append({'family': f, 'g': g, 'seed': seed, 'path': str(p), 'sha256': h, 'child_duplication_exact': True, 'g1_tensor_matches_same_seed_g2_parent': g == 1})
banks = []
for phase in ['calibration', 'audit']:
    for fi, f in enumerate(['S1', 'S2']):
        meta = read(R2 / phase / f'{f}_BANK_BUILD.json')
        bind(R2 / phase / f'{f}_BANK_BUILD.json', role='r2_bank_metadata')
        item = meta['geometry']['2']
        bind(item['path'], item['sha256'], 'r2_representation_geometry')
        orig = np.load(item['path'])
        doc = b['fresh_banks'][f + '_' + phase]
        bind(doc['path'], doc['sha256'], 'reused_r2_document_identities')
        rows = [json.loads(l) for l in Path(doc['path']).read_text().splitlines()]
        assert len(rows) == 2048 and all((rows[i]['domain'] == DOMAIN[i // 512] for i in range(2048)))
        pp = R2 / phase / f'{f}_pairs.jsonl'
        bind(pp, meta['pair_sha256'], 'prior_r2_pairs')
        prior = set(zip(orig['src'].tolist(), orig['tgt'].tolist()))
        rng, key = rngkey('fresh-pairs', phase, f)
        src = []
        tgt = []
        group = []
        pairs = []
        for a in range(4):
            for c in range(4):
                if a == c:
                    continue
                chosen = []
                for pos in rng.permutation(512 * 512):
                    x = a * 512 + int(pos) // 512
                    y = c * 512 + int(pos) % 512
                    if (x, y) not in prior:
                        chosen.append((x, y))
                    if len(chosen) == 1024:
                        break
                gi = len(src) // 1024
                for i, (x, y) in enumerate(chosen):
                    src.append(x)
                    tgt.append(y)
                    group.append(gi)
                    pairs.append({'source_index': x, 'target_index': y, 'source_train_row_id': rows[x]['train_row_id'], 'target_train_row_id': rows[y]['train_row_id'], 'source_domain': DOMAIN[a], 'target_domain': DOMAIN[c], 'coordinate': gi, 'pair_index': i})
        assert not prior.intersection(zip(src, tgt)) and len(set(zip(src, tgt))) == 12288
        path = OUT / 'banks' / f'{f}_{phase}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, h=orig['h'], src=np.array(src), tgt=np.array(tgt), group=np.array(group))
        pairfile = path.with_suffix('.jsonl')
        pairfile.write_text(''.join((json.dumps(x, sort_keys=True) + '\n' for x in pairs)))
        shutil.copy2(doc['path'], path.parent / f'{f}_{phase}_documents.jsonl')
        banks.append({'family': f, 'phase': phase, 'path': str(path), 'sha256': sha(path), 'pair_manifest_sha256': sha(pairfile), 'pair_rng_key': key, 'documents_reused': True, 'fresh_pairs': 12288, 'overlap_with_r2_pairs': 0, 'identities_source_sha256': doc['sha256'], 'source_representation_sha256': item['sha256']})
for line in (OLD / 'OUTPUT_HASHES_v283M.sha256').read_text().splitlines():
    expected, rel = line.split(maxsplit=1)
    p = OLD / rel
    assert sha(p) == expected
seal_before = {str(p): sha(p) for p in OLD.rglob('*') if p.is_file()}
write('HISTORICAL_SEAL_BEFORE_v284M.json', seal_before)
shutil.copy2(OLD / 'EXPERT_BLOCK_BINDING_v283M.json', OUT / 'EXPERT_BLOCK_BINDING_v283M_REFERENCE.json')
for p in (BASE / 'work/input').iterdir():
    bind(p, role='user_taskpack')
    shutil.copy2(p, OUT / p.name)
dirs = []
for point in points:
    for r in range(4):
        v = direction(point['family'], point['g'], point['seed'], r)
        dirs.append({k: point[k] for k in ['family', 'g', 'seed']} | {'direction': r, 'rng_key': v['key'], 'factor_sha256': {k: hashlib.sha256(v[k].tobytes()).hexdigest() for k in ['a', 'b', 'b2', 'e', 'bias', 'parent', 'child']}, 'parameter_norm_per_expert': 1.0, 'whole_direction_norm': 1.0})
write('DIRECTION_BINDINGS_v284M.json', dirs)
write('EXPERT_SIDE_SELECTION_BINDINGS_v284M.json', {'timestamp_before_outcomes': stamp(), 'entries': list(entries.values()), 'copied_points': points, 'banks': banks, 'model_revisions': {'S1': 'a39f36b100fe8a5377810d56c3f4789b9c53ac42', 'S2': '28e66ca6931668447a3bac213f23d990ad3b0e2b'}, 'declared_metric': 'I/m; isotropic output trace 1/d', 'source_expert_block_binding_sha256': sha(OLD / 'EXPERT_BLOCK_BINDING_v283M.json'), 'background': 'logits s*(h W_parent^T - mean_parent), repeated over children; P_ref=Pq(s=1) frozen for both s=0,1', 'preregistered_seeds': 'all 8 per family; same seed indices paired across g', 'freshness': 'New response-blind pairs; previously used r2 document identities/representations. No expert-side response selected or read.'})
write('ENVIRONMENT_v284M.json', {'timestamp': stamp(), 'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'platform': platform.platform(), 'device': 'CPU float64; source representations float32 promoted exactly', 'threads': torch.get_num_threads()})
print('BOUND', len(points), 'copied points;', len(banks), 'fresh pair banks;', len(entries), 'sources', flush=True)
