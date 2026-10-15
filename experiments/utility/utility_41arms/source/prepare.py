# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Fresh immutable entry gate and metadata-only partition. Never decode candidate payloads."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import collections, inspect, shutil, zipfile
import numpy as np
import torch, transformers
from common import *

def main():
    assert not (BUNDLE / 'entry_gate_audit.json').exists(), 'FRESH_ENTRY_REQUIRED'
    checks = []

    def check(name, path, expected):
        row = dict(object=name, **info(path), expected_sha256=expected)
        row['match'] = row['sha256'] == expected
        checks.append(row)
        write(BUNDLE / 'entry_gate_audit.json', dict(status='IN_PROGRESS', checks=checks))
        if not row['match']:
            raise RuntimeError('ENTRY_HASH_MISMATCH:' + name)
    event('FRESH_EXECUTOR_ENTRY', model_loads=0, model_forwards=0, prior_Q3_state_imported=False)
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        check('supplied:' + name, ROOT / name.lstrip('*'), expected)
    check('protocol', ROOT / 'authorities/PROJECT1_V173_PROTOCOL_FREEZE.md', '0e66a4c0eed33121b55bedd681d8840f2594d424f5dcd4cd634c42280c93a55a')
    check('v175_adjudication', ROOT / 'PROJECT1_V175_TECHNICAL_STOP_ADJUDICATION.md', '7b6fd8578fee598007c1ea56a0853a7e91a457e973a0e15d809b41160928d2d3')
    check('amendment', ROOT / 'PROJECT1_V176_ACCESS_AMENDMENT.md', 'e332a2960c9a2e948f70804422337bafb2bead85744f9cd491f032d63b850877')
    q2 = PROJECTS / '2026-10-07/new-chat/outputs/Project1_v174_Q2_ReviewBundle.zip'
    old = PROJECTS / '2026-10-07/yan/outputs/Project1_v175_Q3_ReviewBundle.zip'
    check('q2_bundle', q2, '5ced800acc1f0298b4774d23ffed8fc52966ce57ed785c44b6d102d6efc512c2')
    check('v175_technical_stop_bundle', old, '3ce83f3db3aa7ec70ea3e5de326ea4b191d7ff26f8a786600efafd1ab1312770')
    with zipfile.ZipFile(old) as z:
        for name in ['entry_gate_audit.json', 'model_identity_audit.json', 'access_incident_audit.json']:
            (BUNDLE / 'historical_audits').mkdir(exist_ok=True)
            (BUNDLE / 'historical_audits' / name).write_bytes(z.read(name))
        identities = json.loads(z.read('model_identity_audit.json'))
    check('candidate_frame', FRAME, '1d11b58a49999d80c8e43ae788a0110df7d796e8c8b67a44346f7997198c42db')
    check('bank_v_manifest', BANK / 'Formal_Bank_V_v1.jsonl', 'c6e9e45422a0d364af651e68123f44868628cbebaebee2c8a5aad402df38dbda')
    hashes = {str(p.relative_to(BANK)): sha(p) for p in sorted((BANK / 'tokens').glob('*.json'))}
    root = digest(json.dumps(hashes, ensure_ascii=False, separators=(',', ':')).encode())
    assert len(hashes) == 448 and root == '47641b4a62c86a6bc7f43854272db2f555f0664dffa6363acd923301ead7788d', 'BANK_V_PAYLOAD_ROOT_MISMATCH'
    write(BUNDLE / 'bank_v_payload_hashes.json', hashes)
    controls = {7: ('6d662eae42dfcb9ef125b78eea0ae5c0d1b28b8a42720a20023eaca5b05e9e2b', 'f82602dfb19e6f313512d883801d18396c2118b2487549ea4b504f0234afa931', '03a36193c0c3fb8aa4a07cc63772f9d77383db8912aff228f5095f1b65b0dddb'), 15: ('92693a06992c94e9f7a8064dcafce8a43d366aa5c250f6253778cbf65a3ac505', '92ee2d764f0f994783f5254dd9a2815355e523ca533f630f1c58a8fee97d6a7b', '28f668994deb2d31e27d1e2b3f768e1b5d11ec582b6c1da4d6aaf15d46a59a7f')}
    for layer, (candidate, null, archive) in controls.items():
        p = NULLS / f'frozen_null_controls_layer_{layer}_v164.npz'
        check(f'null_npz_L{layer}', p, archive)
        with np.load(p, allow_pickle=False) as z:
            assert z['candidate'].shape == (128,) and z['null_controls'].shape == (256, 128)
            assert digest(np.ascontiguousarray(z['candidate']).tobytes()) == candidate, 'CANDIDATE_RAW_HASH_MISMATCH'
            assert digest(np.ascontiguousarray(z['null_controls']).tobytes()) == null, 'NULL_RAW_HASH_MISMATCH'
            assert np.array_equal(z['directions'], np.vstack([z['candidate'], z['null_controls']]))
        target = BUNDLE / 'controls' / p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, target)
    assert identities['commit_sha'] == MODEL_SHA
    for i, row in enumerate(identities['files']):
        check(f'41B_file_{i}', Path(row['path']), row['sha256'])
    assert SNAPSHOT.is_dir() and SNAPSHOT.name == MODEL_SHA
    cfg = read(SNAPSHOT / 'config.json')
    index = read(SNAPSHOT / 'model.safetensors.index.json')
    for name in ['config.json', 'model.safetensors.index.json'] + sorted(set(index['weight_map'].values())):
        p = SNAPSHOT / name
        matches = [r for r in identities['files'] if Path(r['path']).resolve() == p.resolve()]
        assert len(matches) == 1, 'SNAPSHOT_IDENTITY_FAILURE'
    assert all((cfg[k] == v for k, v in dict(num_hidden_layers=16, num_experts=64, num_experts_per_tok=8, hidden_size=2048, model_type='olmoe').items())), 'ARCHITECTURE_MISMATCH'
    write(BUNDLE / 'entry_gate_audit.json', dict(status='PASS', time_UTC=now(), checks=checks, bank_v_payload_root_sha256=root, model_sha=MODEL_SHA, model_loads=0, model_forwards=0))
    for p in [ROOT / 'authorities/PROJECT1_V173_PROTOCOL_FREEZE.md', ROOT / 'authorities/PROJECT1_V174_Q2_ADJUDICATION.md'] + list(ROOT.glob('*.md')):
        t = BUNDLE / 'authorities' / p.name
        t.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, t)
    shutil.copyfile(ROOT / 'SHA256SUMS.txt', BUNDLE / 'authorities/INPUT_SHA256SUMS.txt')
    event('ENTRY_GATE_PASS', audit_sha256=sha(BUNDLE / 'entry_gate_audit.json'))
    rows = [json.loads(line) for line in (ROOT / 'metadata/candidate_frame_metadata.jsonl').open()]
    bank = [json.loads(line) for line in (ROOT / 'metadata/bank_v_metadata.jsonl').open()]
    assert all((set(r) == {'document_id', 'domain', 'token_array_sha256'} for r in rows + bank))
    byid = {r['document_id']: r for r in rows}
    excluded = {r['document_id'] for r in bank}
    assert len(rows) == len(byid) == 1792 and len(bank) == len(excluded) == 448
    assert all((byid[r['document_id']] == r for r in bank))
    assert collections.Counter((r['domain'] for r in bank)) == {d: 64 for d in DOMAINS}
    actual = [json.loads(line) for line in (BANK / 'Formal_Bank_V_v1.jsonl').open()]
    assert {r['document_id']: {k: r[k] for k in ('document_id', 'domain', 'token_array_sha256')} for r in actual} == {r['document_id']: r for r in bank}
    parts = {name: [] for name in ('ADAPT', 'HELDOUT', 'RESERVE')}
    for domain in DOMAINS:
        unused = sorted([r for r in rows if r['domain'] == domain and r['document_id'] not in excluded], key=lambda r: r['token_array_sha256'])
        assert len(unused) == 192 and len({r['token_array_sha256'] for r in unused}) == 192
        for name, part in zip(parts, (unused[:32], unused[32:96], unused[96:])):
            parts[name].extend(part)
    manifests = {}
    for name, part in parts.items():
        p = BUNDLE / f'Q3_{name}_MANIFEST_v176.jsonl'
        p.write_bytes(b''.join((canonical(r) + b'\n' for r in part)))
        manifests[name] = dict(file=p.name, sha256=sha(p), documents=len(part), per_domain=dict(collections.Counter((r['domain'] for r in part))), token_hash_metadata_root_sha256=digest(canonical({r['document_id']: r['token_array_sha256'] for r in part})))
    sets = [{r['document_id'] for r in part} for part in parts.values()]
    assert list(map(len, sets)) == [224, 448, 672] and len(set.union(*sets)) == 1344 and (not set.union(*sets) & excluded)
    write(BUNDLE / 'bank_overlap_audit.json', dict(status='PASS', candidate_frame_documents=1792, excluded_bank_v_documents=448, unused=1344, zero_overlap=True, manifests=manifests))
    for p in (ROOT / 'metadata').glob('*.jsonl'):
        shutil.copyfile(p, BUNDLE / p.name)
    write(BUNDLE / 'runtime_identity.json', dict(python=sys.version, executable=sys.executable, torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, platform=platform.platform(), mps_built=torch.backends.mps.is_built(), mps_available=torch.backends.mps.is_available(), device='mps', stored_weights='BF16', attention='eager', MPS_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'), disk_free_bytes=shutil.disk_usage(ROOT).free))
    assert torch.backends.mps.is_available(), 'MPS_UNAVAILABLE'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1', 'MPS_FALLBACK_FORBIDDEN'
    write(BUNDLE / 'arm_manifest.json', dict(arms=ARMS, random_pairing='identical indices 0..38 at L7/L15', layers=[7, 15], initial_alphas=[0, 0], epochs=2, documents_per_epoch=224, optimizer_steps=56, heldout_documents_per_arm=448))
    event('METADATA_PARTITION_FROZEN', manifests=manifests)
    print('ENTRY PASS; metadata partitions 224 / 448 / 672; no Q3 payload decoded', flush=True)
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        write(BUNDLE / 'technical_stop_trigger.json', dict(stage='A_OR_B', exception_class=type(e).__name__, reason=str(e) if isinstance(e, (AssertionError, RuntimeError)) else 'ENTRY_GATE_OPERATION_FAILURE', time_UTC=now()))
        event('TECHNICAL_STOP', stage='A_OR_B', exception_class=type(e).__name__)
        print('TECHNICAL STOP: entry gate failure; see saved audit', flush=True)
        sys.exit(2)
