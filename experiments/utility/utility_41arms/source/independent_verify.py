# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Portable offline verifier. Does not import the scientific runner or access payloads/models."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import collections, csv, hashlib, json, math, sys
from pathlib import Path
DOMAINS = ('dclm', 'starcoder', 'pes2o', 'arxiv', 'open-web-math', 'algebraic-stack', 'wiki')
ARMS = ['BASELINE', 'CANDIDATE'] + [f'RANDOM_{i:03d}' for i in range(39)]

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def digest(x):
    return hashlib.sha256(x).hexdigest()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def rows(p):
    if not Path(p).exists():
        return []
    with Path(p).open() as f:
        return [json.loads(line) for line in f]

def require(value, reason):
    if not value:
        raise RuntimeError(reason)

def chain(p):
    values = rows(p)
    prev = '0' * 64
    for i, row in enumerate(values):
        payload = dict(row)
        h = payload.pop('event_sha256')
        require(payload['sequence'] == i and payload['previous_event_sha256'] == prev and (digest(canonical(payload)) == h), 'EVENT_CHAIN_FAILURE')
        require('token_ids' not in payload, 'PAYLOAD_IN_LOG')
        prev = h
    return values

def verify(root, verify_manifest=True):
    root = Path(root)
    checks = []
    if verify_manifest:
        declared = {}
        for line in (root / 'SHA256SUMS_REVIEW.txt').read_text().splitlines():
            value, name = line.split('  ', 1)
            p = root / name
            require(p.is_file() and sha(p) == value, 'BUNDLE_MEMBER_HASH_FAILURE:' + name)
            declared[name] = value
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS_REVIEW.txt'}
        require(actual == set(declared), 'UNDECLARED_OR_MISSING_BUNDLE_MEMBER')
        checks.append('all bundle payload members match SHA256SUMS_REVIEW.txt')
    entry = read(root / 'entry_gate_audit.json')
    require(entry['status'] == 'PASS' and all((r['match'] and r['sha256'] == r['expected_sha256'] for r in entry['checks'])), 'ENTRY_GATE_NOT_PASS')
    require(entry['model_sha'] == 'e5642d2d7b37c138775ddf59f685f91788e173b7', 'MODEL_IDENTITY_FAILURE')
    for r in entry['checks']:
        if r['object'].startswith('null_npz_L'):
            require(sha(root / 'controls' / Path(r['path']).name) == r['expected_sha256'], 'CONTROL_ARCHIVE_IDENTITY_FAILURE')
    payload_hashes = read(root / 'bank_v_payload_hashes.json')
    payload_root = digest(json.dumps(payload_hashes, ensure_ascii=False, separators=(',', ':')).encode())
    require(len(payload_hashes) == 448 and payload_root == '47641b4a62c86a6bc7f43854272db2f555f0664dffa6363acd923301ead7788d', 'BANK_V_PAYLOAD_ROOT_FAILURE')
    checks.append('immutable entry identities passed; bulky sources referenced by exact hashes')
    frame = rows(root / 'candidate_frame_metadata.jsonl')
    bank = rows(root / 'bank_v_metadata.jsonl')
    require(sha(root / 'candidate_frame_metadata.jsonl') == '6252dfeac16c0288995a80a11534db8e1277d78dfbe32b58bae14b57da74e479', 'FRAME_METADATA_HASH')
    require(sha(root / 'bank_v_metadata.jsonl') == '956dea46356fecf75a32c2b4405c51819e6bf8e4d6b6c142cabea4ef7ecd9384', 'BANK_METADATA_HASH')
    require(len(frame) == 1792 and len({r['document_id'] for r in frame}) == 1792 and (len(bank) == 448), 'METADATA_COUNTS')
    excluded = {r['document_id'] for r in bank}
    expected = {b: [] for b in ['ADAPT', 'HELDOUT', 'RESERVE']}
    for d in DOMAINS:
        unused = sorted([r for r in frame if r['domain'] == d and r['document_id'] not in excluded], key=lambda r: r['token_array_sha256'])
        require(len(unused) == 192, 'UNUSED_DOMAIN_COUNT')
        for b, part in zip(expected, [unused[:32], unused[32:96], unused[96:]]):
            expected[b].extend(part)
    freeze = read(root / 'Q3_SCIENTIFIC_FREEZE_v176.json')
    fh = sha(root / 'Q3_SCIENTIFIC_FREEZE_v176.json')
    require(fh == (root / 'Q3_SCIENTIFIC_FREEZE_v176.sha256').read_text().split()[0], 'FREEZE_HASH_FAILURE')
    require(sha(root / 'authorities/PROJECT1_V173_PROTOCOL_FREEZE.md') == freeze['scientific_protocol_sha256'] == '0e66a4c0eed33121b55bedd681d8840f2594d424f5dcd4cd634c42280c93a55a', 'PROTOCOL_AUTHORITY_IDENTITY_FAILURE')
    require(sha(root / 'authorities/PROJECT1_V176_ACCESS_AMENDMENT.md') == freeze['amendment_sha256'] == 'e332a2960c9a2e948f70804422337bafb2bead85744f9cd491f032d63b850877', 'AMENDMENT_AUTHORITY_IDENTITY_FAILURE')
    require(freeze['arms'] == ARMS and freeze['scientific_outcomes_generated_at_seal'] == 0 and (freeze['Q3_model_forwards_at_seal'] == 0), 'FREEZE_PROSPECTIVE_FAILURE')
    require(freeze['V175_RAW_CANDIDATE_FRAME_ACCESS_OCCURRED'] is True and freeze['V175_HELDOUT_FIRST_ROW_DISPLAYED'] is True and (freeze['V175_RESERVE_BYTES_TRAVERSED'] is True) and (freeze['V175_MODEL_FORWARDS'] == 0) and (freeze['V175_Q3_OUTCOMES_GENERATED'] is False), 'HISTORICAL_DISCLOSURE_FAILURE')
    require('Q3_RESERVE_UNOPENED' not in freeze, 'RETIRED_FIELD_RECREATED')
    for b, spec in freeze['manifests'].items():
        actual = rows(root / spec['file'])
        require(actual == expected[b] and sha(root / spec['file']) == spec['sha256'], 'PARTITION_MEMBERSHIP_FAILURE:' + b)
        require(all((set(r) == {'document_id', 'domain', 'token_array_sha256'} for r in actual)), 'PAYLOAD_IN_MANIFEST')
    require([len(expected[b]) for b in expected] == [224, 448, 672], 'PARTITION_TOTALS')
    require(sha(root / 'blind_byte_index.json') == freeze['blind_index_sha256'], 'INDEX_HASH_FAILURE')
    index = read(root / 'blind_byte_index.json')
    require(index['payloads_decoded'] == 0 and len(index['rows']) == 1792, 'INDEX_CONTRACT_FAILURE')
    require(sha(root / 'prospective_source_provenance.json') == freeze['prospective_source_sha256'], 'PROSPECTIVE_SOURCE_AUDIT_HASH')
    for name, h in read(root / 'prospective_source_provenance.json')['source_hashes'].items():
        if isinstance(h, str):
            require(sha(root / 'source' / name) == h, 'SCIENTIFIC_SOURCE_CHANGED:' + name)
        else:
            require(sha(root / 'runtime_source/modeling_olmoe.py') == h['sha256'], 'TRANSFORMERS_SOURCE_CHANGED')
    checks.append('prospective freeze, historical disclosures, exact partitions and frozen source hashes')
    events = chain(root / 'execution_events.jsonl')
    accesses = chain(root / 'access_events.jsonl')
    names = [r['event'] for r in events]
    require(names.index('SCIENTIFIC_FREEZE_SEALED') < names.index('G112_PREFLIGHT_START'), 'FREEZE_ORDER_FAILURE')
    seals = {r['arm']: r for r in events if r['event'] == 'ARM_ADAPTER_SEALED'}
    permitted = [r for r in accesses if r['result'] == 'PERMITTED']
    for r in permitted:
        require(r['arm'] in ARMS and r['bank'] != 'RESERVE', 'FORBIDDEN_ACCESS')
        if r['bank'] == 'HELDOUT':
            require(r['access_class'] == 'evaluation' and r['arm'] in seals and (r['time_UTC'] >= seals[r['arm']]['time_UTC']), 'PREMATURE_HELDOUT_ACCESS')
        else:
            require(r['bank'] == 'ADAPT' and r['access_class'] == 'adaptation', 'WRONG_ADAPT_ACCESS_CLASS')
    require(read(root / 'access_firewall_preflight_audit.json')['real_payload_decodes'] == 0, 'FIREWALL_PREFLIGHT_PAYLOAD_ACCESS')
    checks.append('execution/access hash chains; reserve forbidden; heldout follows per-arm seal')
    completed = []
    for arm in ARMS:
        folder = root / 'arms' / arm
        if not (folder / 'completion_receipt.json').exists():
            continue
        r = read(folder / 'adaptation_receipt.json')
        cr = read(folder / 'completion_receipt.json')
        require(sha(folder / 'adaptation_receipt.json') == (folder / 'adaptation_receipt.sha256').read_text().split()[0], 'ADAPT_RECEIPT_HASH')
        require(sha(folder / 'completion_receipt.json') == (folder / 'completion_receipt.sha256').read_text().split()[0], 'COMPLETION_RECEIPT_HASH')
        require(r['arm'] == arm and r['scientific_freeze_sha256'] == fh and (r['state_sha256'] == sha(folder / 'final_alpha.json')) and (r['status'] == 'SEALED_FINAL'), 'STATE_SEAL_FAILURE')
        require(r['epochs_completed'] == 2 and r['adapt_documents_processed'] == 448 and (r['optimizer_steps'] == 56), 'ADAPT_BUDGET_FAILURE')
        require(all((r[k] == 0 and cr[k] == 0 for k in ['base_parameter_updates', 'base_parameters_with_grad', 'base_trainable_parameters'])), 'BASE_MUTATION')
        require(cr['nll_table_sha256'] == sha(folder / 'heldout_nll.json') and cr['adaptation_receipt_sha256'] == sha(folder / 'adaptation_receipt.json'), 'COMPLETION_ARTIFACT_HASH')
        require(cr['scientific_freeze_sha256'] == fh and cr['arm'] == arm and (cr['status'] == 'COMPLETE'), 'COMPLETION_BINDING')
        nll = read(folder / 'heldout_nll.json')
        require(len(nll) == 448 and {r['document_id'] for r in nll} == {r['document_id'] for r in expected['HELDOUT']}, 'HELDOUT_TABLE_MEMBERSHIP')
        require(all((r['arm'] == arm and r['predicted_tokens'] == 255 for r in nll)), 'NLL_CONTRACT')
        for b, times in [('ADAPT', 2), ('HELDOUT', 1)]:
            counts = collections.Counter((r['document_id'] for r in permitted if r['arm'] == arm and r['bank'] == b))
            require(counts == {r['document_id']: times for r in expected[b]}, 'ACCESS_MULTIPLICITY_FAILURE')
        bydomain = {d: sorted([r for r in expected['ADAPT'] if r['domain'] == d], key=lambda r: r['token_array_sha256']) for d in DOMAINS}
        expected_order = [bydomain[d][i]['document_id'] for i in range(32) for d in DOMAINS] * 2
        actual_order = [r['document_id'] for r in permitted if r['arm'] == arm and r['bank'] == 'ADAPT']
        require(actual_order == expected_order, 'TWO_EPOCH_ROUND_ROBIN_ORDER_FAILURE')
        state = read(folder / 'final_alpha.json')
        require(set(state) == {'arm', 'alpha7', 'alpha15'} and state['arm'] == arm and all((math.isfinite(state[k]) and abs(state[k]) <= 0.020000001 for k in ['alpha7', 'alpha15'])), 'ALPHA_STATE_OR_BOUND_FAILURE')
        if arm == 'BASELINE':
            require(state['alpha7'] == state['alpha15'] == 0, 'BASELINE_DUMMY_STATE_FAILURE')
        completed.append(arm)
    is_stop = (root / 'technical_stop_trigger.json').exists()
    if is_stop:
        require(not (root / 'endpoint.json').exists(), 'SCIENCE_INTERPRETATION_AFTER_STOP')
        require('TECHNICAL_STOP' in names, 'STOP_EVENT_ABSENT')
        checks.append('technical stop preserved; no endpoint or bootstrap interpretation')
    else:
        require(completed == ARMS, '41_ARM_COMPLETION_FAILURE')
        require(read(root / 'engineering_preflight_audit.json')['status'] == 'PASS', 'ENGINEERING_NOT_PASS')
        require(names.index('G112_PREFLIGHT_PASS') < names.index('ARM_START'), 'SCIENCE_BEFORE_PREFLIGHT')
        require(names.count('ENDPOINT_SEALED') == 1, 'ENDPOINT_EXECUTION_COUNT')
        import numpy as np
        raw_expected = {7: ('6d662eae42dfcb9ef125b78eea0ae5c0d1b28b8a42720a20023eaca5b05e9e2b', 'f82602dfb19e6f313512d883801d18396c2118b2487549ea4b504f0234afa931'), 15: ('92693a06992c94e9f7a8064dcafce8a43d366aa5c250f6253778cbf65a3ac505', '92ee2d764f0f994783f5254dd9a2815355e523ca533f630f1c58a8fee97d6a7b')}
        for l, (candidate_hash, null_hash) in raw_expected.items():
            with np.load(root / 'controls' / f'frozen_null_controls_layer_{l}_v164.npz', allow_pickle=False) as z:
                require(z['candidate'].shape == (128,) and z['null_controls'].shape == (256, 128), 'RAW_CONTROL_SHAPE')
                require(digest(np.ascontiguousarray(z['candidate']).tobytes()) == candidate_hash and digest(np.ascontiguousarray(z['null_controls']).tobytes()) == null_hash, 'RAW_CONTROL_HASH_IDENTITY_FAILURE')
                require(np.array_equal(z['directions'], np.vstack([z['candidate'], z['null_controls']])), 'CONTROL_ORIGINAL_ORDER_FAILURE')
        endpoint = read(root / 'endpoint.json')
        summary = read(root / 'per_arm_summary.json')
        sealed_event = next((r for r in events if r['event'] == 'ENDPOINT_SEALED'))
        require(sealed_event['endpoint_sha256'] == sha(root / 'endpoint.json') and sealed_event['bootstrap_sha256'] == sha(root / 'bootstrap_replicates.npy'), 'ENDPOINT_OR_BOOTSTRAP_SEAL_FAILURE')
        require(endpoint['seed_string'] == freeze['bootstrap']['seed_string'] and endpoint['replicates'] == 10000 and (endpoint['seed_integer'] == int.from_bytes(hashlib.sha256(b'PROJECT1-Q3-BOOTSTRAP-v173').digest()[:8], 'little')), 'BOOTSTRAP_FROZEN_IDENTITY_FAILURE')
        with (root / 'heldout_nll.csv').open(newline='') as f:
            combined = list(csv.DictReader(f))
        expected_combined = {(r['arm'], r['document_id']): r for a in ARMS for r in read(root / 'arms' / a / 'heldout_nll.json')}
        require(len(combined) == len(expected_combined) == 18368 and len({(r['arm'], r['document_id']) for r in combined}) == 18368, 'COMBINED_NLL_TABLE_COUNTS_FAILURE')
        for r in combined:
            actual = expected_combined[r['arm'], r['document_id']]
            require(r['domain'] == actual['domain'] and float(r['nll']) == actual['nll'] and (int(r['predicted_tokens']) == 255), 'COMBINED_NLL_TABLE_VALUE_FAILURE')
        table_event = next((r for r in events if r['event'] == 'ALL_41_ARMS_COMPLETE'))
        require(table_event['table_sha256'] == sha(root / 'heldout_nll.csv'), 'COMBINED_NLL_TABLE_SEAL_FAILURE')
        L = []
        for arm in ARMS:
            table = read(root / 'arms' / arm / 'heldout_nll.json')
            require(all((np.isfinite(r['nll']) for r in table)), 'NONFINITE_NLL_TABLE')
            L.append(np.mean([np.mean([r['nll'] for r in table if r['domain'] == d]) for d in DOMAINS]))
        L = np.asarray(L)
        U = L[0] - L
        D = float(U[1] - np.mean(U[2:]))
        require(np.isclose(endpoint['D'], D, rtol=0, atol=1e-12), 'D_RECOMPUTATION_FAILURE')
        require([r['arm'] for r in summary] == ARMS and np.allclose([r['L'] for r in summary], L, rtol=0, atol=1e-12), 'ARM_SUMMARY_FAILURE')
        require(np.allclose(list(endpoint['random_utilities'].values()), [U[ARMS.index(a)] for a in endpoint['random_utilities']], rtol=0, atol=1e-12), 'RANDOM_UTILITIES_FAILURE')
        boot = np.load(root / 'bootstrap_replicates.npy', allow_pickle=False)
        require(boot.shape == (10000, 2) and np.isfinite(boot).all(), 'BOOTSTRAP_ARTIFACT_FAILURE')
        ci = np.quantile(boot, [0.025, 0.975], axis=0, method='linear')
        require(np.array_equal(ci[:, 0], endpoint['D_CI95']) and np.array_equal(ci[:, 1], endpoint['candidate_vs_baseline_CI95']), 'BOOTSTRAP_CI_FAILURE')
        label = 'BENEFICIAL_RELATIVE_TO_RANDOM' if ci[0, 0] > 0 else 'HARMFUL_RELATIVE_TO_RANDOM' if ci[1, 0] < 0 else 'NULL_OR_UNRESOLVED'
        require(endpoint['primary_label'] == label and endpoint['Q3_COMPLETE'] is True, 'LABEL_FAILURE')
        require(endpoint['candidate_rank'] == 1 + int(np.count_nonzero(U[2:] > U[1])), 'RANK_FAILURE')
        require(endpoint['lower_inclusive_matched_null_percentile'] == float(np.mean(U[2:] <= U[1])) and endpoint['upper_inclusive_matched_null_percentile'] == float(np.mean(U[2:] >= U[1])), 'PERCENTILE_FAILURE')
        checks.append('41 x 448 NLL values, all receipts, domain-balanced endpoint, stored bootstrap CIs and primary label')
    return dict(status='PASS', bundle_type='TECHNICAL_STOP' if is_stop else 'Q3_COMPLETE', checks=checks, completed_arms=len(completed), expected_arms=41, permitted_ADAPT_decodes=sum((r['bank'] == 'ADAPT' for r in permitted)), permitted_HELDOUT_decodes=sum((r['bank'] == 'HELDOUT' for r in permitted)), permitted_RESERVE_decodes=0, denied_access_events=sum((r['result'] == 'DENIED' for r in accesses)), payloads_or_models_opened_by_verifier=0, bootstrap_regenerated_by_verifier=False)
if __name__ == '__main__':
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    print(json.dumps(verify(root), indent=2))
