# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Independent pre-model bank gates with source-range replay."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
import build_banks_v1p2 as b
ROOT = Path(__file__).parent

def main():
    bank_raw = (ROOT / 'Formal_Banks_v1p2.json').read_bytes()
    bank = json.loads(bank_raw)
    shard_raw = (ROOT / 'Formal_Shard_Sampling_v1p2.json').read_bytes()
    shard = json.loads(shard_raw)
    tree_raw = (b.OLD / 'historical_tree.json').read_bytes()
    tree = json.loads(tree_raw)
    old_ids, old_texts, old_tokens, _ = b.load_exclusions()
    checks = {}
    checks['revision_exact'] = bank['dataset_revision'] == shard['dataset_revision'] == b.REV
    checks['model_sha_exact'] = bank['model_sha'] == b.MODEL_SHA
    checks['seven_domains_exact'] = list(bank['banks']) == list(shard['domains']) == b.DOMAINS
    checks['historical_tree_hash_exact'] = b.sha(tree_raw) == shard['historical_tree_sha256']
    checks['sampling_seal_hash_exact'] = b.sha(shard_raw) == bank['sampling_artifact_sha256']
    all_rows = [x for d in b.DOMAINS for n, _ in b.SIZES for x in bank['banks'][d][n]]
    checks['counts_exact'] = all(({n: len(bank['banks'][d][n]) for n, _ in b.SIZES} == dict(b.SIZES) for d in b.DOMAINS))
    checks['length_256'] = all((len(x['token_ids']) == 256 for x in all_rows))
    checks['source_disjoint'] = len({x['source_id'] for x in all_rows}) == len(all_rows)
    checks['text_disjoint'] = len({x['source_sha256'] for x in all_rows}) == len(all_rows)
    checks['token_disjoint'] = len({x['token_sha256'] for x in all_rows}) == len(all_rows)
    checks['pilot_extension_excluded'] = all((x['source_id'] not in old_ids and x['source_sha256'] not in old_texts and (x['token_sha256'] not in old_tokens) for x in all_rows))
    checks['stored_token_hashes'] = all((b.sha(json.dumps(x['token_ids'], separators=(',', ':')).encode()) == x['token_sha256'] for x in all_rows))
    all_ids, all_texts, all_tokens = (set(), set(), set())
    tokenizer = AutoTokenizer.from_pretrained(b.MODEL, revision=b.MODEL_SHA, local_files_only=True)
    replay = {}
    for domain in b.DOMAINS:
        files = [e for e in tree if e['path'].startswith(f'data/{domain}/') and 'size' in e]
        files.sort(key=lambda e: (b.rank(f"20260929|formal-v1p2|shard|{domain}|{e['path']}"), e['path']))
        expected_R = min(32, len(files))
        stats = shard['domains'][domain]
        assert stats['complete_frozen_shard_count'] == len(files)
        assert stats['selected_R'] == expected_R
        assert stats['selected_paths_in_hash_rank'] == [e['path'] for e in files[:expected_R]]
        q0, rem = divmod(352, expected_R)
        selected = []
        for j, entry in enumerate(files[:expected_R]):
            q = q0 + (j < rem)
            got, stat = b.choose_shard(entry, q, tokenizer, all_ids, all_texts, all_tokens, old_ids, old_texts, old_tokens)
            assert stat == stats['per_shard'][j], (domain, j, 'shard statistic mismatch')
            selected += got
            all_ids.update((x['source_id'] for x in got))
            all_texts.update((x['source_sha256'] for x in got))
            all_tokens.update((x['token_sha256'] for x in got))
            print('REPLAY', domain, j + 1, '/', expected_R, len(got), flush=True)
        next_unused = expected_R
        while len(selected) < 352:
            entry = files[next_unused]
            got, stat = b.choose_shard(entry, 352 - len(selected), tokenizer, all_ids, all_texts, all_tokens, old_ids, old_texts, old_tokens)
            stat['deficiency_fill'] = True
            assert stat == stats['per_shard'][next_unused]
            selected += got
            all_ids.update((x['source_id'] for x in got))
            all_texts.update((x['source_sha256'] for x in got))
            all_tokens.update((x['token_sha256'] for x in got))
            next_unused += 1
        selected.sort(key=lambda x: (b.rank('20260929|formal-v1p2|bank|' + x['source_id'] + '|' + x['token_sha256']), x['source_id']))
        actual = [x for name, _ in b.SIZES for x in bank['banks'][domain][name]]
        assert selected == actual, (domain, 'bank allocation / source replay mismatch')
        replay[domain] = {'replayed_shards': next_unused, 'replayed_documents': len(selected)}
    checks['multi_shard_rank_quota_and_source_replay'] = True
    star = [x for n, _ in b.SIZES for x in bank['banks']['starcoder'][n]]
    prefixes = {Path(x['source_shard']).name.rsplit('-', 1)[0] for x in star}
    checks['starcoder_multishard_multiprefix'] = len({x['source_shard'] for x in star}) > 1 and len(prefixes) > 1
    checks['bankwise_multishard'] = all((len({x['source_shard'] for x in bank['banks'][d][n]}) >= 2 for d in b.DOMAINS for n, _ in b.SIZES if shard['domains'][d]['selected_R'] > 1))
    result = {'schema': 'Bank_Replay_Verification_v1p2', 'passed': all(checks.values()), 'checks': checks, 'bank_manifest_sha256': b.sha(bank_raw), 'sampling_sha256': b.sha(shard_raw), 'replay': replay, 'starcoder_shard_count': len({x['source_shard'] for x in star}), 'starcoder_filename_prefixes': sorted(prefixes)}
    (ROOT / 'Bank_Replay_Verification_v1p2.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('pre-model bank verification failed')
    print('PASSED', result['bank_manifest_sha256'], flush=True)
if __name__ == '__main__':
    main()
