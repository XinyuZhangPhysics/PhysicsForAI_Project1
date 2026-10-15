# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Tokenizer-only frozen commit certification. Never fetches weight files."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from pathlib import Path
from . import MODEL_REPO
from .binding import LEGACY_SHA, TARGET_SHA, CHECKS
from .reports import sha256, write_json
ASSETS = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model')
METADATA_FIELDS = ('_name_or_path', 'name_or_path', '_commit_hash', 'revision', 'transformers_version')
CORPUS = ('A small mixture of experts predicts the next token.', '中文与 café、naïve、Ω、数学 ∑ 的 NFC 文本。', 'Café and é: combining marks.', 'def square(x):\n    return x ** 2\n# café\n', 'x^2 + y^2 = z^2; \\int_0^1 f(x) dx; α ≤ β', '  leading\tspaces\n\ntrailing  ', '\r\nCRLF\rCR\nLF', '', '🧪🌍 zero\u200bwidth\xa0space')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode()

def semantic_asset(path):
    if path.suffix == '.json':
        data = json.loads(path.read_text())
        removed = {}
        if isinstance(data, dict):
            for key in METADATA_FIELDS:
                if key in data:
                    removed[key] = data.pop(key)
        return (hashlib.sha256(canonical(data)).hexdigest(), removed)
    return (sha256(path), {})

def certify(output, cache_dir, meta):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    snapshots = {}
    tokenizers = {}
    assets = {}
    api = HfApi()
    for label, revision in (('source', LEGACY_SHA), ('target', TARGET_SHA)):
        files = set(api.list_repo_files(MODEL_REPO, revision=revision))
        selected = sorted(files.intersection(ASSETS))
        if 'tokenizer.json' not in selected or 'tokenizer_config.json' not in selected:
            raise RuntimeError('Required tokenizer assets are missing.')
        folder = Path(snapshot_download(MODEL_REPO, revision=revision, allow_patterns=selected, cache_dir=cache_dir))
        snapshots[label] = folder
        tokenizers[label] = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        assets[label] = {}
        for name in selected:
            semantic, removed = semantic_asset(folder / name)
            assets[label][name] = dict(raw_sha256=sha256(folder / name), semantic_sha256=semantic, metadata_only_fields_removed=removed)
    a, b = (tokenizers['source'], tokenizers['target'])
    special_tokens = sorted(set(a.all_special_tokens) | set(b.all_special_tokens))
    corpus = list(CORPUS) + special_tokens + ['prefix ' + s + ' suffix' for s in special_tokens]
    encodings = []
    for text in corpus:
        encodings.append(dict(text=text, source=a.encode(text, add_special_tokens=False), target=b.encode(text, add_special_tokens=False)))

    def effective(t):
        return dict(tokenizer_class=type(t).__name__, model_max_length=t.model_max_length, padding_side=t.padding_side, truncation_side=t.truncation_side, clean_up_tokenization_spaces=t.clean_up_tokenization_spaces, special_tokens_map=t.special_tokens_map, backend=json.loads(t.backend_tokenizer.to_str()))
    ea, eb = (effective(a), effective(b))
    semantic_hashes = {label: {name: r['semantic_sha256'] for name, r in group.items()} for label, group in assets.items()}
    checks = dict(zip(CHECKS, (type(a).__name__ == type(b).__name__, ea == eb, len(a) == len(b) and a.vocab_size == b.vocab_size, a.get_vocab() == b.get_vocab(), a.special_tokens_map == b.special_tokens_map and a.all_special_ids == b.all_special_ids, semantic_hashes['source'] == semantic_hashes['target'], all((row['source'] == row['target'] for row in encodings)))))
    report = dict(scope='tokenizer_only_equivalence', source_revision_commit_sha=LEGACY_SHA, target_revision_commit_sha=TARGET_SHA, tokenizer_class={l: type(t).__name__ for l, t in tokenizers.items()}, vocabulary_size={l: dict(effective=len(t), base=t.vocab_size) for l, t in tokenizers.items()}, exact_vocabulary_token_id_equal=checks['exact_vocabulary_token_id_equal'], special_token_map_equal=checks['special_token_map_equal'], assets=assets, semantic_asset_hashes=semantic_hashes, effective_config_sha256={'source': hashlib.sha256(canonical(ea)).hexdigest(), 'target': hashlib.sha256(canonical(eb)).hexdigest()}, vocabulary_sha256={l: hashlib.sha256(canonical(t.get_vocab())).hexdigest() for l, t in tokenizers.items()}, special_token_maps={l: t.special_tokens_map for l, t in tokenizers.items()}, deterministic_test_corpus_encodings=encodings, declared_special_tokens_tested=special_tokens, checks=checks, TOKENIZER_EQUIVALENT=all(checks.values()), weight_files_downloaded=False, metadata_only_fields_allowed=list(METADATA_FIELDS))
    write_json(output, report, meta)
    return report
