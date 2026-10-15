# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Tokenize the 2048-document preflight-only response bank."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from build_banks import filehash, read_selected_texts

def main(manifest: Path, data: Path, model_dir: Path, out: Path) -> None:
    start = time.perf_counter()
    rows = [json.loads(line) for line in manifest.open()]
    assert len(rows) == 2048 and all((r['bank'] == 'preflight_benchmark' for r in rows))
    texts = read_selected_texts(data, rows)
    read_time = time.perf_counter() - start
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    eos = tok.eos_token_id
    ids, masks = ([], [])
    for row in rows:
        tokens = tok(texts[row['text_sha256']], add_special_tokens=False, truncation=True, max_length=256)['input_ids']
        if not tokens:
            tokens = [eos]
        ids.append(tokens + [eos] * (256 - len(tokens)))
        masks.append([1] * len(tokens) + [0] * (256 - len(tokens)))
    tokenization_time = time.perf_counter() - start - read_time
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}, out)
    result = {'status': 'PREFLIGHT_ONLY', 'documents': 2048, 'manifest_sha256': filehash(manifest), 'tensor_sha256': filehash(out), 'read_parquet_seconds': read_time, 'tokenization_seconds': tokenization_time, 'total_seconds': time.perf_counter() - start}
    out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
if __name__ == '__main__':
    main(*(Path(p) for p in sys.argv[1:5]))
