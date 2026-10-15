# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""Materialize ONLY the permanently excluded J1 preflight response bank."""
from __future__ import annotations
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

def sha(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def materialize(manifest: Path, data: Path, model_dir: Path, out: Path):
    rows = [json.loads(line) for line in manifest.open()]
    if len(rows) != 2048 or not all((r['bank'] == 'preflight' for r in rows)):
        raise ValueError('only full J1 preflight ID bank is allowed')
    wanted = {}
    for r in rows:
        wanted.setdefault(r['shard'], {})[r['row_ordinal']] = r['text_sha256']
    texts = {}
    for shard, ordinals in sorted(wanted.items()):
        matches = list(data.glob(f'train-{shard:05d}-of-00048-*.parquet'))
        if len(matches) != 1:
            raise FileNotFoundError(f'pinned shard {shard}')
        pf = pq.ParquetFile(matches[0])
        offset = 0
        for rg in range(pf.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            local = [(ordinal, h) for ordinal, h in ordinals.items() if offset <= ordinal < offset + size]
            if local:
                col = pf.read_row_group(rg, columns=['text']).column(0)
                for ordinal, h in local:
                    value = col[ordinal - offset].as_py()
                    if hashlib.sha256(value.encode('utf-8')).hexdigest() != h:
                        raise ValueError('pinned-source text hash mismatch')
                    texts[h] = value
            offset += size
    if len(texts) != 2048:
        raise ValueError('missing selected texts')
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    eos = tok.eos_token_id
    ids, masks = ([], [])
    for row in rows:
        t = tok(texts[row['text_sha256']], add_special_tokens=False, truncation=True, max_length=256)['input_ids']
        if not t:
            t = [eos]
        ids.append(t + [eos] * (256 - len(t)))
        masks.append([1] * len(t) + [0] * (256 - len(t)))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'input_ids': torch.tensor(ids, dtype=torch.long), 'attention_mask': torch.tensor(masks, dtype=torch.long)}, out)
    record = {'status': 'J1_PREFLIGHT_ONLY', 'documents': 2048, 'manifest_sha256': sha(manifest), 'tensor_sha256': sha(out), 'future_calibration_evaluated': False, 'future_audit_evaluated': False}
    out.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
if __name__ == '__main__':
    import sys
    print(json.dumps(materialize(*(Path(x) for x in sys.argv[1:5]))))
