# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
WORK = ROOT / 'work'
DATA = Path(_p1_path('path_8c5dea5cdb13'))

def sha(b):
    return hashlib.sha256(b).hexdigest()
rows = [json.loads(x) for x in (OUT / 'calibration_bank_manifest.jsonl').open()]
assert len(rows) == 2048
wanted = defaultdict(dict)
for i, row in enumerate(rows):
    wanted[row['shard']][row['row_ordinal']] = (i, row['text_sha256'])
texts = [None] * 2048
for shard, ordinals in sorted(wanted.items()):
    pf = pq.ParquetFile(next(DATA.glob(f'train-{shard:05d}-of-00048-*.parquet')))
    offset = 0
    for rg in range(pf.num_row_groups):
        size = pf.metadata.row_group(rg).num_rows
        local = [(o, i, h) for o, (i, h) in ordinals.items() if offset <= o < offset + size]
        if local:
            col = pf.read_row_group(rg, columns=['text']).column(0)
            for o, i, h in local:
                value = col[o - offset].as_py()
                assert sha(value.encode('utf-8')) == h
                texts[i] = value
        offset += size
    print('read shard', shard, flush=True)
assert all((x is not None for x in texts))
tok = AutoTokenizer.from_pretrained(str(WORK / 'model'), local_files_only=True)
eos = tok.eos_token_id
ids = []
masks = []
for row, value in zip(rows, texts):
    t = tok(value, add_special_tokens=False, truncation=True, max_length=256)['input_ids']
    if not t:
        t = [eos]
    a = t + [eos] * (256 - len(t))
    b = [1] * len(t) + [0] * (256 - len(t))
    assert sha(json.dumps({'input_ids': a, 'attention_mask': b}, sort_keys=True, separators=(',', ':')).encode()) == row['tokenized_sequence_sha256']
    ids.append(a)
    masks.append(b)
path = WORK / 's1_calibration_bank.pt'
torch.save({'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks)}, path)
print('CALIBRATION_MATERIALIZED', sha(path.read_bytes()), flush=True)
