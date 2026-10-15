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
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/exp_s1'
BASE = ROOT / 'outputs'
WORK = ROOT / 'work'
DATA = Path(_p1_path('path_8c5dea5cdb13'))

def sha(b):
    return hashlib.sha256(b).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
opening = OUT / 'AUDIT_OPENING_RECORD.json'
if not opening.exists():
    raise RuntimeError('AUDIT_NOT_OPENED')
record = json.loads(opening.read_text())
assert record['status'] == 'AUDIT_OPENED_AFTER_FROZEN_PREREQUISITES'
assert record['analysis_code_sha256']['materialize_audit.py'] == file_sha(Path(__file__))
manifest = BASE / 'audit_bank_manifest.jsonl'
data = json.loads((BASE / 'DATA_MANIFEST_HASHES.json').read_text())
assert file_sha(manifest) == data['manifest_sha256']['audit']
rows = [json.loads(x) for x in manifest.open()]
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
    print('AUDIT_TEXT_SHARD', shard, flush=True)
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
bank = WORK / 's1_audit_bank.pt'
batch = {'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks)}
torch.save(batch, bank)
model = AutoModelForCausalLM.from_pretrained(str(WORK / 'model'), dtype=torch.float32, local_files_only=True).to('mps').eval()
mlp = model.gpt_neox.layers[3].mlp
collected = []
with torch.inference_mode():
    for start in range(0, 2048, 8):
        x = batch['input_ids'][start:start + 8].to('mps')
        mask = batch['attention_mask'][start:start + 8].to('mps')
        captured = []
        hook = mlp.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
        model.gpt_neox(input_ids=x, attention_mask=mask, use_cache=False)
        hook.remove()
        h = captured[0]
        last = mask.sum(dim=1).long() - 1
        collected.append(h[torch.arange(len(h), device='mps'), last].cpu())
rep = WORK / 's1_audit_representation.pt'
h = torch.cat(collected, dim=0).contiguous()
assert tuple(h.shape) == (2048, 512) and bool(torch.isfinite(h).all())
torch.save({'h': h}, rep)
result = {'status': 'AUDIT_REPRESENTATION_MATERIALIZED_AFTER_OPENING', 'audit_opening_record_sha256': file_sha(opening), 'bank_manifest_sha256': file_sha(manifest), 'pair_manifest_sha256': file_sha(BASE / 'audit_pair_manifest.jsonl'), 'tokenized_bank_tensor_sha256': file_sha(bank), 'representation_tensor_sha256': file_sha(rep), 'documents': 2048, 'representation': 'untouched base model layer-4 MLP input at final nonpadding token'}
(OUT / 'AUDIT_REPRESENTATION.json').write_text(json.dumps(result, indent=2) + '\n')
print('AUDIT_REPRESENTATION_READY', result['representation_tensor_sha256'], flush=True)
