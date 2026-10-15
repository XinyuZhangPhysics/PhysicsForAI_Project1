# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
from common import *
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM

def build(f, m, device):
    if m == 'audit':
        verify_freeze()
        assert (ROOT / 'audit/AUDIT_OPENING_RECORD.json').exists()
    b = bind()
    item = b['fresh_banks'][f + '_' + m]
    assert sha(item['path']) == item['sha256']
    rows = [json.loads(l) for l in Path(item['path']).open()]
    texts = {}
    wanted = {}
    corpus = {}
    for r in rows:
        wanted.setdefault(r['shard'], {})[r['row_ordinal']] = r['text_sha256']
    for sh, ordinals in sorted(wanted.items()):
        p = next(DATA.glob(f'train-{sh:05d}-of-00048-*.parquet'))
        corpus[str(p)] = sha(p)
        pf = pq.ParquetFile(p)
        offset = 0
        for rg in range(pf.num_row_groups):
            size = pf.metadata.row_group(rg).num_rows
            local = [(o, h) for o, h in ordinals.items() if offset <= o < offset + size]
            if local:
                col = pf.read_row_group(rg, columns=['text']).column(0)
                for o, h in local:
                    text = col[o - offset].as_py()
                    assert digest(text.encode()) == h
                    texts[h] = text
            offset += size
    assert len(texts) == 2048
    tok = AutoTokenizer.from_pretrained(str(MODELS[f]), local_files_only=True)
    eos = tok.eos_token_id
    ids = []
    masks = []
    for r in rows:
        x = tok(texts[r['text_sha256']], add_special_tokens=False, truncation=True, max_length=256)['input_ids'] or [eos]
        ids.append(x + [eos] * (256 - len(x)))
        masks.append([1] * len(x) + [0] * (256 - len(x)))
    path = ROOT / m / f'{f}_tokens.pt'
    torch.save({'input_ids': torch.tensor(ids), 'attention_mask': torch.tensor(masks)}, path)
    del texts
    model = AutoModelForCausalLM.from_pretrained(str(MODELS[f]), torch_dtype=torch.float32, local_files_only=True).to(device).eval()
    dense = model.gpt_neox.layers[3].mlp if f == 'S1' else model.model.layers[19].mlp
    base = model.gpt_neox if f == 'S1' else model.model
    captured = []
    hook = dense.register_forward_pre_hook(lambda mod, args: captured.append(args[0].detach()))
    hs = []
    with torch.inference_mode():
        for i in range(0, 2048, 8):
            x = torch.tensor(ids[i:i + 8], device=device)
            mask = torch.tensor(masks[i:i + 8], device=device)
            captured.clear()
            base(input_ids=x, attention_mask=mask, use_cache=False)
            h = captured[0]
            last = mask.sum(1).long() - 1
            hs.append(h[torch.arange(len(h), device=device), last].cpu())
            if i % 512 == 0:
                print('REPRESENTATION', m, f, i, flush=True)
    hook.remove()
    h = torch.cat(hs).contiguous()
    assert h.shape == (2048, 512 if f == 'S1' else 576)
    torch.save({'h': h}, ROOT / m / f'{f}_representation.pt')
    with torch.inference_mode():
        hm = h.to(device)
        if f == 'S1':
            features = dense.act(dense.dense_h_to_4h(hm))
            outweight = dense.dense_4h_to_h.weight
            outbias = dense.dense_4h_to_h.bias
        else:
            features = dense.act_fn(dense.gate_proj(hm)) * dense.up_proj(hm)
            outweight = dense.down_proj.weight
            outbias = None
        feature = features.cpu().double().numpy()
        ow = outweight.cpu().double().numpy()
    rng = np.random.default_rng(np.random.SeedSequence([b['fresh_pair_seed'], 1 if f == 'S1' else 2, 1 if m == 'calibration' else 2]))
    src = []
    tgt = []
    groups = []
    pairs = []
    for a, source in enumerate(DOMAINS):
        for bb, target in enumerate(DOMAINS):
            if a == bb:
                continue
            group = len(pairs) // 1024
            for n, pos in enumerate(rng.choice(512 * 512, 1024, replace=False)):
                x = a * 512 + int(pos) // 512
                y = bb * 512 + int(pos) % 512
                src.append(x)
                tgt.append(y)
                groups.append(group)
                pairs.append({'source_train_row_id': rows[x]['train_row_id'], 'target_train_row_id': rows[y]['train_row_id'], 'source_domain': source, 'target_domain': target, 'pair_index': n})
    pp = ROOT / m / f'{f}_pairs.jsonl'
    pp.write_text(''.join((json.dumps(r, sort_keys=True) + '\n' for r in pairs)))
    src = np.array(src)
    tgt = np.array(tgt)
    groups = np.array(groups)
    meta = {}
    for g in [2, 4, 8]:
        width = feature.shape[1] // g
        child = np.empty((12288, g))
        out = np.empty((2048, g, h.shape[1]))
        for c in range(g):
            sl = slice(c * width, (c + 1) * width)
            for start in range(0, 12288, 1024):
                end = start + 1024
                child[start:end, c] = (feature[src[start:end], sl] * feature[tgt[start:end], sl]).mean(1)
            out[:, c] = feature[:, sl] @ ow[:, sl].T
            if outbias is not None:
                out[:, c] += outbias.detach().cpu().double().numpy() / g
        kc = np.repeat((child - child.mean(1, keepdims=True))[:, None, :], 4, axis=1).reshape(12288, 4 * g)
        assert np.max(abs(kc @ proj(g)['C'] - kc)) < 1e-12
        cp = ROOT / m / f'{f}_g{g}_geometry.npz'
        np.savez_compressed(cp, h=h.numpy(), src=src, tgt=tgt, group=groups, kappa_C=kc, child_output=out)
        meta[str(g)] = {'path': str(cp), 'sha256': sha(cp), 'K_C': float(np.mean((kc * kc).sum(1)))}
    write(ROOT / m / f'{f}_BANK_BUILD.json', {'timestamp': now(), 'family': f, 'membership': m, 'model_path': str(MODELS[f]), 'device': device, 'torch': torch.__version__, 'documents': 2048, 'pairs': 12288, 'token_sha256': sha(path), 'identity_sha256': item['sha256'], 'pair_sha256': sha(pp), 'corpus_shards': corpus, 'geometry': meta, 'representation': 'MLP input at final nonpadding token; layer index 3 / 19; 256-token truncation; base-model eval float32', 'geometry_convention': 'GELU or SwiGLU activation product mean over each equal contiguous child partition, repeated four parents then C-projected'})
    del model, dense, base, features
    if device == 'mps':
        torch.mps.empty_cache()
    print('BANK_READY', f, m, flush=True)
if __name__ == '__main__':
    import sys
    build(*sys.argv[1:])
