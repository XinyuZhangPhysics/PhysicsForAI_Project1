# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""H1: exact frozen 120 arms; paired streams; float32 deterministic plain SGD."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from pathlib import Path
import json, time, sys, hashlib
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
from preflight import PrivateMoE, ROOT, OUT, WORK, MODEL, sha, tsha, norm, write, log
CKPT = OUT / 'checkpoints'
CKPT.mkdir(exist_ok=True)

def context(model, h):
    pos = torch.arange(h.shape[1]).unsqueeze(0)
    mask = create_causal_mask(config=model.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos)
    rot = model.model.rotary_emb(h, position_ids=pos)
    return dict(attention_mask=mask, position_embeddings=rot, position_ids=pos, use_cache=False)

def prefix(model, ids):
    with torch.no_grad():
        h = model.model.embed_tokens(ids)
        kw = context(model, h)
        for layer in model.model.layers[:20]:
            h = layer(h, **kw)
    return h.detach()

def suffix(model, h, ids, return_logits=False):
    kw = context(model, h)
    for layer in model.model.layers[20:]:
        h = layer(h, **kw)
    h = model.model.norm(h)
    logits = model.lm_head(h).float()
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
    return (loss, logits) if return_logits else loss

def verify_seal():
    seal = json.loads((OUT / 'P2H_PREFLIGHT_SEAL.json').read_text())
    assert seal['status'] == 'PASS'
    for n, h in seal['artifact_sha256'].items():
        assert sha(OUT / n) == h, n
    for p, h in seal['source_sha256'].items():
        assert sha(p) == h, p
    assert sha(WORK / 'p2h_private_seeds.pt') == seal['seed_file_sha256']
    return seal

def benchmark():
    verify_seal()
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    moe = PrivateMoE(model.model.layers[20].mlp)
    model.model.layers[20].mlp = moe
    ids = torch.from_numpy(np.load(WORK / 'p2h_banks/F0.npy'))[:4]
    seed = torch.load(WORK / 'p2h_private_seeds.pt', weights_only=True)[0]
    with torch.no_grad():
        moe.M.copy_(seed)
        moe.U.copy_(torch.randn(576, generator=torch.Generator().manual_seed(444)) * 1e-05)
    t = time.monotonic()
    hf = prefix(model, ids)
    pl = time.monotonic() - t
    a = model(input_ids=ids, labels=ids, use_cache=False)
    ag = torch.autograd.grad(a.loss, (moe.M, moe.U))
    af = float(a.loss.detach())
    al = a.logits.detach()
    del a
    t = time.monotonic()
    b, bl = suffix(model, hf, ids, True)
    bg = torch.autograd.grad(b, (moe.M, moe.U))
    sl = time.monotonic() - t
    errs = [norm(x - y) / max(norm(x), 1e-30) for x, y in zip(ag, bg)]
    le = norm(al - bl.detach()) / norm(al)
    assert max(errs) <= 1e-05 and le <= 1e-06, (errs, le)
    rec = dict(status='PASS', prefix_seconds=pl, suffix_forward_backward_seconds=sl, full_loss=af, cached_prefix_loss=float(b.detach()), logits_relative_error=le, gradient_relative_errors=errs, training_steps=0, optimization='Compute frozen layers 0..19 once per seed minibatch; reuse exact float32 boundary hidden states across 15 arms. All target and downstream layers evaluate normally.')
    write('P2H_COMPUTE_VERIFICATION.json', rec)
    log(json.dumps(rec))
    return model

def run():
    verify_seal()
    comp = json.loads((OUT / 'P2H_COMPUTE_VERIFICATION.json').read_text())
    assert comp['status'] == 'PASS'
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    moe = PrivateMoE(model.model.layers[20].mlp)
    model.model.layers[20].mlp = moe
    spec = np.load(OUT / 'P2H_CROSS_HESSIAN_SPECTRUM.npz')
    v = torch.from_numpy(spec['v'])
    u = torch.from_numpy(spec['u'])
    cluster = np.where(spec['s'] / spec['s'][0] >= 0.95)[0]
    assert spec['s'][1] / spec['s'][0] < 0.95, 'Leading singular cluster requires additional H0 modes before science'
    preds = json.loads((OUT / 'P2H_PREDICTIONS.json').read_text())
    seeds = torch.load(WORK / 'p2h_private_seeds.pt', weights_only=True)
    bankseal = json.loads((OUT / 'P2H_DATA_BANK_SEAL.json').read_text())
    paths = []
    checkpoints = []
    writer = None
    start = time.monotonic()

    def observe(j, k, n, M, U, loss=None):
        m = float((M.double() * v[:, :, 0].double()).sum())
        ur = float((U.double() * u[:, 0].double()).sum())
        mn = norm(M)
        un = norm(U)
        a2 = m * m + ur * ur
        e = preds['predictions'][k]['e_plus']
        z = e[0] * m + e[1] * ur
        row = dict(seed=j, arm=k, epsilon=preds['predictions'][k]['epsilon'], r_eta=preds['predictions'][k]['r_eta'], update=n, M_norm=mn, U_norm=un, m1=m, u1=ur, z_plus=z, A1=a2 ** 0.5, F_lead=a2 / (mn * mn + un * un), C_plus=z * z / a2, NLL_before_update=loss)
        if n in preds['checkpoints']:
            path = CKPT / f'seed{j}_arm{k:02d}_n{n:02d}.npz'
            assert not path.exists(), path
            np.savez(path, M=M.numpy(), U=U.numpy())
            path.chmod(292)
            checkpoints.append(dict(seed=j, arm=k, epsilon=row['epsilon'], r_eta=row['r_eta'], update=n, path=str(path), sha256=sha(path), M_value_sha256=tsha(M), U_value_sha256=tsha(U)))
        return row
    trajectory = OUT / 'P2H_TRAJECTORIES.parquet'
    assert not trajectory.exists()
    for j in range(8):
        stream = torch.from_numpy(np.load(WORK / f'p2h_banks/train_{j}.npy'))
        assert tsha(stream) == bankseal['banks'][f'train_{j}']['token_array_sha256']
        states = [(seeds[j].clone(), torch.zeros(576)) for _ in range(15)]
        rows = [observe(j, k, 0, *states[k]) for k in range(15)]
        for n in range(1, 65):
            ids = stream[4 * (n - 1):4 * n]
            hidden = prefix(model, ids)
            for k, pred in enumerate(preds['predictions']):
                with torch.no_grad():
                    moe.M.copy_(states[k][0])
                    moe.U.copy_(states[k][1])
                optimizer = torch.optim.SGD([dict(params=[moe.M], lr=pred['eta_M']), dict(params=[moe.U], lr=pred['eta_U'])], momentum=0, weight_decay=0)
                optimizer.zero_grad(set_to_none=True)
                nll = suffix(model, hidden, ids)
                loss = nll + 0.5 * pred['mu'] * moe.M.square().sum() + 0.5 * pred['lambda_'] * moe.U.square().sum()
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    states[k][0].copy_(moe.M)
                    states[k][1].copy_(moe.U)
                rows.append(observe(j, k, n, *states[k], float(nll.detach())))
            if n % 8 == 0:
                log(f'H1 seed {j}: {n}/64 updates across all 15 arms; elapsed {time.monotonic() - start:.0f}s')
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(trajectory, table.schema)
        writer.write_table(table)
        write('P2H_CHECKPOINT_INDEX.json', dict(status='IN_PROGRESS' if j < 7 else 'COMPLETE', checkpoints=checkpoints, seed_completed=j, checkpoint_updates=preds['checkpoints'], total_required=1080))
    if writer:
        writer.close()
    write('P2H_TRAINING_COMPLETION.json', dict(status='COMPLETE', primary_runs=120, updates_per_run=64, training_tokens=7864320, logged_rows=7800, checkpoints=len(checkpoints), elapsed_seconds=time.monotonic() - start, plain_SGD=True, autocast=False, gradient_clipping=False, source_sha256=sha(__file__), preflight_seal_sha256=sha(OUT / 'P2H_PREFLIGHT_SEAL.json')))
    log('H1 COMPLETE: 120 arms, 7680 updates, 7864320 training tokens')
if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        run()
