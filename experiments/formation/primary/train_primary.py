# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""R3: the frozen 30 deterministic sign-paired full-F0 arms, fail-closed locality."""
import sys as _p1_sys
from pathlib import Path as _P1Path
_p1_root = next((p for p in _P1Path(__file__).resolve().parents if (p / 'project1_paths.py').is_file()))
_p1_sys.path.insert(0, str(_p1_root))
from project1_paths import resolve_path as _p1_path
import os, socket, time, math
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_TELEMETRY='1')

def deny(*a, **kw):
    raise RuntimeError('P2J1_NETWORK_PROHIBITED')
socket.create_connection = deny
socket.socket.connect = deny
socket.socket.connect_ex = deny
from common import *
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def norm(x):
    return float(torch.linalg.vector_norm(x.double()))

class PrivateMoE(nn.Module):

    def __init__(self, dense):
        super().__init__()
        self.dense = dense
        dense.requires_grad_(False)
        self.M = nn.Parameter(torch.zeros_like(dense.down_proj.weight))
        self.U = nn.Parameter(torch.zeros(576))
        self.register_buffer('c', torch.tensor([0.5, 0.5, -0.5, -0.5]))

    def forward(self, h):
        phi = F.silu(self.dense.gate_proj(h)) * self.dense.up_proj(h)
        y = self.dense.down_proj(phi)
        p = torch.softmax((h @ self.U).unsqueeze(-1) * self.c, dim=-1)
        return y + (p * self.c).sum(-1, keepdim=True) * F.linear(phi, self.M)

def context(model, h):
    pos = torch.arange(h.shape[1]).unsqueeze(0)
    mask = create_causal_mask(config=model.config, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos)
    return dict(attention_mask=mask, position_embeddings=model.model.rotary_emb(h, position_ids=pos), position_ids=pos, use_cache=False)

def suffix(model, h, ids):
    for layer in model.model.layers[20:]:
        h = layer(h, **context(model, h))
    logits = model.lm_head(model.model.norm(h)).float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))

def run():
    seal = verify_seal()
    audit = read(OUT / 'P2J_REUSE_AUDIT.json')
    binding = read(OUT / 'P2J_F0_AND_C_BINDING.json')
    assert not (OUT / 'P2J_TRAJECTORIES.parquet').exists()
    for p, h in audit['consumed_sha256'].items():
        assert sha(p) == h, p
    ids = torch.from_numpy(np.load(binding['F0_file'], allow_pickle=False))
    assert ids.shape == (32, 256)
    z = np.load(binding['cross_hessian_path'], allow_pickle=False)
    v = torch.from_numpy(z['v'][:, :, 0].copy())
    u = torch.from_numpy(z['u'][:, 0].copy())
    vd = v.double()
    ud = u.double()
    s = float(z['s'][0])
    model = AutoModelForCausalLM.from_pretrained(audit['model_dir'], dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert (model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers) == (576, 1536, 30)
    moe = PrivateMoE(model.model.layers[20].mlp)
    model.model.layers[20].mlp = moe
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ['model.layers.20.mlp.M', 'model.layers.20.mlp.U']
    preds = []
    for p in csvread(OUT / 'P2J_FROZEN_CELL_PREDICTIONS.csv'):
        preds.append({k: int(val) if k in ['cell', 'N_cell'] else val if k == 'primary_role' else float(val) for k, val in p.items()})
    hidden = []
    with torch.no_grad():
        for row in ids:
            h = model.model.embed_tokens(row[None])
            kw = context(model, h)
            for layer in model.model.layers[:20]:
                h = layer(h, **kw)
            hidden.append(h.detach())
    start = time.monotonic()
    started = utc()
    updates = 0
    grad_evals = 0
    nll_only = 0
    ckpts = []
    cellstatus = []
    writer = None
    rowcount = 0
    checkpointdir = OUT / 'checkpoints'
    checkpointdir.mkdir()

    def progress(cell, n, N, status):
        p = ROOT / 'work/p2j1_progress.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(status=status, cell=cell, update=n, horizon=N, completed_cells=len(cellstatus), primary_updates=updates, frozen_budget=912, elapsed_seconds=time.monotonic() - start, output_dir=str(OUT)), indent=2) + '\n')
        tmp.replace(p)

    def assign(state):
        with torch.no_grad():
            moe.M.copy_(state[0])
            moe.U.copy_(state[1])

    def eval_bank(state, p, need_gradient):
        nonlocal grad_evals, nll_only
        assign(state)
        losses = []
        if need_gradient:
            gm = torch.zeros_like(moe.M, dtype=torch.float64)
            gu = torch.zeros_like(moe.U, dtype=torch.float64)
            for row, h in zip(ids, hidden):
                loss = suffix(model, h, row[None])
                a, b = torch.autograd.grad(loss, (moe.M, moe.U))
                gm += a.detach().double()
                gu += b.detach().double()
                losses.append(float(loss.detach()))
            gm /= 32
            gu /= 32
            gm += p['mu'] * state[0].double()
            gu += p['lambda_'] * state[1].double()
            grad_evals += 1
            return (gm, gu, float(np.mean(losses)))
        with torch.no_grad():
            for row, h in zip(ids, hidden):
                losses.append(float(suffix(model, h, row[None])))
        nll_only += 1
        return (None, None, float(np.mean(losses)))

    def observe(p, sign, n, state, a_initial):
        M, U = state
        m = float((M.double() * vd).sum())
        nu = float((U.double() * ud).sum())
        mn = norm(M)
        un = norm(U)
        full = math.hypot(mn, un)
        A = math.hypot(m, nu)
        C = (p['e_M'] * m + p['e_U'] * nu) ** 2 / (A * A)
        return dict(cell=p['cell'], epsilon=p['epsilon'], r_eta=p['r_eta'], sign=sign, update=n, N_cell=p['N_cell'], primary_role=p['primary_role'], m1=m, u1=nu, A1=A, M_norm=mn, U_norm=un, A_full=full, F1=A * A / (full * full), C_plus=C, full_amplitude_ratio=full / a_initial, sign_pair_antisymmetry=0.0, F0_NLL=0.0, M_regularizer=0.5 * p['mu'] * mn * mn, U_regularizer=0.5 * p['lambda_'] * un * un, locality_violation=False, locality_reasons='')
    for p in preds:
        N = p['N_cell']
        a0 = p['a0']
        states = [(-a0 * p['e_M'] * v, -a0 * p['e_U'] * u), (a0 * p['e_M'] * v, a0 * p['e_U'] * u)]
        initial = [math.hypot(norm(M), norm(U)) for M, U in states]
        rows = []
        violation = None
        for n in range(N + 1):
            pair = [observe(p, sign, n, state, init) for sign, state, init in zip([-1, 1], states, initial)]
            xp = np.array([pair[1]['m1'], pair[1]['u1']])
            xm = np.array([pair[0]['m1'], pair[0]['u1']])
            E = float(np.linalg.norm(xp + xm) / (np.linalg.norm(xp) + np.linalg.norm(xm)))
            reasons = []
            for rec in pair:
                rec['sign_pair_antisymmetry'] = E
                armreasons = []
                if rec['F1'] < 0.95:
                    armreasons.append('LEADING_PAIR_CAPTURE')
                if rec['C_plus'] < 0.95:
                    armreasons.append('EIGENVECTOR_ALIGNMENT')
                if E > 0.05:
                    armreasons.append('SIGN_ANTISYMMETRY')
                if not 0.7 <= rec['full_amplitude_ratio'] <= 1.4:
                    armreasons.append('FULL_AMPLITUDE_ENVELOPE')
                if not all((math.isfinite(rec[k]) for k in ['F1', 'C_plus', 'A1', 'A_full'])):
                    armreasons.append('NONFINITE_OBSERVABLE')
                rec['locality_reasons'] = ';'.join(armreasons)
                reasons += armreasons
            violating = bool(reasons)
            grads = []
            for rec, state in zip(pair, states):
                gm, gu, loss = eval_bank(state, p, need_gradient=n < N and (not violating))
                rec['F0_NLL'] = loss
                rec['locality_violation'] = violating
                assert math.isfinite(loss)
                grads.append((gm, gu))
                if n % 4 == 0 or n == N or violating:
                    path = checkpointdir / f"cell{p['cell']:02d}_sign{rec['sign']:+d}_n{n:03d}.npz"
                    assert not path.exists()
                    np.savez(path, M=state[0].numpy(), U=state[1].numpy())
                    path.chmod(292)
                    ckpts.append(dict(cell=p['cell'], sign=rec['sign'], update=n, path=str(path), sha256=sha(path), epsilon=p['epsilon'], r_eta=p['r_eta']))
            rows += pair
            rowcount += 2
            if violating:
                violation = dict(update=n, reasons=sorted(set(reasons)))
                log = f"LOCALITY_VIOLATION cell={p['cell']} n={n} reasons={violation['reasons']}"
                print(log, flush=True)
                break
            if n < N:
                states = [((state[0].double() - p['eta_M'] * g[0]).float(), (state[1].double() - p['eta_U'] * g[1]).float()) for state, g in zip(states, grads)]
                updates += 2
            progress(p['cell'], n, N, 'TRAINING')
            if (n + 1) % 8 == 0:
                print(f"R3 cell={p['cell']:02d} eps={p['epsilon']:+g} r={p['r_eta']:g} pair_n={n + 1}/{N} total_updates={updates}/912 elapsed={time.monotonic() - start:.0f}s", flush=True)
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(OUT / 'P2J_TRAJECTORIES.parquet', table.schema)
        writer.write_table(table)
        cellstatus.append(dict(cell=p['cell'], epsilon=p['epsilon'], r_eta=p['r_eta'], role=p['primary_role'], frozen_horizon=N, completed_update=rows[-1]['update'], status='LOCALITY_VIOLATION' if violation else 'HORIZON_COMPLETE', violation=violation))
        print(f"R3 COMPLETE cell={p['cell']:02d} horizon={N} status={cellstatus[-1]['status']}; arms_completed={2 * len(cellstatus)}/30; total_updates={updates}/912 elapsed={time.monotonic() - start:.0f}s", flush=True)
    if writer:
        writer.close()
        (OUT / 'P2J_TRAJECTORIES.parquet').chmod(292)
    write('P2J_CHECKPOINT_INDEX.json', dict(status='COMPLETE', checkpoints=ckpts, checkpoint_rule='every 4 updates, plus initial/final/first locality violation', count=len(ckpts), preflight_seal_sha256=sha(OUT / 'P2J_PREFLIGHT_SEAL.json')))
    write('P2J_TRAINING_COMPLETION.json', dict(status='COMPLETE', primary_runs=30, scientific_cells=15, cell_status=cellstatus, primary_updates=updates, frozen_budget=912, full_F0_gradient_evaluations=grad_evals, final_or_violating_NLL_evaluations=nll_only, logged_rows=rowcount, input_tokens_per_gradient=8192, supervised_tokens_per_gradient=8160, primary_input_tokens=updates * 8192, elapsed_seconds=time.monotonic() - start, started_utc=started, finished_utc=utc(), preflight_seal_sha256=sha(OUT / 'P2J_PREFLIGHT_SEAL.json'), source_sha256={str(f): sha(f) for f in SOURCE.glob('*.py')}, runtime=dict(python=__import__('sys').executable, torch=torch.__version__, transformers=__import__('transformers').__version__, backend='CPU', threads=8, deterministic=True, parameter_dtype='float32', gradient_accumulation_dtype='float64', microbatch_sequences=1, immutable_sequence_order=list(range(32)), dropout=False, stochastic_routing=False, common_parameters_frozen=True, loss_normalization='sum of 32 per-sequence mean NLL gradients /32 once'), counters=COUNTERS))
    progress(14, 0, 0, 'TRAINING_COMPLETE')
    print('R3 COMPLETE: all 30 deterministic arms closed at frozen horizon or fail-closed locality stop', flush=True)
if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        write('P2J1_EXECUTION_STOP.json', dict(status='STOP', error_type=type(e).__name__, error=str(e), created_utc=utc(), counters=COUNTERS))
        raise
