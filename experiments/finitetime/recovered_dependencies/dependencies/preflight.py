# Sanitized from archived historical source; only paths/import bootstrap/comments were changed.
"""P2H H0 only. No multi-step optimizer or science outcomes in this module."""
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
import json, hashlib, time, sys, math, platform
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from audit_assets import ROOT, OUT, MODEL, DATA, PKG, sha
WORK = ROOT / 'work'
BANK = WORK / 'p2h_banks'
BANK.mkdir(exist_ok=True)
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def norm(t):
    return float(torch.linalg.vector_norm(t.double()))

def log(s):
    print(s, flush=True)

class PrivateMoE(nn.Module):

    def __init__(self, dense):
        super().__init__()
        self.dense = dense
        self.dense.requires_grad_(False)
        self.M = nn.Parameter(torch.zeros_like(dense.down_proj.weight))
        self.U = nn.Parameter(torch.zeros(576))
        self.register_buffer('c', torch.tensor([0.5, 0.5, -0.5, -0.5]))

    def forward(self, h):
        phi = F.silu(self.dense.gate_proj(h)) * self.dense.up_proj(h)
        y = self.dense.down_proj(phi)
        p = torch.softmax((h @ self.U).unsqueeze(-1) * self.c, dim=-1)
        return y + (p * self.c).sum(-1, keepdim=True) * F.linear(phi, self.M)

def banks():
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    shard = next(DATA.glob('data/train-00000-*.parquet'))
    selected = []
    seen = set()
    ordinal = 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=128, columns=['text']):
        texts = batch.column(0).to_pylist()
        ids = tok(texts, add_special_tokens=False, truncation=True, max_length=256)['input_ids']
        for text, row in zip(texts, ids):
            if len(row) == 256:
                t = torch.tensor(row, dtype=torch.long)
                h = tsha(t)
                if h not in seen:
                    seen.add(h)
                    selected.append((t, dict(shard=str(shard), row_ordinal=ordinal, text_sha256=hashlib.sha256(text.encode()).hexdigest(), token_array_sha256=h)))
            ordinal += 1
            if len(selected) == 2112:
                break
        if len(selected) == 2112:
            break
    assert len(selected) == 2112, 'PINNED_DATA_NOT_FOUND'
    groups = {'F0': selected[:32], 'FE': selected[-32:]}
    for j in range(8):
        chunk = selected[32 + 256 * j:32 + 256 * (j + 1)]
        perm = np.random.default_rng(j).permutation(256)
        groups[f'train_{j}'] = [chunk[k] for k in perm]
    index = {}
    allhash = []
    for name, group in groups.items():
        ids = torch.stack([x[0] for x in group])
        path = BANK / f'{name}.npy'
        np.save(path, ids.numpy())
        rows = [x[1] for x in group]
        allhash.extend((x['token_array_sha256'] for x in rows))
        index[name] = dict(path=str(path), file_sha256=sha(path), token_array_sha256=tsha(ids), shape=list(ids.shape), tokens=ids.numel(), sequences=rows)
    assert len(allhash) == len(set(allhash)), 'BANK_OVERLAP_OR_PROVENANCE_FAIL'
    write('P2H_DATA_BANK_SEAL.json', dict(status='SEALED', model_commit=MODEL.name, dataset_revision=DATA.name, selection='first 2112 unique eligible documents in pinned train shard 0; first 256 tokenizer tokens; F0 first 32; eight disjoint 256-sequence training pools ordered by numpy default_rng(j); FE last 32', rows_scanned=ordinal, all_sequence_token_hashes_unique=True, banks=index))
    log('DATA BANKS SEALED: 2112 unique sequences; F0/FE 8192 tokens each; 8 streams of 65536 tokens')
    return {name: torch.from_numpy(np.load(rec['path'])) for name, rec in index.items()}

def capture(model, dense, ids):
    hs = []
    ps = []
    gs = []
    for n, row in enumerate(ids):
        cache = {}

        def hook(mod, args, y):
            h = args[0].detach()
            cache['h'] = h
            cache['phi'] = (F.silu(mod.gate_proj(h)) * mod.up_proj(h)).detach()
            cache['y'] = y.detach().requires_grad_(True)
            return cache['y']
        handle = dense.register_forward_hook(hook)
        try:
            loss = model(input_ids=row[None], labels=row[None], use_cache=False).loss
            g = torch.autograd.grad(loss, cache['y'])[0] / len(ids)
        finally:
            handle.remove()
        hs.append(cache['h'].reshape(-1, 576))
        ps.append(cache['phi'].reshape(-1, 1536))
        gs.append(g.detach().reshape(-1, 576))
        if (n + 1) % 8 == 0:
            log(f'F0 adjoint capture {n + 1}/{len(ids)}')
    return (torch.cat(hs), torch.cat(ps), torch.cat(gs))

class CrossOperator:

    def __init__(self, h, p, g, coef=0.25):
        self.h = h
        self.p = p
        self.g = g
        self.coef = coef
        self.calls = 0

    def ct(self, u):
        self.calls += 1
        return -self.coef * (self.g.T @ ((self.h @ u).unsqueeze(-1) * self.p))

    def c(self, m):
        self.calls += 1
        return -self.coef * (self.h.T @ (self.g * F.linear(self.p, m)).sum(-1))

    def gram(self, u):
        return self.c(self.ct(u))

def spectrum(op):
    q = torch.randn(576, 4, generator=torch.Generator().manual_seed(2026100803))
    q = torch.linalg.qr(q).Q
    for it in range(1, 513):
        y = torch.stack([op.gram(q[:, k]) for k in range(4)], dim=1)
        q = torch.linalg.qr(y).Q
        if it < 24:
            continue
        if it % 8:
            continue
        vraw = torch.stack([op.ct(q[:, k]) for k in range(4)], dim=-1)
        small = vraw.flatten(0, 1).T.double() @ vraw.flatten(0, 1).double()
        vals, rot = torch.linalg.eigh(small)
        order = torch.arange(3, -1, -1)
        vals = vals[order]
        rot = rot[:, order].float()
        u = q @ rot
        s = vals.clamp_min(0).sqrt()
        v = torch.einsum('abk,kl->abl', vraw, rot) / s.float()
        residual = []
        for k in range(4):
            residual.append(max(norm(op.c(v[:, :, k]) - s[k].float() * u[:, k]), norm(op.ct(u[:, k]) - s[k].float() * v[:, :, k])) / float(s[k]))
        log(f'Cross-Hessian iteration {it}: s1={s[0]:.8g}; leading residual={residual[0]:.3g}')
        if residual[0] <= 0.0001:
            return (s, u, v, residual, it)
    raise RuntimeError('CROSS_HESSIAN_NUMERICS_FAIL')

def validate_hvp(model, moe, h, p, g, ids):
    original = moe.c.clone()
    gen = torch.Generator().manual_seed(2026100804)
    d = torch.randn(576, 1536, generator=gen)
    d = d / norm(d)
    contrasts = [torch.tensor([1.0, -1.0, 0.0, 0.0]) / math.sqrt(2), torch.tensor([1.0, 1.0, -2.0, 0.0]) / math.sqrt(6), torch.tensor([1.0, 1.0, 1.0, -3.0]) / math.sqrt(12)]
    evidence = []
    for c in [original] + contrasts:
        moe.c.copy_(c)
        moe.M.data.zero_()
        moe.U.data.zero_()
        loss = model(input_ids=ids[:1], labels=ids[:1], use_cache=False).loss
        gm = torch.autograd.grad(loss, moe.M, create_graph=True)[0]
        actual = -torch.autograd.grad((gm * d).sum(), moe.U)[0].detach()
        coef = float((c.square().sum() / 4).item())
        sub = CrossOperator(h[:256], p[:256], g[:256] * 32, coef)
        expected = sub.c(d)
        error = norm(actual - expected) / max(norm(expected), 1e-30)
        evidence.append(dict(contrast=c.tolist(), coefficient=coef, hvp_relative_error=error))
        assert error <= 0.0001, 'CROSS_HESSIAN_NUMERICS_FAIL'
    moe.c.copy_(original)
    return evidence

def gradients(model, moe, ids, mu, lam):
    gm = torch.zeros_like(moe.M)
    gu = torch.zeros_like(moe.U)
    losses = []
    for row in ids:
        loss = model(input_ids=row[None], labels=row[None], use_cache=False).loss
        grads = torch.autograd.grad(loss, (moe.M, moe.U))
        gm += grads[0].detach() / len(ids)
        gu += grads[1].detach() / len(ids)
        losses.append(float(loss.detach()))
    return (gm + mu * moe.M.detach(), gu + lam * moe.U.detach(), float(np.mean(losses)))

def run():
    assert (OUT / 'P2H_ASSET_REUSE_AUDIT.json').exists()
    data = banks()
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), dtype=torch.float32, local_files_only=True, attn_implementation='eager').eval()
    model.requires_grad_(False)
    assert (model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers) == (576, 1536, 30)
    dense = model.model.layers[20].mlp
    h, p, g = capture(model, dense, data['F0'])
    torch.save(dict(h=h, phi=p, g=g), WORK / 'p2h_f0_adjoint.pt')
    op = CrossOperator(h, p, g)
    s, u, v, residual, it = spectrum(op)
    moe = PrivateMoE(dense)
    model.model.layers[20].mlp = moe
    assert [n for n, x in model.named_parameters() if x.requires_grad] == ['model.layers.20.mlp.M', 'model.layers.20.mlp.U']
    hvp = validate_hvp(model, moe, h, p, g, data['F0'])
    symmetry = []
    for rec in hvp[1:]:
        sop = CrossOperator(h, p, g, rec['coefficient'])
        uk = u[:, 0].clone()
        for _ in range(24):
            uk = sop.gram(uk)
            uk = uk / norm(uk)
        symmetry.append(norm(sop.ct(uk)))
    spread = (max(symmetry) - min(symmetry)) / np.mean(symmetry)
    assert spread <= 1e-05, 'EXPERT_LABEL_SYMMETRY_AUDIT_FAIL'
    mu = float(s[0])
    amplitude = 0.0001 * norm(dense.down_proj.weight)
    seeds = []
    forward = []
    with torch.no_grad():
        yd = dense.down_proj(p)
        for j in range(8):
            r = torch.randn(576, 1536, generator=torch.Generator().manual_seed(j))
            r -= float((r.double() * v[:, :, 0].double()).sum()) * v[:, :, 0]
            r /= norm(r)
            sj = 0.2 * v[:, :, 0] + math.sqrt(0.96) * r
            sj /= norm(sj)
            m = amplitude * sj
            seeds.append(m)
            moe.M.copy_(m)
            moe.U.zero_()
            ym = moe(h)
            err = norm(ym - yd) / norm(yd)
            forward.append(err)
            assert err <= 1e-06, 'FUNCTION_PRESERVATION_FAIL'
        moe.M.zero_()
        moe.U.zero_()
    np.savez(OUT / 'P2H_CROSS_HESSIAN_SPECTRUM.npz', s=s.numpy(), u=u.numpy(), v=v.numpy(), relative_residuals=residual, iterations=it, helmert_s1=symmetry)
    torch.save(seeds, WORK / 'p2h_private_seeds.pt')
    report = dict(status='H0_LINEARIZATION_PASSED', s=s.tolist(), relative_residuals=residual, iterations=it, label_symmetry_s1=symmetry, label_symmetry_relative_spread=float(spread), full_model_mixed_hvp_checks=hvp, function_preservation_relative_errors=forward, mu=mu, lambda_c_hess=mu, private_seed_norm=amplitude, backend='CPU', float32_derivatives=True, float64_scalar_reductions=True, python=sys.executable, torch=torch.__version__, transformers=__import__('transformers').__version__, numpy=np.__version__, model_state_parameters_frozen_except_M_U=True)
    write('P2H_LINEARIZATION_DIAGNOSTICS.json', report)
    (OUT / 'P2H_LINEARIZATION_REPORT.md').write_text('# P2H H0 linearization\n\n' + json.dumps(report, indent=2) + '\n\nThe cross-Hessian is evaluated through the exact symmetric-point mixed response: C M = -(1/4) sum_t h_t [g_t^T M phi_t], where g is the full-model NLL adjoint of the target MLP output, normalized over F0. CT is its matrix-free VJP. Independent full-model autograd HVP checks cover the declared contrast and three Helmert contrasts. Target: zero-based layer 20.\n')
    eta0 = min(0.01, 0.01 / mu)
    gate = []
    passed = None
    lam = 0.88 * mu
    etaU = 4 * eta0
    for factor in [1.0, 0.5, 0.25]:
        with torch.no_grad():
            moe.M.copy_(seeds[0])
            moe.U.zero_()
        gm, gu, loss = gradients(model, moe, data['F0'], mu, lam)
        eta = eta0 * factor
        dm = -eta * float((gm.double() * v[:, :, 0].double()).sum())
        du = -4 * eta * float((gu.double() * u[:, 0].double()).sum())
        m0 = float((seeds[0].double() * v[:, :, 0].double()).sum())
        linear = np.array([-eta * mu * m0, 4 * eta * mu * m0])
        actual = np.array([dm, du])
        error = float(np.linalg.norm(actual - linear) / max(np.linalg.norm(linear), 1e-30))
        rgm, rgu, _ = gradients(model, moe, data['F0'], mu, lam)
        replay = dict(M_gradient_exact=torch.equal(gm, rgm), U_gradient_exact=torch.equal(gu, rgu), M_gradient_sha256=tsha(gm), U_gradient_sha256=tsha(gu))
        gate.append(dict(factor=factor, eta_M=eta, actual_top_mode_displacement=actual.tolist(), linear_top_mode_displacement=linear.tolist(), relative_error=error, replay=replay, bank='F0', seed=0, epsilon=0.12, r_eta=4.0))
        log(f"Small-step gate factor={factor}: error={error:.6g}, replay={replay['M_gradient_exact'] and replay['U_gradient_exact']}")
        if error <= 0.05:
            passed = factor
            break
    write('P2H_SMALL_STEP_GATE.json', dict(attempts=gate, chosen_factor=passed, gate_threshold=0.05))
    if passed is None:
        raise RuntimeError('SMALL_STEP_GATE_FAIL')
    eta = eta0 * passed
    pred = []
    for epsilon in [-0.2, -0.08, 0.04, 0.12, 0.3]:
        for ratio in [0.25, 1.0, 4.0]:
            lam = (1 - epsilon) * mu
            A = np.array([[-eta * mu, eta * mu], [ratio * eta * mu, -ratio * eta * lam]], dtype=np.float64)
            sigma = (-(eta * mu + ratio * eta * lam) + math.sqrt((eta * mu - ratio * eta * lam) ** 2 + 4 * eta * ratio * eta * mu ** 2)) / 2
            ev = np.array([1.0, (sigma + eta * mu) / (eta * mu)], dtype=np.float64)
            ev /= np.linalg.norm(ev)
            pred.append(dict(epsilon=epsilon, r_eta=ratio, mu=mu, lambda_=lam, eta_M=eta, eta_U=ratio * eta, sigma_plus=sigma, e_plus=ev.tolist(), A=A.tolist()))
    write('P2H_PREDICTIONS.json', dict(predictions=pred, seeds=list(range(8)), updates=64, tokens_per_update=1024, primary_runs=120, checkpoints=[0, 1, 2, 4, 8, 16, 32, 48, 64], chosen_lr_factor=passed))
    files = ['P2H_ASSET_REUSE_AUDIT.json', 'P2H_DATA_BANK_SEAL.json', 'P2H_CROSS_HESSIAN_SPECTRUM.npz', 'P2H_LINEARIZATION_REPORT.md', 'P2H_LINEARIZATION_DIAGNOSTICS.json', 'P2H_SMALL_STEP_GATE.json', 'P2H_PREDICTIONS.json']
    write('P2H_PREFLIGHT_SEAL.json', dict(status='PASS', sealed_before_multistep_training=True, artifact_sha256={n: sha(OUT / n) for n in files}, source_sha256={str(f): sha(f) for f in Path(__file__).parent.glob('*.py')}, seed_file_sha256=sha(WORK / 'p2h_private_seeds.pt'), adjoint_file_sha256=sha(WORK / 'p2h_f0_adjoint.pt'), deterministic_replay=gate[-1]['replay'], training_updates_before_seal=0))
    log('P2H_PREFLIGHT_SEAL PASS')
if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        write('P2H_H0_STOP.json', dict(status='STOP_BEFORE_SCIENCE', error_type=type(e).__name__, error=str(e), preflight_seal_exists=(OUT / 'P2H_PREFLIGHT_SEAL.json').exists(), primary_training_updates=0))
        sys.exit(1)
