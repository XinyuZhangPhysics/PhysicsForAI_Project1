#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_dump_new(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + "\n")


def csv_dump_new(path, rows):
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def vector_norm(x):
    return float(torch.linalg.vector_norm(x.double()))


def deny_network(*args, **kwargs):
    raise RuntimeError("NETWORK_ACCESS_PROHIBITED_BY_FROZEN_PROTOCOL")


def context(model, h):
    pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=h,
        attention_mask=None,
        past_key_values=None,
        position_ids=pos,
    )
    return dict(
        attention_mask=mask,
        position_embeddings=model.model.rotary_emb(h, position_ids=pos),
        position_ids=pos,
        use_cache=False,
    )


def suffix_loss(model, h, ids, start_layer):
    for layer in model.model.layers[start_layer:]:
        h = layer(h, **context(model, h))
    logits = model.lm_head(model.model.norm(h)).float()
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
    )


class SparseTop2PrivateMoE(nn.Module):
    def __init__(self, dense, hidden_size, delta, contrast):
        super().__init__()
        self.dense = dense
        self.dense.requires_grad_(False)
        self.M = nn.Parameter(torch.zeros_like(dense.down_proj.weight))
        self.U = nn.Parameter(torch.zeros(hidden_size, dtype=dense.down_proj.weight.dtype))
        self.delta = float(delta)
        self.register_buffer("contrast", torch.tensor(contrast, dtype=dense.down_proj.weight.dtype))
        self.register_buffer(
            "baseline_logits",
            torch.tensor([0.0, 0.0, -self.delta, -self.delta], dtype=dense.down_proj.weight.dtype),
        )

    def router_logits(self, h):
        t = h @ self.U
        return self.baseline_logits + t.unsqueeze(-1) * self.contrast

    def routing(self, h):
        logits = self.router_logits(h)
        values, indices = torch.topk(logits, k=2, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(values, dim=-1)
        selected_c = self.contrast[indices]
        beta = (weights * selected_c).sum(dim=-1)
        return beta, logits, indices, values

    def forward(self, h):
        phi = F.silu(self.dense.gate_proj(h)) * self.dense.up_proj(h)
        y = self.dense.down_proj(phi)
        beta, _, _, _ = self.routing(h)
        return y + beta.unsqueeze(-1) * F.linear(phi, self.M)

    @torch.no_grad()
    def support_stats(self, h):
        _, logits, indices, values = self.routing(h)
        expected = torch.tensor([0, 1], device=indices.device)
        selected_sorted = torch.sort(indices, dim=-1).values
        turnover = int((selected_sorted != expected).any(dim=-1).sum().item())
        sorted_logits = torch.sort(logits, dim=-1, descending=True).values
        margin = sorted_logits[..., 1] - sorted_logits[..., 2]
        return dict(
            turnover_tokens=turnover,
            total_tokens=int(margin.numel()),
            minimum_margin=float(margin.min().double()),
            median_margin=float(margin.median().double()),
            maximum_active_logit_magnitude=float(logits[..., :2].abs().max().double()),
            minimum_selected_logit=float(values.min().double()),
        )


class CrossOperator:
    def __init__(self, h, phi, g, coefficient):
        self.h = h
        self.phi = phi
        self.g = g
        self.coefficient = float(coefficient)
        self.calls = 0

    def ct(self, u):
        self.calls += 1
        return -self.coefficient * (self.g.T @ ((self.h @ u).unsqueeze(-1) * self.phi))

    def c(self, m):
        self.calls += 1
        return -self.coefficient * (self.h.T @ (self.g * F.linear(self.phi, m)).sum(-1))

    def gram(self, u):
        return self.c(self.ct(u))


def leading_spectrum(op, cfg):
    block = int(cfg["block_size"])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["gaussian_seed"]))
    q = torch.randn(op.h.shape[1], block, generator=gen, dtype=op.h.dtype)
    q = torch.linalg.qr(q).Q
    min_it = int(cfg["minimum_iterations"])
    every = int(cfg["check_every"])
    max_it = int(cfg["max_iterations"])
    tol = float(cfg["leading_relative_residual_max"])
    for it in range(1, max_it + 1):
        y = torch.stack([op.gram(q[:, k]) for k in range(block)], dim=1)
        q = torch.linalg.qr(y).Q
        if it < min_it or it % every:
            continue
        vraw = torch.stack([op.ct(q[:, k]) for k in range(block)], dim=-1)
        flat = vraw.flatten(0, 1)
        small = flat.T.double() @ flat.double()
        vals, rot = torch.linalg.eigh(small)
        order = torch.arange(block - 1, -1, -1)
        vals = vals[order]
        rot = rot[:, order].to(vraw.dtype)
        u = q @ rot
        s = vals.clamp_min(0).sqrt()
        v = torch.einsum("abk,kl->abl", vraw, rot) / s.to(vraw.dtype)
        residuals = []
        for k in range(block):
            sk = float(s[k])
            r1 = vector_norm(op.c(v[:, :, k]) - s[k].to(v.dtype) * u[:, k])
            r2 = vector_norm(op.ct(u[:, k]) - s[k].to(v.dtype) * v[:, :, k])
            residuals.append(max(r1, r2) / max(sk, 1e-30))
        print(f"spectrum it={it}: s1={float(s[0]):.9g}, residual={residuals[0]:.3g}", flush=True)
        if residuals[0] <= tol:
            return s.cpu(), u.cpu(), v.cpu(), residuals, it
    raise RuntimeError("SPARSE_TOPK_CROSS_HESSIAN_NUMERICS_FAIL")


def cache_prefix(model, ids, start_layer):
    hidden = []
    with torch.no_grad():
        for j, row in enumerate(ids):
            h = model.model.embed_tokens(row[None])
            kw = context(model, h)
            for layer in model.model.layers[:start_layer]:
                h = layer(h, **kw)
            hidden.append(h.detach().cpu())
            if (j + 1) % 8 == 0:
                print(f"prefix cache {j + 1}/{len(ids)}", flush=True)
    return hidden


def capture_tangent_data(model, dense, ids, prefix_hidden, start_layer):
    hs, phis, gs = [], [], []
    for j, (row, h0) in enumerate(zip(ids, prefix_hidden)):
        cache = {}

        def hook(mod, args, output):
            h = args[0].detach()
            cache["h"] = h
            cache["phi"] = (F.silu(mod.gate_proj(h)) * mod.up_proj(h)).detach()
            cache["y"] = output.detach().requires_grad_(True)
            return cache["y"]

        handle = dense.register_forward_hook(hook)
        try:
            loss = suffix_loss(model, h0.clone(), row[None], start_layer)
            g = torch.autograd.grad(loss, cache["y"])[0] / len(ids)
        finally:
            handle.remove()
        hs.append(cache["h"].reshape(-1, cache["h"].shape[-1]).cpu())
        phis.append(cache["phi"].reshape(-1, cache["phi"].shape[-1]).cpu())
        gs.append(g.detach().reshape(-1, g.shape[-1]).cpu())
        if (j + 1) % 8 == 0:
            print(f"tangent capture {j + 1}/{len(ids)}", flush=True)
    return torch.cat(hs), torch.cat(phis), torch.cat(gs)


def origin_equivalence_and_nulls(dense, moe, h, tol):
    gen = torch.Generator().manual_seed(2026111202)
    with torch.no_grad():
        base = dense(h)
        moe.M.zero_()
        moe.U.zero_()
        origin = moe(h)
        origin_err = float((origin - base).abs().max().double())

        dm = torch.randn(moe.M.shape, generator=gen, dtype=moe.M.dtype)
        dm /= torch.linalg.vector_norm(dm)
        moe.M.copy_(1e-3 * dm)
        moe.U.zero_()
        expert_only = moe(h)
        expert_err = float((expert_only - base).abs().max().double())

        du = torch.randn(moe.U.shape, generator=gen, dtype=moe.U.dtype)
        du /= torch.linalg.vector_norm(du)
        moe.M.zero_()
        moe.U.copy_(1e-3 * du)
        router_only = moe(h)
        router_err = float((router_only - base).abs().max().double())

        moe.M.zero_()
        moe.U.zero_()

    if origin_err > tol["origin_output_absolute_max"]:
        raise RuntimeError(f"ORIGIN_OUTPUT_EQUIVALENCE_FAIL: {origin_err}")
    if expert_err > tol["separate_tangent_output_absolute_max"]:
        raise RuntimeError(f"EXPERT_TANGENT_NULL_FAIL: {expert_err}")
    if router_err > tol["separate_tangent_output_absolute_max"]:
        raise RuntimeError(f"ROUTER_TANGENT_NULL_FAIL: {router_err}")
    return dict(
        origin_output_max_abs=origin_err,
        pure_expert_output_max_abs=expert_err,
        pure_router_output_max_abs=router_err,
    )


def hvp_validation(model, moe, h, phi, g, ids, prefix_hidden, start_layer, coefficient, max_rel):
    gen = torch.Generator().manual_seed(2026111203)
    d = torch.randn(moe.M.shape, generator=gen, dtype=moe.M.dtype)
    d /= torch.linalg.vector_norm(d)
    with torch.no_grad():
        moe.M.zero_()
        moe.U.zero_()
    loss = suffix_loss(model, prefix_hidden[0].clone(), ids[:1], start_layer)
    gm = torch.autograd.grad(loss, moe.M, create_graph=True)[0]
    actual = -torch.autograd.grad((gm * d).sum(), moe.U)[0].detach().cpu()
    sub = CrossOperator(h[: ids.shape[1]], phi[: ids.shape[1]], g[: ids.shape[1]] * len(ids), coefficient)
    expected = sub.c(d.cpu())
    rel = vector_norm(actual - expected) / max(vector_norm(expected), 1e-30)
    if rel > max_rel:
        raise RuntimeError(f"SPARSE_TOPK_HVP_FAIL: {rel}")
    return dict(relative_error=rel, actual_norm=vector_norm(actual), expected_norm=vector_norm(expected))


def full_bank_gradient(model, moe, ids, prefix_hidden, start_layer, mu, lam):
    gm = torch.zeros_like(moe.M, dtype=torch.float64)
    gu = torch.zeros_like(moe.U, dtype=torch.float64)
    losses = []
    for row, h0 in zip(ids, prefix_hidden):
        loss = suffix_loss(model, h0.clone(), row[None], start_layer)
        a, b = torch.autograd.grad(loss, (moe.M, moe.U))
        gm += a.detach().double()
        gu += b.detach().double()
        losses.append(float(loss.detach()))
    gm /= len(ids)
    gu /= len(ids)
    gm += mu * moe.M.detach().double()
    gu += lam * moe.U.detach().double()
    return gm, gu, float(np.mean(losses))


def predicted_cell(s, eps, alpha, r_eta):
    mu = s
    lam = s * (1.0 - eps)
    eta_x = alpha / mu
    eta_u = r_eta * eta_x
    T = np.array(
        [
            [1.0 - eta_x * mu, eta_x * s],
            [eta_u * s, 1.0 - eta_u * lam],
        ],
        dtype=float,
    )
    vals, vecs = np.linalg.eig(T)
    order = np.argsort(vals.real)
    k = int(order[-1])
    sigma = float(vals[k].real)
    e = vecs[:, k].real
    e /= np.linalg.norm(e)
    if e[0] < 0:
        e = -e
    other = float(vals[int(order[-2])].real)
    return dict(
        epsilon=float(eps),
        mu=float(mu),
        lambda_=float(lam),
        eta_X=float(eta_x),
        eta_U=float(eta_u),
        sigma_plus=sigma,
        sigma_minus=other,
        gamma_pred=float(math.log(abs(sigma))),
        e_M=float(e[0]),
        e_U=float(e[1]),
        T=T.tolist(),
    )


def fit_rate(rows):
    n = np.array([r["update"] for r in rows], dtype=float)
    y = np.log(np.array([r["A_lead"] for r in rows], dtype=float))
    p = np.polyfit(n, y, 1)
    pred = np.polyval(p, n)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(p[0]), r2


def run(args):
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_dir).resolve()
    f0_path = Path(args.f0_npy).resolve()
    if not f0_path.is_file():
        raise FileNotFoundError(f0_path)
    ids = torch.from_numpy(np.load(f0_path, allow_pickle=False)).long()
    if list(ids.shape) != [cfg["sequence_count"], cfg["sequence_length"]]:
        raise RuntimeError(f"F0_SHAPE_MISMATCH: {tuple(ids.shape)}")

    source_files = [Path(__file__).resolve(), cfg_path]
    config_seal = dict(
        status="FROZEN_BEFORE_MODEL_MEASUREMENTS",
        experiment_id=cfg["experiment_id"],
        authority_commit=cfg["scientific_authority_commit"],
        config_sha256=sha256_file(cfg_path),
        source_sha256={str(p): sha256_file(p) for p in source_files},
        f0_path=str(f0_path),
        f0_sha256=sha256_file(f0_path),
        model_dir=str(model_dir),
        model_revision=cfg["model_revision"],
        created_unix=time.time(),
        outcome_dependent_science_changes=0,
        training_updates_before_seal=0,
    )
    json_dump_new(out / "STK_CONFIG_SEAL.json", config_seal)

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network

    torch.set_num_threads(int(cfg["runtime"]["torch_num_threads"]))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(bool(cfg["runtime"]["deterministic_algorithms"]))
    torch.manual_seed(0)

    started = time.time()
    if model_dir.name != cfg["model_revision"]:
        raise RuntimeError(
            f"MODEL_REVISION_PATH_MISMATCH: expected directory basename {cfg['model_revision']}, got {model_dir.name}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval().cpu()
    model.requires_grad_(False)
    layer_index = int(cfg["layer_zero_based"])
    if (int(model.config.hidden_size), int(model.config.intermediate_size), len(model.model.layers)) != (576, 1536, 30):
        raise RuntimeError(
            f"MODEL_ARCHITECTURE_MISMATCH: {(model.config.hidden_size, model.config.intermediate_size, len(model.model.layers))}"
        )
    if layer_index >= len(model.model.layers):
        raise RuntimeError("LAYER_INDEX_OUT_OF_RANGE")
    dense = model.model.layers[layer_index].mlp
    hidden_size = int(model.config.hidden_size)
    if dense.down_proj.weight.shape[0] != hidden_size:
        raise RuntimeError("UNEXPECTED_FFN_SHAPE")

    prefix_hidden = cache_prefix(model, ids, layer_index)
    baseline_losses = []
    with torch.no_grad():
        for row, h0 in zip(ids, prefix_hidden):
            baseline_losses.append(float(suffix_loss(model, h0.clone(), row[None], layer_index)))

    h, phi, g = capture_tangent_data(model, dense, ids, prefix_hidden, layer_index)
    routing = cfg["routing"]
    moe = SparseTop2PrivateMoE(
        dense,
        hidden_size=hidden_size,
        delta=-float(routing["inactive_logit_offset"]),
        contrast=routing["contrast"],
    )
    model.model.layers[layer_index].mlp = moe
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    expected_trainable = [
        f"model.layers.{layer_index}.mlp.M",
        f"model.layers.{layer_index}.mlp.U",
    ]
    if trainable != expected_trainable:
        raise RuntimeError(f"TRAINABLE_PARAMETER_SET_MISMATCH: {trainable}")

    eq = origin_equivalence_and_nulls(dense, moe, h[: ids.shape[1]], cfg["preflight_tolerances"])
    wrapper_losses = []
    with torch.no_grad():
        for row, h0 in zip(ids, prefix_hidden):
            wrapper_losses.append(float(suffix_loss(model, h0.clone(), row[None], layer_index)))
    loss_err = float(np.max(np.abs(np.array(wrapper_losses) - np.array(baseline_losses))))
    if loss_err > float(cfg["preflight_tolerances"]["origin_loss_absolute_max"]):
        raise RuntimeError(f"ORIGIN_LOSS_EQUIVALENCE_FAIL: {loss_err}")
    eq["origin_bank_loss_max_abs"] = loss_err

    origin_support_parts = [moe.support_stats(x) for x in [h]]
    origin_support = origin_support_parts[0]
    if origin_support["turnover_tokens"] != 0:
        raise RuntimeError("ORIGIN_SUPPORT_NOT_FIXED")
    if origin_support["minimum_margin"] < float(routing["minimum_allowed_top2_top3_margin"]):
        raise RuntimeError("ORIGIN_SUPPORT_MARGIN_FAIL")

    coefficient = 0.5
    op = CrossOperator(h, phi, g, coefficient=coefficient)
    svals, uvecs, vvecs, residuals, iterations = leading_spectrum(op, cfg["spectrum"])
    spectrum_path = out / "STK_CROSS_HESSIAN_SPECTRUM.npz"
    np.savez(
        spectrum_path,
        s=svals.numpy(),
        u=uvecs.numpy(),
        v=vvecs.numpy(),
        relative_residuals=np.array(residuals),
        iterations=np.array(iterations),
        coefficient=np.array(coefficient),
    )
    hvp = hvp_validation(
        model,
        moe,
        h,
        phi,
        g,
        ids,
        prefix_hidden,
        layer_index,
        coefficient,
        float(cfg["spectrum"]["hvp_relative_error_max"]),
    )

    s1 = float(svals[0])
    u1 = uvecs[:, 0].float()
    v1 = vvecs[:, :, 0].float()
    predictions = [
        predicted_cell(
            s1,
            eps,
            float(cfg["dynamics"]["dimensionless_expert_step_alpha"]),
            float(cfg["dynamics"]["r_eta"]),
        )
        for eps in cfg["dynamics"]["epsilon"]
    ]
    preflight = dict(
        status="PASS",
        training_updates_before_seal=0,
        origin_equivalence=eq,
        origin_support=origin_support,
        mixed_operator_coefficient=coefficient,
        analytic_beta_derivative_at_origin=0.5,
        leading_singular_value=s1,
        leading_relative_residual=float(residuals[0]),
        spectrum_iterations=int(iterations),
        hvp_validation=hvp,
        spectrum_sha256=sha256_file(spectrum_path),
        predictions=predictions,
        predictions_frozen_before_nonlinear_trajectories=True,
        source_sha256={str(p): sha256_file(p) for p in source_files},
        config_sha256=sha256_file(cfg_path),
        f0_sha256=sha256_file(f0_path),
    )
    json_dump_new(out / "STK_PREFLIGHT_SEAL.json", preflight)

    rows = []
    horizon = int(cfg["dynamics"]["horizon_updates"])
    a0 = float(cfg["dynamics"]["initial_joint_amplitude"])
    min_margin_gate = float(cfg["primary_acceptance"]["minimum_top2_top3_margin"])
    for cell_index, pred in enumerate(predictions):
        for sign in cfg["dynamics"]["signs"]:
            M = (sign * a0 * pred["e_M"] * v1).clone()
            U = (sign * a0 * pred["e_U"] * u1).clone()
            with torch.no_grad():
                moe.M.copy_(M)
                moe.U.copy_(U)
            initial_full = math.hypot(vector_norm(M), vector_norm(U))
            for n in range(horizon + 1):
                with torch.no_grad():
                    m = float((moe.M.detach().double() * v1.double()).sum())
                    ucoord = float((moe.U.detach().double() * u1.double()).sum())
                    mn = vector_norm(moe.M.detach())
                    un = vector_norm(moe.U.detach())
                    full = math.hypot(mn, un)
                    alead = math.hypot(m, ucoord)
                    capture = (alead * alead) / max(full * full, 1e-30)
                    align = ((pred["e_M"] * m + pred["e_U"] * ucoord) ** 2) / max(alead * alead, 1e-30)
                    support = moe.support_stats(h)
                    if support["turnover_tokens"] != 0:
                        raise RuntimeError(
                            f"SUPPORT_TURNOVER_FAIL cell={cell_index} sign={sign} n={n} count={support['turnover_tokens']}"
                        )
                    if support["minimum_margin"] < min_margin_gate:
                        raise RuntimeError(
                            f"SUPPORT_MARGIN_FAIL cell={cell_index} sign={sign} n={n} margin={support['minimum_margin']}"
                        )
                row = dict(
                    cell=cell_index,
                    epsilon=pred["epsilon"],
                    sign=sign,
                    update=n,
                    m1=m,
                    u1=ucoord,
                    A_lead=alead,
                    M_norm=mn,
                    U_norm=un,
                    A_full=full,
                    leading_pair_capture=capture,
                    predicted_eigenvector_alignment=align,
                    full_amplitude_ratio=full / initial_full,
                    support_turnover_tokens=support["turnover_tokens"],
                    support_total_tokens=support["total_tokens"],
                    minimum_top2_top3_margin=support["minimum_margin"],
                    median_top2_top3_margin=support["median_margin"],
                    causal_lm_loss=None,
                )
                if n == horizon:
                    with torch.no_grad():
                        losses = [
                            float(suffix_loss(model, h0.clone(), r[None], layer_index))
                            for r, h0 in zip(ids, prefix_hidden)
                        ]
                    row["causal_lm_loss"] = float(np.mean(losses))
                rows.append(row)
                if n < horizon:
                    gm, gu, _ = full_bank_gradient(
                        model,
                        moe,
                        ids,
                        prefix_hidden,
                        layer_index,
                        pred["mu"],
                        pred["lambda_"],
                    )
                    with torch.no_grad():
                        moe.M.copy_((moe.M.detach().double() - pred["eta_X"] * gm).float())
                        moe.U.copy_((moe.U.detach().double() - pred["eta_U"] * gu).float())
            print(f"trajectory cell={cell_index} eps={pred['epsilon']:+.3f} sign={sign:+d} complete", flush=True)

    csv_dump_new(out / "STK_TRAJECTORIES.csv", rows)

    rate_rows = []
    verdict_cells = []
    acc = cfg["primary_acceptance"]
    for cell_index, pred in enumerate(predictions):
        sign_fits = []
        cell_rows = [r for r in rows if r["cell"] == cell_index]
        for sign in cfg["dynamics"]["signs"]:
            arm = [r for r in cell_rows if r["sign"] == sign]
            rate, r2 = fit_rate(arm)
            sign_fits.append((sign, rate, r2))
        pair_rate = float(np.mean([x[1] for x in sign_fits]))
        rel = abs(pair_rate - pred["gamma_pred"]) / max(abs(pred["gamma_pred"]), 1e-30)
        sign_rates_correct = all((rate > 0) == (pred["gamma_pred"] > 0) for _, rate, _ in sign_fits)
        correct = sign_rates_correct and ((pair_rate > 0) == (pred["gamma_pred"] > 0))
        min_capture = min(r["leading_pair_capture"] for r in cell_rows)
        turnover = sum(r["support_turnover_tokens"] for r in cell_rows)
        min_margin = min(r["minimum_top2_top3_margin"] for r in cell_rows)
        passed = (
            correct
            and rel <= float(acc["pair_mean_relative_rate_error_max"])
            and min_capture >= float(acc["minimum_leading_pair_capture"])
            and turnover == int(acc["support_turnover_must_equal"])
            and min_margin >= float(acc["minimum_top2_top3_margin"])
        )
        for sign, rate, r2 in sign_fits:
            rate_rows.append(
                dict(
                    cell=cell_index,
                    epsilon=pred["epsilon"],
                    sign=sign,
                    gamma_pred=pred["gamma_pred"],
                    gamma_obs=rate,
                    r2=r2,
                    pair_mean_gamma_obs=pair_rate,
                    pair_mean_relative_error=rel,
                    cell_pass=passed,
                )
            )
        verdict_cells.append(
            dict(
                cell=cell_index,
                epsilon=pred["epsilon"],
                gamma_pred=pred["gamma_pred"],
                pair_mean_gamma_obs=pair_rate,
                pair_mean_relative_error=rel,
                correct_sign=correct,
                sign_rates_correct=sign_rates_correct,
                minimum_leading_pair_capture=min_capture,
                support_turnover_tokens=turnover,
                minimum_top2_top3_margin=min_margin,
                passed=passed,
            )
        )
    csv_dump_new(out / "STK_GROWTH_RATES.csv", rate_rows)
    overall = all(x["passed"] for x in verdict_cells)
    decision = dict(
        status="SEALED",
        verdict="SPARSE_FIXED_SUPPORT_BRIDGE_SUPPORTED" if overall else "SPARSE_FIXED_SUPPORT_BRIDGE_NOT_SUPPORTED",
        cells=verdict_cells,
        acceptance=acc,
        interpretation_ceiling=(
            "This is a controlled renormalized top-2 fixed-support bridge. It does not reproduce OLMoE's exact no-renormalization routing convention and does not establish native onset."
        ),
        elapsed_seconds=time.time() - started,
        runtime=dict(
            python=sys.version,
            platform=platform.platform(),
            torch=torch.__version__,
            transformers=__import__("transformers").__version__,
            device="cpu",
        ),
        artifact_sha256={
            "STK_CONFIG_SEAL.json": sha256_file(out / "STK_CONFIG_SEAL.json"),
            "STK_PREFLIGHT_SEAL.json": sha256_file(out / "STK_PREFLIGHT_SEAL.json"),
            "STK_CROSS_HESSIAN_SPECTRUM.npz": sha256_file(spectrum_path),
            "STK_TRAJECTORIES.csv": sha256_file(out / "STK_TRAJECTORIES.csv"),
            "STK_GROWTH_RATES.csv": sha256_file(out / "STK_GROWTH_RATES.csv"),
        },
    )
    json_dump_new(out / "STK_PRIMARY_DECISION.json", decision)

    manifest_files = sorted(p for p in out.iterdir() if p.is_file())
    manifest = "".join(f"{sha256_file(p)}  {p.name}\n" for p in manifest_files)
    (out / "SHA256SUMS_STK.txt").write_text(manifest)
    print(json.dumps({"verdict": decision["verdict"], "cells": verdict_cells}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--f0-npy", required=True)
    ap.add_argument("--output-dir", required=True)
    run(ap.parse_args())
