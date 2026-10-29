from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

EXPECTED_PARENT_RUNNER_SHA256 = "b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744"
Q_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIT_Q = (4, 8, 16, 32)
PROTOCOL_ID = "Project1_E5_AmplitudeDomainLocality_v1_2026-10-29"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jwrite(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_parent(source_root: Path):
    scripts = source_root / "scripts"
    runner = scripts / "run_generic_local_dynamics.py"
    core = scripts / "generic_core.py"
    protocol = source_root / "protocol.json"
    for p in (
        runner, core, protocol,
        source_root / "inputs" / "F0.npy",
        source_root / "inputs" / "P2H_CROSS_HESSIAN_SPECTRUM.npz",
    ):
        if not p.exists():
            raise FileNotFoundError(f"required parent asset missing: {p}")
    got = sha256_file(runner)
    if got != EXPECTED_PARENT_RUNNER_SHA256:
        raise RuntimeError(f"parent runner SHA-256 mismatch: {got} != {EXPECTED_PARENT_RUNNER_SHA256}")
    import sys
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("project1_generic_parent", runner)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import parent runner")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, got, sha256_file(core), sha256_file(protocol)


def finite_tensor(*xs: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(x).all()) for x in xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    source_root = args.source_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / x).exists() for x in ("amplitude_domain_scan.csv", "amplitude_domain_scan_manifest.json")):
        raise FileExistsError("refusing to overwrite an existing E5 scientific result directory")

    t0 = time.monotonic()
    base, runner_sha, core_sha, protocol_sha = load_parent(source_root)
    P = base.P
    if P["protocol_id"] != "Project1_GenericLocalDynamics_PRX_v1_2026-10-27":
        raise RuntimeError(f"unexpected parent protocol: {P['protocol_id']}")
    if tuple(int(x) for x in P["direction_seeds"]) != tuple(range(2026102701, 2026102709)):
        raise RuntimeError("parent direction seeds differ from frozen E5 design")

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    model_dir = base.discover_model()
    ids = torch.from_numpy(np.load(base.F0_PATH, allow_pickle=False))
    spec_np = np.load(base.SPEC_PATH, allow_pickle=False)
    s = torch.from_numpy(spec_np["s"].copy())
    u = torch.from_numpy(spec_np["u"].copy())
    v = torch.from_numpy(spec_np["v"].copy())

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    model.requires_grad_(False)
    if (model.config.hidden_size, model.config.intermediate_size, model.config.num_hidden_layers) != (576, 1536, 30):
        raise RuntimeError("model architecture mismatch")
    dense = model.model.layers[20].mlp

    h, phi, g = base.capture_operator(model, dense, ids)
    op = base.FrozenCOperator(h, phi, g)
    rv = base.norm_t(op.c(v[:, :, 0]) - s[0] * u[:, 0]) / float(s[0])
    ru = base.norm_t(op.ct(u[:, 0]) - s[0] * v[:, :, 0]) / float(s[0])
    if max(rv, ru) > 5e-5:
        raise RuntimeError(f"parent operator smoke mismatch: c(v)={rv}, ct(u)={ru}")

    A0 = base.aligned_central_amplitude(float(P["a0_aligned_raw"]), base.MU)
    delta = 1e-12 * A0
    directions = []
    for seed in P["direction_seeds"]:
        Xd, Ud = base.make_generic_direction(int(seed), (576, 1536), (576,), base.MU)
        directions.append((Xd.astype(np.float32), Ud.astype(np.float32)))

    moe = base.PrivateMoE(dense)
    model.model.layers[20].mlp = moe
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable != ["model.layers.20.mlp.M", "model.layers.20.mlp.U"]:
        raise RuntimeError(f"trainable parameter mismatch: {trainable}")
    hidden = [base.prefix_one(model, row) for row in ids]

    rows: list[dict] = []
    scientific_nonfinite: list[dict] = []
    gradient_evals = 0

    for side, eps in (("stable", -0.08), ("unstable", 0.08)):
        lam = base.MU * (1.0 - eps)
        for di, (Xd, Ud) in enumerate(directions, 1):
            seed = int(P["direction_seeds"][di - 1])
            for q in Q_GRID:
                A = float(q * A0)
                X0 = torch.from_numpy((A * Xd).astype(np.float32))
                U0 = torch.from_numpy((A * Ud).astype(np.float32))
                z0 = (X0, U0)
                pred1 = base.linear_step(z0, op, lam)
                diag_initial = base.cubic_diag(U0, h, delta)
                CX0 = op.c(X0)
                CTU0 = op.ct(U0)
                diag0 = base.combined_norm(-base.ETA_X * base.MU * X0, -base.ETA_U * lam * U0)
                offdiag0 = base.combined_norm(base.ETA_X * CTU0, base.ETA_U * CX0)
                rec = {
                    "side": side, "epsilon": eps, "lambda_": lam,
                    "direction": di, "seed": seed, "q": int(q), "A": A,
                    "status": "PENDING",
                    "coupling_CX0_raw": base.norm_t(CX0),
                    "coupling_CT_U0_raw": base.norm_t(CTU0),
                    "first_step_diag_raw": diag0,
                    "first_step_offdiag_raw": offdiag0,
                    "coupling_activity_ratio": offdiag0 / (diag0 + delta),
                }
                rec.update({f"initial_{k}": val for k, val in diag_initial.items()})
                gm, gu, nll = base.bank_gradient(model, moe, ids, hidden, z0, lam)
                gradient_evals += 1
                rec["F0_NLL_before_update"] = nll
                if not finite_tensor(gm, gu):
                    rec["status"] = "NONFINITE_GRADIENT"
                    scientific_nonfinite.append({"side": side, "direction": di, "q": q, "status": rec["status"]})
                    rows.append(rec)
                    continue
                state1 = ((X0.double() - base.ETA_X * gm).float(), (U0.double() - base.ETA_U * gu).float())
                if not finite_tensor(state1[0], state1[1]):
                    rec["status"] = "NONFINITE_STATE"
                    scientific_nonfinite.append({"side": side, "direction": di, "q": q, "status": rec["status"]})
                    rows.append(rec)
                    continue
                met = base.project_metrics(state1, pred1, z0, v, u, delta)
                rec.update({k: base.finite_or_none(val) for k, val in met.items()})
                post_diag = base.cubic_diag(state1[1], h, delta)
                rec.update({f"post_{k}": val for k, val in post_diag.items()})
                rec["status"] = "OK"
                rows.append(rec)
                print(f"COMPLETE {side} direction={di} q={q} elapsed={time.monotonic()-t0:.1f}s", flush=True)

    if gradient_evals != 2 * 8 * len(Q_GRID):
        raise RuntimeError(f"gradient evaluation count mismatch: {gradient_evals}")

    write_csv(out / "amplitude_domain_scan.csv", rows)
    manifest = {
        "status": "COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": P["protocol_id"],
        "parent_runner_sha256": runner_sha,
        "parent_generic_core_sha256": core_sha,
        "parent_protocol_sha256": protocol_sha,
        "model_dir": str(model_dir),
        "model_commit": P["model_commit"],
        "layer_zero_based": 20,
        "F0_shape": list(ids.shape),
        "F0_file_sha256": sha256_file(base.F0_PATH),
        "spectrum_file_sha256": sha256_file(base.SPEC_PATH),
        "mu": base.MU,
        "eta_X": base.ETA_X,
        "eta_U": base.ETA_U,
        "A_align": A0,
        "q_grid": list(Q_GRID),
        "quadratic_fit_q": list(FIT_Q),
        "direction_seeds": [int(x) for x in P["direction_seeds"]],
        "epsilon": {"stable": -0.08, "unstable": 0.08},
        "gradient_evaluations": gradient_evals,
        "expected_gradient_evaluations": 2 * 8 * len(Q_GRID),
        "operator_pair1_residuals": {"c_v": rv, "ct_u": ru},
        "scientific_nonfinite": scientific_nonfinite,
        "elapsed_seconds": time.monotonic() - t0,
        "output_sha256": {"amplitude_domain_scan.csv": sha256_file(out / "amplitude_domain_scan.csv")},
        "interpretation_boundary": "single-step amplitude-domain scan; no long-time saturation or specialization claim",
    }
    jwrite(out / "amplitude_domain_scan_manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "gradient_evaluations": gradient_evals, "nonfinite_cells": len(scientific_nonfinite)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
