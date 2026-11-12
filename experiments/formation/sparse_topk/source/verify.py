#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(out_dir):
    out = Path(out_dir)
    required = [
        "STK_CONFIG_SEAL.json",
        "STK_PREFLIGHT_SEAL.json",
        "STK_CROSS_HESSIAN_SPECTRUM.npz",
        "STK_TRAJECTORIES.csv",
        "STK_GROWTH_RATES.csv",
        "STK_PRIMARY_DECISION.json",
        "SHA256SUMS_STK.txt",
    ]
    assert all((out / n).is_file() for n in required), "missing deliverable"

    decision = json.loads((out / "STK_PRIMARY_DECISION.json").read_text())
    pre = json.loads((out / "STK_PREFLIGHT_SEAL.json").read_text())
    assert pre["status"] == "PASS"
    assert pre["training_updates_before_seal"] == 0
    assert pre["predictions_frozen_before_nonlinear_trajectories"] is True
    assert pre["origin_support"]["turnover_tokens"] == 0
    assert pre["origin_support"]["minimum_margin"] >= 4.0
    assert pre["leading_relative_residual"] <= 1e-4
    assert pre["hvp_validation"]["relative_error"] <= 1e-4

    spec = np.load(out / "STK_CROSS_HESSIAN_SPECTRUM.npz")
    assert float(spec["s"][0]) > 0
    assert float(spec["relative_residuals"][0]) <= 1e-4
    assert abs(float(spec["coefficient"]) - 0.5) < 1e-15

    with (out / "STK_TRAJECTORIES.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 36, f"expected 36 trajectory rows, got {len(rows)}"
    keys = {(int(r["cell"]), int(r["sign"]), int(r["update"])) for r in rows}
    assert keys == {(c, s, n) for c in [0, 1] for s in [-1, 1] for n in range(9)}
    assert all(int(r["support_turnover_tokens"]) == 0 for r in rows)
    assert min(float(r["minimum_top2_top3_margin"]) for r in rows) >= 4.0
    assert min(float(r["leading_pair_capture"]) for r in rows) >= 0.99

    for n, h in decision["artifact_sha256"].items():
        assert sha(out / n) == h, n

    manifest = {}
    for line in (out / "SHA256SUMS_STK.txt").read_text().splitlines():
        h, n = line.split(maxsplit=1)
        manifest[n] = h
        assert sha(out / n) == h, n

    result = {
        "status": "PASS",
        "scientific_verdict": decision["verdict"],
        "leading_singular_value": pre["leading_singular_value"],
        "cells": decision["cells"],
        "verified_files": len(manifest),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    main(ap.parse_args().output_dir)
