from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Q_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIT_Q = (4, 8, 16, 32)


def jwrite(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def slope(xs, ys):
    x = np.log(np.asarray(xs, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y-y.mean())**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0
    return float(coef[1]), float(coef[0]), r2


def first_cross(qs, vals, threshold):
    for q, v in zip(qs, vals):
        if math.isfinite(v) and v >= threshold:
            return int(q)
    return None


def stats(x):
    a = np.asarray([float(v) for v in x if pd.notna(v) and math.isfinite(float(v))], dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "median": float(np.median(a)),
        "q1": float(np.quantile(a, 0.25)),
        "q3": float(np.quantile(a, 0.75)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    required = {"side", "direction", "q", "status", "error", "prediction_error_abs_central", "linear_increment_central", "initial_max_abs_router_field", "initial_rms_relative_tanh_remainder"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if sorted(df.q.unique().tolist()) != list(Q_GRID):
        raise ValueError(f"q grid differs from frozen protocol: {sorted(df.q.unique().tolist())}")

    rows = []
    summary = {"protocol_decision": {}, "sides": {}, "fit_q": list(FIT_Q), "q_grid": list(Q_GRID)}
    all_primary_ok = True

    for side in ("stable", "unstable"):
        sdf = df[df.side == side]
        side_summary = {"q": {}, "direction_slopes": []}
        med_errors = []
        for q in Q_GRID:
            qdf = sdf[(sdf.q == q) & (sdf.status == "OK")]
            rec = {
                "side": side,
                "q": int(q),
                "ok_cells": int(len(qdf)),
                "error": stats(qdf.error),
                "abs_central_residual": stats(qdf.prediction_error_abs_central),
                "linear_increment_central": stats(qdf.linear_increment_central),
                "max_abs_router_field": stats(qdf.initial_max_abs_router_field),
                "rms_relative_tanh_remainder": stats(qdf.initial_rms_relative_tanh_remainder),
            }
            side_summary["q"][str(q)] = rec
            rows.append(rec)
            med_errors.append(rec["error"].get("median", float("nan")))

        fit_vals = [side_summary["q"][str(q)]["error"].get("median", float("nan")) for q in FIT_Q]
        fit_complete = all(math.isfinite(v) and v > 0 for v in fit_vals)
        alpha = intercept = r2 = None
        if fit_complete:
            alpha, intercept, r2 = slope(FIT_Q, fit_vals)

        for direction in range(1, 9):
            ddf = sdf[(sdf.direction == direction) & (sdf.q.isin(FIT_Q)) & (sdf.status == "OK")].sort_values("q")
            if list(ddf.q.astype(int)) == list(FIT_Q) and np.all(ddf.error.to_numpy(dtype=float) > 0):
                a, b, rr = slope(ddf.q.to_numpy(), ddf.error.to_numpy())
                side_summary["direction_slopes"].append({"direction": direction, "alpha": a, "r2": rr})

        q1 = first_cross(Q_GRID, med_errors, 0.01)
        q5 = first_cross(Q_GRID, med_errors, 0.05)
        anchor = side_summary["q"]["1"]["error"].get("median", float("nan"))
        complete_through_32 = all(side_summary["q"][str(q)]["ok_cells"] == 8 for q in (1, 2, 4, 8, 16, 32))
        anchor_pass = math.isfinite(anchor) and anchor < 5e-6
        slope_pass = alpha is not None and 1.5 <= alpha <= 2.5
        side_pass = anchor_pass and complete_through_32 and slope_pass
        all_primary_ok = all_primary_ok and side_pass

        side_summary.update({
            "quadratic_fit": {"alpha": alpha, "intercept": intercept, "r2": r2, "window": list(FIT_Q)},
            "anchor_q1_median_error": anchor,
            "anchor_pass": anchor_pass,
            "complete_through_q32": complete_through_32,
            "slope_pass": slope_pass,
            "primary_side_pass": side_pass,
            "q_1pct": q1,
            "q_1pct_statement": f"q_1pct={q1}" if q1 is not None else "q_1pct>256",
            "q_5pct": q5,
            "q_5pct_statement": f"q_5pct={q5}" if q5 is not None else "q_5pct>256",
        })
        summary["sides"][side] = side_summary

    summary["protocol_decision"] = {
        "label": "LOCALITY_SCALING_RESOLVED" if all_primary_ok else "LOCALITY_SCALING_NOT_RESOLVED",
        "rule": "q=1 median error <5e-6, all 8 directions finite through q=32, and fixed-window slope alpha in [1.5,2.5] on both sides",
    }

    flat = []
    for rec in rows:
        flat.append({
            "side": rec["side"],
            "q": rec["q"],
            "ok_cells": rec["ok_cells"],
            "error_median": rec["error"].get("median", ""),
            "error_q1": rec["error"].get("q1", ""),
            "error_q3": rec["error"].get("q3", ""),
            "abs_residual_median": rec["abs_central_residual"].get("median", ""),
            "linear_increment_median": rec["linear_increment_central"].get("median", ""),
            "max_router_field_median": rec["max_abs_router_field"].get("median", ""),
            "tanh_relative_remainder_median": rec["rms_relative_tanh_remainder"].get("median", ""),
        })
    pd.DataFrame(flat).to_csv(args.output / "amplitude_domain_summary.csv", index=False)
    jwrite(args.output / "amplitude_domain_summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(args.output / "Fig_amplitude_domain_scan.pdf") as pdf:
        fig, ax = plt.subplots()
        for side in ("stable", "unstable"):
            qs = np.asarray(Q_GRID, dtype=float)
            ys = np.asarray([summary["sides"][side]["q"][str(q)]["error"].get("median", np.nan) for q in Q_GRID])
            ax.loglog(qs, ys, marker="o", label=side)
        ref_q = np.asarray(FIT_Q, dtype=float)
        y4 = summary["sides"]["stable"]["q"]["4"]["error"].get("median", np.nan)
        if math.isfinite(float(y4)) and float(y4) > 0:
            ax.loglog(ref_q, float(y4)*(ref_q/4.0)**2, linestyle="--", label=r"$q^2$ reference")
        ax.axhline(0.01, linestyle=":", label="1% error")
        ax.axhline(0.05, linestyle=":", label="5% error")
        ax.set_xlabel(r"amplitude multiplier $q=A/A_{\rm align}$")
        ax.set_ylabel(r"median one-step relative error $e_1$")
        ax.set_title("Amplitude-domain locality scan")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        for side in ("stable", "unstable"):
            qs = np.asarray(Q_GRID, dtype=float)
            ys = np.asarray([summary["sides"][side]["q"][str(q)]["error"].get("median", np.nan)/(q*q) for q in Q_GRID])
            ax.semilogx(qs, ys, marker="o", label=side)
        ax.set_xlabel(r"amplitude multiplier $q$")
        ax.set_ylabel(r"median $e_1/q^2$")
        ax.set_title("Quadratic-rescaled local-map error")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        for side in ("stable", "unstable"):
            qs = np.asarray(Q_GRID, dtype=float)
            ys = np.asarray([summary["sides"][side]["q"][str(q)]["max_abs_router_field"].get("median", np.nan) for q in Q_GRID])
            ax.loglog(qs, ys, marker="o", label=side)
        ax.set_xlabel(r"amplitude multiplier $q$")
        ax.set_ylabel("median max |router field|")
        ax.set_title("Router-field scale across the amplitude scan")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(summary["protocol_decision"], indent=2))
    for side in ("stable", "unstable"):
        s = summary["sides"][side]
        print(side, "alpha=", s["quadratic_fit"]["alpha"], s["q_1pct_statement"], s["q_5pct_statement"])


if __name__ == "__main__":
    main()
