# PhysicsForAI Project 1 — source and reproducibility release

Code and project-owned reproducibility material accompanying *Expert-Specific Response in Mixture-of-Experts*.

The 2026-10-29 public tree is synchronized to the active manuscript. It contains the retained copied-state sector-selection, full-tangent, expert-side, background-conversion, duplication-state local-dynamics, and finite-time analysis families. The fixed-routing OLMoE, trained sparse-MoE tomography/objective-response, and utility programs removed from the manuscript remain withdrawn from the current branch.

## Generic non-mode-aligned local dynamics

The recovered historical source and complete public-safe provenance handoff are canonical under `experiments/genericlocal/`. The full runner is `public_sanitized_source/scripts/run_generic_local_dynamics.py` with SHA-256 `b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744`. All seven Python/test files retain exact historical bytes; only two launcher/config path strings were mechanically substituted. Original and public hashes, redacted execution logs, manifests, seals, audit records and source bindings are published together. See `experiments/genericlocal/README_PUBLIC_SYNC.md`.

Compact generic-local records are under `evidence/generic_local_dynamics/`; diagonal, spectrum, growth and stationary-background records are under `analysis/astra_r2_node1/`, `analysis/astra_r2_node3/` and `analysis/astra_r2_node6/`. `reproducibility/PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv` records original/public hashes for all 48 canonical handoff and evidence files. The v15 manifest uses the existing hash-bound public account-path normalization; no scientific field changed. Large scientific arrays and the full trajectory table remain in retained scientific storage.

## Verification

This completion audit passed with Python 3.14.7, NumPy 2.5.3 and SciPy 1.18.1. The earlier clean operator release was tested with Python 3.12. To check this release:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
```

The clean tests cover expert-index selection, signed mixed pushforward, mixed-Hessian adjointness/exact-discrete local growth, and finite-time ordering/cumulant identities. The published generic-local source is not a standalone sealed rerun bundle because the historical execution also requires retained private F0/operator inputs, package seals, and the pinned upstream model.

## Evidence identity

The growth figure preserves the 11-cell P2J primary aggregate and the separately frozen P2L confirmation; P2L is never substituted into the primary aggregate. The stationary-background common-absolute-`lambda` record explicitly labels the nonzero-background values at `lambda=lambda_c(0)` as theory-only references, not measured trajectories. The diagonal-only comparison is explicitly a leading-four projected-subspace ablation, not a full-state diagonal-only reconstruction.

Project-owned code/documentation are MIT licensed; upstream assets retain their original licenses.
