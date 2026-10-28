# PhysicsForAI Project 1 — source and reproducibility release

Code and project-owned reproducibility material accompanying *Expert-Specific Response in Mixture-of-Experts*.

The 2026-10-28 public tree is synchronized to the active manuscript. It contains the retained copied-state sector-selection, full-tangent, expert-side, background-conversion, duplication-state local-dynamics, and finite-time analysis families. The fixed-routing OLMoE, trained sparse-MoE tomography/objective-response, and utility programs removed from the manuscript remain withdrawn from the current branch.

## Generic non-mode-aligned local dynamics

The historically executed source for the 2026-10-27 generic-local experiment has been recovered and cryptographically bound to its archived execution record. The public-safe source/provenance handoff is:

`experiments/genericlocal/GENERIC_LOCAL_PUBLIC_SYNC_READY.zip`

Its seven Python files are byte-identical to the recovered historical package. Only one launcher path and one preferred-model path string were sanitized; the archive records original/public hashes and the reversible substitutions. The public archive is source/provenance only: model weights, corpus text, F0/input arrays, binary results, caches, credentials, and private paths are excluded.

Compact manuscript-facing evidence is under `reproducibility/final/`. Full large scientific tables remain in the retained scientific archive and are bound by SHA-256 in `PUBLIC_EVIDENCE_BINDINGS.tsv`.

## Verification

Python 3.12 is the tested interpreter for the clean operator checks:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
```

The clean tests cover expert-index selection, signed mixed pushforward, mixed-Hessian adjointness/exact-discrete local growth, and finite-time ordering/cumulant identities. The sanitized generic-local historical package is not a standalone rerun bundle because its original sealed package also requires retained private inputs and the pinned upstream model.

## Evidence identity

The growth figure preserves the 11-cell P2J primary aggregate and the separately frozen P2L confirmation; P2L is never substituted into the primary aggregate. The stationary-background common-absolute-`lambda` record explicitly labels the nonzero-background values at `lambda=lambda_c(0)` as theory-only references, not measured trajectories. The diagonal-only comparison is explicitly a leading-four projected-subspace ablation, not a full-state diagonal-only reconstruction.

Project-owned code/documentation are MIT licensed; upstream assets retain their original licenses.
