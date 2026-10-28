# PhysicsForAI Project 1 — source and reproducibility release

Code and project-owned reproducibility material accompanying *Expert-Specific Response in Mixture-of-Experts*.

The current public tree is scoped to analyses retained in the active manuscript: copied-state sector selection, full expert-block tangent validation, expert-side selection, background-dependent mode conversion, duplication-state local instability/dynamics, and finite-time batch-conditioned propagation. Experimental programs removed from the manuscript are not retained in the current branch.

`experiments/` contains sanitized historical execution/analysis source only for retained manuscript families. `PUBLIC_CODE_MANIFEST.tsv` summarizes coverage and `PUBLIC_SOURCE_BINDINGS.tsv` binds individual public files to archived source records. `reference_implementation/` contains clean mathematical implementations and data-independent tests for retained operators.

## Verification

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
```

The retained post-baseline records are E1 and E4a/E4b. E2/E3 and the trained sparse-MoE fixed-routing/response/utility program were removed from the active manuscript and have been withdrawn from the current public branch.

## Generic-local provenance status

The active manuscript also contains the 2026-10-27 generic non-mode-aligned local-dynamics test. Its sealed numerical outputs and manifest are retained in the scientific archive. The exact historically executed model-facing source has not yet been recovered in the public-release audit, so no reconstructed source is presented as the historical execution source.

Large model-facing caches, gradients, full-state captures, copyrighted corpus text/token banks, and third-party model weights are not redistributed. Full model-facing reruns require pinned upstream models/datasets and retained scientific inputs. Project-owned code is MIT licensed; upstream assets retain their own licenses.
