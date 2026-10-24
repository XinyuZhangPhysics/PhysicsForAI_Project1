# PhysicsForAI Project 1 — source and reproducibility release

Code and project-owned derived reproducibility material accompanying *Expert-Specific Response in Mixture-of-Experts*.

The public release contains implementation code, configuration templates, tests, environment specifications, provenance metadata, and a reader-facing reproducibility layer under `reproducibility/`. That directory publishes frozen headline numerical summaries, claim-to-source mappings, pinned model/data revisions, protocol/basis hashes, statistical-unit conventions, and the final E1–E4 follow-up registry used by the integrated manuscript revision.

`experiments/` contains sanitized versions of retained historical experiment and analysis source snapshots. Every script is bound to the corresponding archived source record in `PUBLIC_SOURCE_BINDINGS.tsv`; `PUBLIC_CODE_MANIFEST.tsv` summarizes coverage. Path substitutions were checked for code equivalence and do not alter the scientific algorithms.

`reference_implementation/` contains a clean implementation of the mathematical operators used in the paper, unit tests, and optional reconstruction scripts. The clean implementation is not represented as the historically executed end-to-end model pipeline.

## Source-level verification

Python 3.12 is the tested interpreter.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
```

These tests exercise the selection rules, mixed pushforward, local growth map, native response metric, utility action, and finite-time ordering algebra without requiring experimental output files.

## Public derived numerical material

See `reproducibility/README.md`. The public derived layer is intentionally limited to project-owned summaries and provenance that can be redistributed without copying third-party assets. It includes the headline values needed to audit the paper's main quantitative claims and the mapping from each analysis family to its retained public source group.

The 2026-10-24 integrated revision adds the frozen E1–E4 follow-up record. The new tests separate population operator shift from realized-sample conditioning (E1), cross-fitted signed descent from second-moment visibility (E2), held-out from independent reserve utility (E3), and the calibrated formation law from damping and stationary-background deformations (E4a/E4b). See `reproducibility/postpaper_E1_E4.tsv` and `reproducibility/postpaper_protocols.tsv`.

Large per-document arrays, model-facing caches, gradients, full-state captures, copyrighted corpus text/token banks, and third-party model weights are not redistributed here. Exact larger post-paper result bundles and execution-provenance records are retained in the versioned scientific archive and may be obtained from the author on reasonable request. Third-party models and datasets must be obtained from their original providers under their respective licenses.

## Optional numerical reconstruction

The reconstruction scripts are retained as source code. When an authorized larger derived-data bundle is available, point the scripts to it with

```sh
export PROJECT1_DATA_DIR=/path/to/project1-derived-data
python reference_implementation/reproduce/reproduce_paper.py
```

The expected data schema and claim-to-code mapping are documented under `reference_implementation/docs/` and `reproducibility/`.

## Historical model-facing code

Use `PUBLIC_CODE_MANIFEST.tsv` to choose an experiment/source group and `PUBLIC_SOURCE_BINDINGS.tsv` for individual scripts. Full model-facing reruns require the pinned upstream models/corpora and the retained large intermediate inputs/execution environments. Do not import historical execution scripts casually: some execute at module top level.

Project-owned code is MIT licensed; upstream assets retain their own licenses.
