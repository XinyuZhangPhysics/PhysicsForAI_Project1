# PhysicsForAI Project 1 — source release

Code accompanying *Expert-Specific Response in Mixture-of-Experts*.

The current public release is **source-only**: it contains implementation code, configuration templates, tests, environment specifications, and provenance metadata, but no project-owned experimental output arrays, frozen result tables, or intermediate scientific captures.

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

## Optional numerical reconstruction

The reconstruction scripts are retained as source code, but their project-owned numerical inputs are not bundled in this repository. When an authorized data bundle is available, point the scripts to it with

```sh
export PROJECT1_DATA_DIR=/path/to/project1-derived-data
python reference_implementation/reproduce/reproduce_paper.py
```

The expected data schema and claim-to-code mapping are documented under `reference_implementation/docs/`. Project-owned numerical outputs may be provided by the author on reasonable request. Third-party model weights and corpora must be obtained from their original providers under their respective licenses.

## Historical model-facing code

Use `PUBLIC_CODE_MANIFEST.tsv` to choose an experiment/source group and `PUBLIC_SOURCE_BINDINGS.tsv` for individual scripts. Full model-facing reruns require the pinned upstream models/corpora and the retained private raw inputs and execution environments. Do not import historical execution scripts casually: some execute at module top level.

No model weights, copyrighted corpus text/token banks, gradients, full-state captures, or private research notes are redistributed. Project-owned code is MIT licensed; upstream assets retain their own licenses.
