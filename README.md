# PhysicsForAI Project 1 — 2026-10-21

Code accompanying *Expert-Private Response in Mixture-of-Experts*.

`experiments/` contains sanitized versions of actual retained experiment and analysis source snapshots. Every script is bound to the private archived source path and SHA256 in `PUBLIC_SOURCE_BINDINGS.tsv`; `PUBLIC_CODE_MANIFEST.tsv` summarizes experiment coverage. Path substitutions are reversible and were checked with Python AST equivalence; scientific algorithms were not rewritten. A few historical source records are candidates rather than execution-bound; their binding column states this explicitly.

`reference_implementation/` preserves the previously prepared clean mathematics, compact derived data, tests and table/figure reconstruction. It is a later clean implementation and is not represented as historical execution source.

## Frozen numerical reconstruction

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
python reference_implementation/reproduce/reproduce_paper.py
```

## Historical model-facing code

Use `PUBLIC_CODE_MANIFEST.tsv` to choose the experiment/run source group and `PUBLIC_SOURCE_BINDINGS.tsv` for individual scripts. Obtain pinned upstream weights/corpus under their licenses and the necessary private raw inputs. Recreate the historical relative input layout in a writable working directory as identified by the archived source and run seals. Copy `configs/paths.template.json` to a local configuration, fill each required path for that script, and set `PROJECT1_PATHS_FILE`. Use writable working copies for output paths. The private archive's per-experiment README and protocol/config seals identify entry points and the historical environment. Install `requirements-experiments.txt` in a separate environment matched to those seals. Do not import execution scripts casually: some execute at module top level. No model-facing experiment is automatically run by the numerical reconstruction commands above.

Raw model weights, copyrighted text/token banks, gradients, full-state captures and private research notes are excluded. Project-owned code is MIT; project-owned derived data are CC BY 4.0; upstream assets keep their own licenses. Full model-facing reruns require those external/private assets. Data DOI pending. The 2026-10-21 release snapshot is the manuscript-content reference; PDF pixel identity is a separate build issue.
