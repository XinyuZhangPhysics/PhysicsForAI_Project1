# Expert-Specific Response in Mixture-of-Experts — reference implementation

This directory contains the clean mathematical implementation used to audit the operators and numerical analysis in the paper. The current public release is source-only: experimental output arrays, frozen result tables, and intermediate captures are not bundled here.

## Code-level checks

From `reference_implementation/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

The unit tests require no experimental data. They independently check the expert-index projectors and selection rule, the signed mixed pushforward, cross-Hessian adjointness and exact-discrete growth, the native response metric, the router-bias action, and finite-time ordering/cumulant identities.

## Optional paper-level reconstruction

The reconstruction code is retained so that the reported numerical summaries can be regenerated from an authorized project-owned derived-data bundle. The data bundle is not part of this repository. Set

```sh
export PROJECT1_DATA_DIR=/path/to/project1-derived-data
python reproduce/reproduce_paper.py
```

The script fails explicitly if the external data directory is absent. The required filenames and claim-to-code correspondence are summarized in `docs/PAPER_RESULT_MAP.md`. Project-owned numerical outputs may be obtained from the author on reasonable request, subject to practical transfer constraints.

## Scope

The modules under `src/` are clean analysis/reference implementations; they are not the historically executed model-facing pipelines. Historical experiment source snapshots are kept separately under the repository-level `experiments/` tree. Full end-to-end reruns additionally require third-party model/corpus assets, retained raw banks/captures, and the historical execution environments.

No model weights, corpus text, logits caches, gradients, frozen result arrays, or private paths are redistributed in this source-only release. Code and documentation are MIT licensed; upstream assets retain their own licenses.
