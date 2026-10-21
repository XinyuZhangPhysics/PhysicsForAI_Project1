# Expert-Private Response in Mixture-of-Experts

Expert-private learning-response channels in mixture-of-experts networks. This 2026-10-21 public reconstruction is mapped to the manuscript and technical companion dated **2026-10-21**; the frozen numerical inputs come from the retained scientific archive and are bound by the published provenance manifests and hashes. The research question is how copy symmetry selects accessible response sectors, mixed curvature forms a local private instability, native geometry makes that response visible to an objective, and batch-conditioned maps propagate it over finite time.

## Quick reproduction

Python 3.12 is the tested interpreter. From a fresh clone:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python reproduce/reproduce_paper.py
python reproduce/verify_release.py
sha256sum -c expected/OUTPUT_SHA256SUMS.txt
```

The last command checks generated text/CSV outputs; figure/PDF pixels are not a scientific identity requirement. On macOS, `shasum -a 256 -c expected/OUTPUT_SHA256SUMS.txt` can replace `sha256sum` if coreutils is unavailable. Generated tables, figures and `verification.json` are in `results/`.

## What this reconstructs

The public frozen-output layer checks 10,752 selection coordinates, signed conversion residuals, formation discrete rates and primary/confirmation errors, native scalar and split gates, 41-arm anonymous utility NLL/bootstrap, and 360 finite-time case/sector records including the 88.756436 reference/ordered ratio. See `docs/PAPER_RESULT_MAP.md` and `docs/REPRODUCIBILITY_LEVELS.md` for claim-specific limits. `expected/OUTPUT_SHA256SUMS.txt` and `PUBLIC_CHECKSUMS.json` are the machine checks.

## Scope and limits

The six public modules are clean mathematical/reconstruction implementations. They are **not** the historically executed end-to-end model pipelines. A full model-facing rerun requires externally obtained pinned model/corpus assets, retained frozen banks, raw captures, archived execution sources and original environments. Historical source hashes are provenance references. No model weights, corpus text, logits caches or private paths are redistributed. In particular, the public finite-time summary does not replay the 885,312-dimensional raw map sequence. Exact manuscript-PDF clean-build pixel equivalence is an open manuscript-build issue, independent of the numerical checks here.

Code/documentation are MIT; project-owned derived data are CC BY 4.0. Upstream assets retain their licenses. Cite `CITATION.cff`. Data archive: **DOI pending**; the archival data DOI will be added upon deposit. No publication status or DOI is implied.
