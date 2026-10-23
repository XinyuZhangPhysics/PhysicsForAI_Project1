# Paper result map — source-only release

This document maps manuscript claims to the public analysis code. Numerical result files are not bundled in the repository; paper-level reconstruction requires the author-supplied project-owned derived-data bundle via `PROJECT1_DATA_DIR`.

| ID | Manuscript component | Clean implementation | External numerical input required for paper-level reconstruction |
|---|---|---|---|
| P1-E01 | native route-freeze response | `src/moe_private_response/native_tomography.py` | yes |
| P1-E02 | copied router-side two-sided selection | `src/moe_private_response/selection.py` | yes |
| P1-E03 | full expert-block tangent sector purity | `src/moe_private_response/selection.py` | yes |
| P1-E04 | expert-parent selection branch | `src/moe_private_response/selection.py` | yes |
| P1-E05 | background-assisted signed mode conversion | `src/moe_private_response/mode_conversion.py` | yes |
| P1-E06 | measured mixed curvature and local threshold | `src/moe_private_response/formation.py` | yes |
| P1-E07 | exact-discrete local growth and mobility crossover | `src/moe_private_response/formation.py` | yes |
| P1-E08 | high-mobility confirmation | `src/moe_private_response/formation.py` | yes |
| P1-E09 | native six-control response geometry | `src/moe_private_response/native_tomography.py` | yes |
| P1-E10 | persistent leading response eigendirection | `src/moe_private_response/native_tomography.py` | yes |
| P1-E11 | local/objective response and signed-control tests | `src/moe_private_response/native_tomography.py` | yes |
| P1-E12 | router-intervention utility comparison | `src/moe_private_response/q3.py` | yes |
| P1-E13 | ordered, sequence-mean, and reference-mean propagation | `src/moe_private_response/finite_time.py` | yes |
| P1-E14 | matched finite-time sampling control | `src/moe_private_response/finite_time.py` | yes |
| P1-E15 | cumulant/log hierarchy and replication streams | `src/moe_private_response/finite_time.py` | yes |

The data-independent identities underlying these analyses are exercised by `tests/test_paper.py`. Full model-facing reruns require the additional private and third-party inputs described in `MODEL_FACING_RERUNS.md`.
