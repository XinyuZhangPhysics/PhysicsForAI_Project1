# Paper result map — public v1

This is the paper-facing source/evidence index for the public v1 release. It maps the experiments discussed in the paper to the released implementation and verification artifacts without exposing authoring-workflow revision identifiers.

| ID | v1 experiment family | Primary public source/evidence | Public verification | Status |
|---|---|---|---|---|
| P1-E01 | Fixed-routing expert-identity response | `experiments/routefreeze/primary/source/` | `experiments/routefreeze/primary/tests/test_u4_protocol.py` and independent replication tree | **MODEL-FACING SOURCE MAPPED** |
| P1-E02 | Response selection and full expert-block tangent | `experiments/copiedselection/`; `experiments/fulltangent/full_tangent/source/` | experiment-local validation and `reference_implementation/src/moe_private_response/selection.py` | **MODEL-FACING SOURCE MAPPED** |
| P1-E03 | Background-assisted mode conversion | `experiments/modeconversion/conversion/source/` | `verify_results.py` and `reference_implementation/src/moe_private_response/mode_conversion.py` | **MODEL-FACING SOURCE MAPPED** |
| P1-E04 | Controlled formation and locality | `experiments/formation/` | experiment-local seals/checks and `reference_implementation/src/moe_private_response/formation.py` | **MODEL-FACING SOURCE MAPPED** |
| P1-E04R | Pythia-70M cross-model replication | `evidence/pythia_crossmodel/` | compact frozen outputs and source-package SHA-256 | **COMPACT OUTPUTS; MODEL-FACING SOURCE PACKAGE NOT REDISTRIBUTED** |
| P1-E05 | Trained OLMoE 12-dimensional mixed curvature and source-to-heldout transfer | `experiments/nmix/` | `experiments/nmix/verify_reference_outputs.py` plus resampling/fixed-membership/orientation controls | **MODEL-FACING SOURCE MAPPED** |
| P1-E06 | Five-checkpoint native-OLMoE longitudinal response geometry | `evidence/native_longitudinal/`; `reference_implementation/src/moe_private_response/native_tomography.py` | `evidence/native_longitudinal/verify_longitudinal.py` | **COMPACT REPLAY / PROVENANCE BOUNDARY** |

## Interpretation of status labels

**MODEL-FACING SOURCE MAPPED** means the public release contains source sufficient to audit the stated model-facing measurement protocol, subject to external model/corpus dependencies.

**COMPACT OUTPUTS** means frozen outputs and cryptographic source identity are public, but the original model-facing execution package itself is not redistributed.

**COMPACT REPLAY / PROVENANCE BOUNDARY** means the frozen sufficient objects and a clean independent replay are public, while an exact historical execution dependency is deliberately not represented as if it were public source.

No unavailable execution source is replaced by reconstructed code and labeled as the historical implementation.
