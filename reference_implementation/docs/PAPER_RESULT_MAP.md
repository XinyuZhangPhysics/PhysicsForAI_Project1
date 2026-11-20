# Paper result map — public v1

This is the paper-facing source/evidence index for the public v1 release. It maps the experiments discussed in the paper to the public source and compact verification artifacts without using authoring-workflow revision identifiers.

| ID | v1 experiment family | Primary public source | Public verification/evidence | Audit status |
|---|---|---|---|---|
| P1-E01 | Fixed-routing expert-identity response | `experiments/routefreeze/primary/source/u4_execute.py`, `u4_finalize_metrics.py` | `experiments/routefreeze/primary/tests/test_u4_protocol.py` and retained experiment outputs/provenance | **MODEL-FACING SOURCE MAPPED** |
| P1-E02 | Response selection and full expert-block tangent | `experiments/copiedselection/S1/source/response_s1.py`, `experiments/copiedselection/S2_source/science_response.py`, `experiments/fulltangent/full_tangent/source/stage45.py` | experiment-local verification/provenance files and `reference_implementation/src/moe_private_response/selection.py` | **MODEL-FACING SOURCE MAPPED** |
| P1-E03 | Background-assisted mode conversion | `experiments/modeconversion/conversion/source/experiment.py`, `finalize.py` | `verify_results.py` and `reference_implementation/src/moe_private_response/mode_conversion.py` | **MODEL-FACING SOURCE MAPPED** |
| P1-E04 | Controlled formation, locality, and cross-model replication | `experiments/formation/primary/preflight_resume.py`, `train_primary.py`, `analyze_primary.py`; sparse-top-k controls under `experiments/formation/sparse_topk/` | compact Pythia replication outputs under `evidence/pythia_crossmodel/`; `reference_implementation/src/moe_private_response/formation.py` | **FORMATION SOURCE MAPPED**; Pythia source-package hash is public, but the frozen Pythia source package itself is not redistributed here |
| P1-E05 | Trained OLMoE 12-dimensional mixed curvature and source-to-heldout transfer | `experiments/nmix/source_to_heldout/source/` plus `experiments/nmix/common/config.json` | `experiments/nmix/verify_reference_outputs.py`, basis/document resampling, fixed-membership, and orientation-null subexperiments | **MODEL-FACING SOURCE MAPPED** |
| P1-E06 | Five-checkpoint longitudinal native-OLMoE response geometry | `experiments/nativeolmoe/tomography/tomography_engine.py`, `experiments/nativeolmoe/five_checkpoint_response/work/production_resume_v2.py`, `final_t4.py` | frozen six-control basis `reference_implementation/data/derived/native_control_basis.npz` and compact tomography matrices/summaries | **MEASUREMENT SOURCE MAPPED**; the exact historical builder that produced the final frozen six-control basis is not redistributed |

## Scope of the public audit

The statuses above refer to whether the paper's stated protocol can be traced to model-facing public source, not merely whether final numerical tables can be regenerated from already-derived arrays.

For P1-E04, the Pythia compact evidence is sufficient to verify the reported frozen outputs but not to independently audit the original model-facing implementation from this repository alone. For P1-E06, the frozen six-control basis and the downstream five-checkpoint measurement code are public; the historical basis-construction execution is represented by provenance rather than by a complete public builder.

No unavailable execution source is replaced by reconstructed code and labeled as the historical implementation.
