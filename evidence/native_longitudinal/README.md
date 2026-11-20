# Native OLMoE longitudinal response geometry — public v1

This directory provides the compact public replay for the five-checkpoint native-OLMoE longitudinal response measurement.

## Public objects

- `control_basis.npz` — frozen six-control family used at all checkpoints;
- `checkpoint_41_tomography.npz` — baseline checkpoint response operators and sufficient statistics;
- `checkpoint_5033_tomography.npz` — terminal checkpoint response operators and sufficient statistics;
- `intermediate_checkpoints_tomography.npz` — corresponding arrays for the intermediate checkpoints;
- `tomography_summary.csv`, `bootstrap_quantiles.csv`, `endpoint_eigenvalues.csv`, and `layer15_checkpoint5033_secondary_geometry.csv` — compact derived summaries;
- `expected_gates.json` and `response_scalars.json` — frozen scalar checks used by the compact replay;
- `verify_longitudinal.py` — independent compact verification of the released operators and summary quantities.

The clean operator implementation is `reference_implementation/src/moe_private_response/native_tomography.py`.

## Routing convention

The released tangent implementation uses the native selected support without within-support renormalization. Perturbed full-softmax probabilities are gathered on the selected experts, and the router-bias mean contains all 64 softmax probabilities. This is the convention stated in the paper.

## Provenance boundary

The frozen six-control basis and the downstream compact response operators are public. The exact historical execution scaffolding that generated the final six-control basis and orchestrated the full model-facing five-checkpoint run is not redistributed in this clean public v1 release. Those historical sources were audited against the paper protocol before curation, but they contain private execution scaffolding and authoring-stage identifiers.

Accordingly, this directory is labeled **compact replay / provenance boundary**, not as a substitute for the historical model-facing execution source. No reconstructed code is represented as the original executed implementation.
