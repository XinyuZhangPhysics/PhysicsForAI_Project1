# Reproducibility — active manuscript release

The repository contains retained experiment source and project-owned metadata for the active manuscript *Local Symmetry Breaking and Router–Expert Coupling in Mixture-of-Experts*.

## Upstream reconstruction

- `upstream_selection/S1/`: exact S1 calibration/audit row identities and hashes plus eight training-seed sequence manifests.
- `upstream_selection/S2/`: exact S2 calibration/audit row identities and hashes plus eight training-seed document manifests and the frozen bank index.
- `UPSTREAM_RECONSTRUCTION.md`: pinned third-party revisions and reconstruction workflow.
- `verify_upstream_manifests.py`: verifies every referenced SlimPajama row against its published SHA-256 hash.

Third-party model weights and corpus text are not redistributed. Generated token banks, representation captures, F0 arrays, mixed operators/spectra, checkpoints, and trajectory tables are derived outputs of the public pipeline rather than hidden project-owned inputs.

## Canonical evidence and provenance

`claim_to_source.tsv` maps retained manuscript analysis families. `core_results.tsv` preserves frozen numerical headlines; `statistical_design.tsv` records inferential units. `frozen_assets.tsv` and `PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv` bind canonical public files and distinguish original/public hashes.

The final scope-strengthening experiments are retained as compact frozen releases:

- `../evidence/heldout_transfer/`: E6/E6b prospective held-out-bank source-map, blind-replication, and target-oracle summaries. These are negative scope results for fixed small-bank numerical operator transfer and remain active manuscript boundaries.
- `../evidence/pythia_crossmodel/`: E7v2 aligned-rate and generic one-step cross-model summaries after remeasuring the Pythia operator.
- `../evidence/nmix/`: mature OLMoE Bank-A/Bank-B mixed-block necessity source, protocol, compact measured evidence, results, adjudication, and deterministic replay.
- `postpaper_protocols.tsv`: frozen package SHA-256 values, endpoints, decision rules, and provenance scope for E6, E6b, and E7v2.

## NMIX reproducibility

The NMIX materials under `../evidence/nmix/` bind the executed production source and frozen protocol/configuration to compact measured control-space evidence: deterministic Bank-A/Bank-B identities and hashes, frozen directions/contrast basis, raw/symmetrized Bank-A Hessians, all 72 signed Bank-B state metrics, and exact control-space states/responses sufficient to reconstruct the reported primary errors and paired-direction bootstrap. The deterministic checker independently reconstructs the published interval and frozen adjudication.

The raw 448-record token bank, third-party model weights/tokenizer, and large acceleration/routing-logit caches are not redistributed. Their identities and acquisition/reconstruction requirements are recorded, while the exact private scientific archive has separately passed remote hash readback and archive-only restoration.

The historical pretrained-OLMoE response-geometry / utility program remains paused and outside the active claim set. It must not be conflated with NMIX.

Existing clean operator tests: `python -m unittest discover -s reference_implementation/tests -v`. NMIX compact-record consistency: `python evidence/nmix/verify_nmix_release.py`.
