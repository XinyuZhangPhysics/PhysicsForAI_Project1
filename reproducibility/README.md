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
- `../evidence/nmix/`: mature OLMoE Bank-A/Bank-B mixed-block necessity protocol, compact results, adjudication, and candidate-release completeness record.
- `postpaper_protocols.tsv`: frozen package SHA-256 values, endpoints, decision rules, and provenance scope for E6, E6b, and E7v2.

## NMIX candidate-release status

The NMIX scientific result is final (`GRADE_A_DELETE_NATIVE_QUALIFICATION`), but this synchronization branch is not yet the final public reproducibility release. The compact protocol and aggregate outcomes are verified. The sealed revision-4 result bundle additionally contains the exact executed production source, frozen contrast/direction arrays, all 72 signed Bank-B states, raw/symmetrized Hessians, support diagnostics, and all 10,000 bootstrap replicates. Those bytes are not currently exportable through the available Project Library surface in this synchronization session.

Accordingly, `../evidence/nmix/RELEASE_COMPLETENESS.md` is binding: `main` should not be advanced to this candidate until exact production source/config plus compact per-direction sufficient statistics (or equivalent raw gradient/Hessian arrays) are publicly bound and independently checked. This makes the Data Availability verification explicit rather than inferring execution identity from a reconstructed implementation.

The historical pretrained-OLMoE response-geometry / utility program remains paused and outside the active claim set. It must not be conflated with NMIX.

Existing clean operator tests: `python -m unittest discover -s reference_implementation/tests -v`. NMIX compact-record consistency: `python evidence/nmix/verify_nmix_release.py`.
