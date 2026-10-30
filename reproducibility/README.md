# Reproducibility — active manuscript release

The repository contains both the retained experiment source and the project-owned metadata needed to reconstruct the exact upstream samples used by the active manuscript.

## Upstream reconstruction

- `upstream_selection/S1/`: exact S1 calibration/audit row identities and hashes plus eight training-seed sequence manifests.
- `upstream_selection/S2/`: exact S2 calibration/audit row identities and hashes plus eight training-seed document manifests and the frozen bank index.
- `UPSTREAM_RECONSTRUCTION.md`: pinned third-party revisions and reconstruction workflow.
- `verify_upstream_manifests.py`: verifies every referenced SlimPajama row against its published SHA-256 hash.

Pair manifests are not duplicated here because they are generated deterministically from the published bank identities, published pair seeds, and the public S1/S2 pairing code.

Third-party model weights and corpus text are not redistributed. Generated token banks, representation captures, F0 arrays, mixed operators/spectra, checkpoints, and trajectory tables are derived outputs of the public pipeline rather than hidden project-owned inputs.

## Canonical evidence and provenance

`claim_to_source.tsv` maps retained manuscript analysis families. `core_results.tsv` preserves frozen numerical headlines; `statistical_design.tsv` records inferential units. `frozen_assets.tsv` and `PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv` bind canonical public files and distinguish original/public hashes.

The final scope-strengthening experiments are retained as compact frozen releases:

- `../evidence/heldout_transfer/`: compact E6/E6b prospective held-out-bank source-map, blind-replication, and target-oracle summaries.
- `../evidence/pythia_crossmodel/`: compact E7v2 aligned-rate and generic one-step cross-model summaries.
- `postpaper_protocols.tsv`: exact frozen package SHA-256 values, endpoints, decision rules, and provenance scope for E6, E6b, and E7v2.

The package identities and frozen decision endpoints are listed in `postpaper_protocols.tsv`. These additions delimit finite-sample operator transfer and support cross-model reproduction of the local mechanism without replacing the historical SmolLM2 primary analyses.

Existing clean operator tests: `python -m unittest discover -s reference_implementation/tests -v`.

Scientific qualifications documented in the evidence/provenance records remain unchanged. Withdrawn experiment families remain outside the active public scope.
