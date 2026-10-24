# Public reproducibility layer

This directory supplements the historical source snapshots in \`experiments/\` with reader-facing provenance and frozen numerical summaries for *Expert-Specific Response in Mixture-of-Experts*.

The files here are deliberately limited to project-owned derived information that can be redistributed without copying third-party model weights or corpus content:

- \`core_results.tsv\`: frozen headline numerical summaries used in the integrated manuscript.
- \`claim_to_source.tsv\`: mapping from manuscript analysis families to public historical source groups and the relevant inferential unit.
- \`frozen_assets.tsv\`: pinned upstream revisions, principal protocol/basis hashes, and post-paper package/result hashes.
- \`statistical_design.tsv\`: sampling and uncertainty semantics for the main analysis families.
- \`postpaper_E1_E4.tsv\`: final E1–E4 labels and headline quantities after independent adjudication.
- \`postpaper_protocols.tsv\`: frozen endpoints, decision rules, package hashes, and scope notes for E1–E4.

The E1–E4 program was executed after the earlier manuscript baseline was frozen. It was designed to test specific boundary questions without retrospectively redefining the earlier endpoints. E3 requires an additional distinction: the decision to use the sealed reserve set was made after the original held-out result, but the reserve-only protocol was frozen before any reserve model forward or reserve outcome was generated. It is therefore an independent-data prospective replication relative to reserve outcomes, not an original preregistration.

These files are not substitutes for the original model/data providers. Full model-facing reruns still require the pinned upstream checkpoints and datasets, and some large project-owned intermediate captures are not included in this repository. The repository does not redistribute copyrighted corpus text, model weights, token banks, gradients, or full-state captures.

The historical scripts in \`experiments/\` are retained for provenance. \`reference_implementation/\` provides a clean mathematical implementation and tests. The post-paper registry points back to the corresponding historical source family and binds the additional frozen execution package by SHA-256; exact larger output bundles are retained in the scientific archive rather than duplicated in GitHub.
