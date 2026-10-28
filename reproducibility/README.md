# Public reproducibility layer

This directory contains project-owned provenance and compact numerical evidence for the active manuscript *Expert-Specific Response in Mixture-of-Experts*.

Core registries:
- `core_results.tsv`: retained headline numerical summaries.
- `claim_to_source.tsv`: active manuscript claims mapped to public source/evidence.
- `frozen_assets.tsv`: pinned revisions, protocol IDs, and SHA-256 identities.
- `statistical_design.tsv`: inferential-unit and resampling semantics.
- `postpaper_E1_E4.tsv` and `postpaper_protocols.tsv`: retained E1 and E4a/E4b records.

Final synchronization additions are under `final/`:
- generic-local headline and leading-four diagnostics;
- projected diagonal-only ablation source/summary;
- saved-recurrence singular-spectrum reconstruction source/results;
- P2J/P2L growth-figure provenance;
- stationary-background common-absolute-`lambda` figure data;
- SHA bindings for larger frozen records not redistributed in GitHub.

The exact generic-local model-facing runner, local scientific dependency, and public protocol are published under `experiments/genericlocal/`. The runner is split into three exact text chunks; their concatenation is hash-bound to the recovered historical source.

Removed fixed-routing OLMoE, trained sparse-MoE tomography/objective-response, and utility experiments remain outside the active public evidence layer.
