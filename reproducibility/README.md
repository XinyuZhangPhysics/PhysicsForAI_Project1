# Public reproducibility layer

This directory supplements the historical source snapshots in `experiments/` with reader-facing provenance and frozen numerical summaries for *Expert-Specific Response in Mixture-of-Experts*.

The files here are deliberately limited to project-owned derived information that can be redistributed without copying third-party model weights or corpus content:

- `core_results.tsv`: frozen headline numerical summaries used in the manuscript.
- `claim_to_source.tsv`: mapping from manuscript analysis families to public experiment/source directories and the relevant inferential unit.
- `frozen_assets.tsv`: pinned upstream revisions and principal protocol/basis hashes.
- `statistical_design.tsv`: sampling and uncertainty semantics for the main analysis families.

These files are not substitutes for the original model/data providers. Full model-facing reruns still require the pinned upstream checkpoints and datasets, and some large project-owned intermediate captures are not included in this repository. The repository does not redistribute copyrighted corpus text, model weights, token banks, gradients, or full-state captures.

The historical scripts in `experiments/` are retained for provenance. `reference_implementation/` provides a clean mathematical implementation and tests. Where a historical execution source was superseded by a certified repair, the public claim-to-source map points to the retained source group used by the final evidence chain rather than to obsolete exploratory code.
