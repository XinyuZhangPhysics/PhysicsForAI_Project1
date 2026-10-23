# 2026-10-23 — source-only release

The public tree has been converted from the earlier frozen-output reconstruction to a source-only code release for *Expert-Specific Response in Mixture-of-Experts*.

Included:
- sanitized historical experiment/analysis source snapshots;
- clean mathematical/reference implementations;
- configuration and environment files;
- data-independent unit tests;
- reconstruction scripts that accept an external data directory;
- provenance and claim-to-code documentation.

Not included:
- project-owned experimental output arrays or frozen result tables;
- expected-output hash files tied to those arrays;
- model weights, corpus content, gradients, full-state captures, or private research records.

Project-owned numerical outputs required for optional reconstruction may be provided by the author on reasonable request. This release does not claim to provide a turnkey end-to-end rerun of third-party model-facing experiments.
