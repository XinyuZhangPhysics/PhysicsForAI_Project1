# Model-facing rerun boundary

The public clean modules implement retained mathematical operators; they are not turnkey historical model-facing pipelines. Full reruns require frozen scientific inputs, seeds/protocol seals, pinned upstream revisions, precision settings, and the applicable historical execution environment.

For the 2026-10-27 generic non-mode-aligned local-dynamics experiment, the exact executed model-facing source has now been recovered. Its seven Python files are byte-identical to the historical package; only two path strings in the launcher/configuration were sanitized for publication. The public model-facing source is under `experiments/genericlocal/`; the canonical full runner is `public_sanitized_source/scripts/run_generic_local_dynamics.py`, with exact historical SHA-256 `b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744`.

That sanitized partial tree cannot pass the original full-package verifier by itself because the sealed run depended on private F0/operator inputs and the original package manifest/layout. This is an explicit dependency boundary, not a missing-source ambiguity.

Removed fixed-routing OLMoE, trained sparse-MoE tomography/objective-response, and utility programs remain withdrawn.

Canonical compact records and original/public hashes are listed in `reproducibility/PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv` (repository-root relative). See `reproducibility/README.md` for scientific qualifications, privacy normalization and retained private dependencies.
