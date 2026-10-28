# Model-facing rerun boundary

The public clean modules implement retained mathematical operators; they are not turnkey historical model-facing pipelines. Full reruns require frozen scientific inputs, seeds/protocol seals, pinned upstream revisions, precision settings, and the applicable historical execution environment.

For the 2026-10-27 generic non-mode-aligned local-dynamics experiment, the exact executed model-facing source has now been recovered. Its seven Python files are byte-identical to the historical package; only two path strings in the launcher/configuration were sanitized for publication. The public model-facing source is under `experiments/genericlocal/`; concatenating the three runner chunks restores the exact recovered runner.

That sanitized partial tree cannot pass the original full-package verifier by itself because the sealed run depended on private F0/operator inputs and the original package manifest/layout. This is an explicit dependency boundary, not a missing-source ambiguity.

Removed fixed-routing OLMoE, trained sparse-MoE tomography/objective-response, and utility programs remain withdrawn.
