# PhysicsForAI Project 1 — reproducible experiments

This repository contains source code, fixed configurations, random seeds, asset identities, and compact reference outputs needed to reproduce the numerical experiments in Project 1.

## Layout

- `experiments/` — executable experiment code and experiment-local reproducibility metadata.
- `analysis/` — standalone numerical analyses that operate on saved experiment outputs.
- `configs/` and `reference_implementation/` — shared configurations and reusable implementations.
- `evidence/` — compact legacy reference outputs retained for experiments that predate the experiment-local layout.
- `requirements-*.txt` — tested software requirements.

Controlled expert-formation experiments are grouped under `experiments/formation/`, including the dense duplication studies and the fixed-support sparse top-k control. The mature OLMoE experiments are grouped under `experiments/nmix/` as peer experiments: `source_to_heldout`, `orientation`, `basis_resampling`, `document_resampling`, and `fixed_membership`.

## Reproducibility convention

An experiment directory may contain:

- source code used for the measurement;
- model/data revision identifiers and cryptographic hashes;
- deterministic seeds and coordinate definitions;
- compact sufficient statistics or reference outputs for checking an independent rerun;
- a verifier that recomputes numerical quantities from saved artifacts.

Local machine paths, model caches, raw third-party datasets, model weights, and runtime logs are not part of the repository.

## External assets

For the OLMoE experiments, use `allenai/OLMoE-1B-7B-0924` at the checkpoint/revision specified in `experiments/nmix/common/config.json`. The 448-record bank itself is not redistributed; its SHA-256, size, domain composition, sequence rule, and deterministic source/held-out assignment are recorded in the same directory.

The controlled formation experiments use the model/data identities recorded in their experiment-local configuration and reference metadata. Third-party model and corpus bytes are not redistributed.

## License

See `LICENSE`.
