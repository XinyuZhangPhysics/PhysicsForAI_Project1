# PhysicsForAI Project 1 — public v1 reproducibility release

This repository contains the source code, fixed configurations, random seeds, asset identities, and compact reference outputs associated with the public v1 paper.

## Layout

- `experiments/` — executable experiment code and experiment-local reproducibility metadata.
- `analysis/` — standalone numerical analyses operating on saved experiment outputs.
- `configs/` and `reference_implementation/` — shared configurations, compact replay code, and reusable implementations.
- `evidence/` — compact paper-facing evidence and retained reference outputs.
- `requirements-*.txt` — tested software requirements.

The paper-facing experiment-to-source index is `reference_implementation/docs/PAPER_RESULT_MAP.md`. The independent source-to-paper audit is `CODE_PAPER_PROTOCOL_AUDIT.md`.

Controlled expert-formation experiments are grouped under `experiments/formation/`. The trained OLMoE mixed-curvature experiments are grouped under `experiments/nmix/` as peer experiments: `source_to_heldout`, `orientation`, `basis_resampling`, `document_resampling`, and `fixed_membership`. Native OLMoE longitudinal response is released as a compact replay under `evidence/native_longitudinal/`, with the clean operator implementation in `reference_implementation/src/moe_private_response/native_tomography.py`.

## Reproducibility convention

An experiment directory may contain:

- source code used for the measurement;
- model/data revision identifiers and cryptographic hashes;
- deterministic seeds and coordinate definitions;
- compact sufficient statistics or reference outputs for checking an independent rerun;
- a verifier that recomputes numerical quantities from saved artifacts.

Local machine paths, model caches, raw third-party datasets, model weights, and runtime logs are not part of the repository.

The result map distinguishes three cases explicitly: model-facing source mapped in the repository, compact-output replay from frozen sufficient statistics, and provenance-only boundaries where an original execution dependency is not redistributed.

## External assets

For the trained OLMoE mixed-curvature experiments, use `allenai/OLMoE-1B-7B-0924` at the checkpoint/revision specified in `experiments/nmix/common/config.json`. The 448-record bank itself is not redistributed; its SHA-256, size, domain composition, sequence rule, and deterministic source/held-out assignment are recorded in the same directory.

Controlled formation and cross-model experiments use the model/data identities recorded in their experiment-local metadata. Third-party model and corpus bytes are not redistributed.

## License

See `LICENSE`.