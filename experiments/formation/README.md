# Formation

This directory contains controlled SmolLM2 expert-formation experiments used to test the local mixed router--expert theory.

- `primary/`, `confirmation/`, and `spectrum/` contain the dense duplication measurements and their deterministic analyses.
- `sparse_topk/` contains the fixed-support sparse-routing control using the same controlled duplication setting.
- `formation_model_dependency/` contains retained model-dependent helper implementations used by the formation experiments.

Experiment-local source files, fixed parameters, asset identities, and compact reference outputs are retained where practical. Machine-specific paths, model weights, raw corpus files, and runtime logs are not part of the repository.
