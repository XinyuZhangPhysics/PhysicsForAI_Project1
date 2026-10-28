# Reproducibility — active manuscript scope

1. Run data-independent operator tests with `python -m unittest discover -s tests -v`.
2. Use the repository-level `experiments/` source groups and `PUBLIC_SOURCE_BINDINGS.tsv` for retained historical execution-source provenance.
3. Use `reproducibility/` for retained numerical summaries, frozen asset hashes, and statistical-unit semantics.
4. Full model-facing reruns require pinned third-party models/corpora and retained scientific inputs.

The generic-local exact executed source is the one unresolved public-provenance item and is not reconstructed here.
