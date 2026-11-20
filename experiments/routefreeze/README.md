# Fixed-routing expert-identity response — public v1

This directory contains the model-facing source and an independent replication implementation for the fixed-routing OLMoE expert-identity experiment reported in the paper.

The protocol holds each token's native top-8 expert labels and routing weights fixed while testing expert-label dependence. The released implementation includes:

- 64 experts with native top-8 routing labels and weights frozen during the response measurement;
- four declared domains, with separate source-validation and target-test examples;
- 16 deterministic global expert-label permutations;
- an expert-only pseudo-gradient on the source bank;
- three finite-difference radii used to verify the local linear regime;
- a 10,000-draw bootstrap over the 16 global permutations;
- explicit route-freeze invariance and estimator checks.

The source under `primary/` is the retained model-facing implementation used for the reported experiment after path sanitization. The `independent_replication/` tree provides a separate implementation and synthetic/null diagnostics.

Machine-specific model/data locations are supplied through the repository path configuration. Third-party model weights and corpus text are not redistributed.
