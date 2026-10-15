# Model-facing rerun boundary

No new scientific experiment was requested or run. The public modules offer
the core operator operations, not a historically verified end-to-end model runner.
Retained execution-bound sources are indexed by hash in execution_source_hashes.json.
They are private source snapshots because they contain machine-specific paths,
bank dependencies and uncertain source redistribution terms. Full reruns require
the original frozen banks, seeds, protocol seals, model revisions, FP32/BF16
settings and dependency receipts. Model weights stay on the Mac and are excluded
from GitHub and Zenodo. `configs/paper_models.json` supplies verified revisions
where recovered, and explicitly identifies revisions not resolved here.

Native gain multiplies cached selected expert outputs by 1+g. Bias acts on all
64 logits before Top-k; the clean-soft tangent holds selected support fixed,
uses full-softmax mean bias and does not renormalize Top-8 weights. Local Gram
divides by 256*2048; objective Gram squares the suffix-adjoint contraction without
another division. Six controls and the generalized metric are declared spaces.
Q3 uses two scalar amplitudes at zero-based layers 7/15 and adds FP32 64-vectors
to mlp.gate outputs before Top-k. The historical Q3 model is BF16 on MPS.
Formation captures mixed curvature at the copied origin and uses frozen leading
singular vectors for initialization. Finite-time C_n maps are batch-conditioned
at the same frozen dense copied origin, without trajectory recentering.
