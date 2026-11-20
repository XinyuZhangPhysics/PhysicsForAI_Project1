# Public v1 compact paper evidence

This directory collects small derived arrays used by the public v1 paper figures and the primary trained-checkpoint source-to-heldout calculation.

## Contents

- `H_src.tsv`: the symmetric 12x12 source Hessian, copied byte-for-byte from `experiments/nmix/source_to_heldout/reference/H_A_SYM.tsv`.
- `heldout_responses_a002.tsv`: the 24 signed held-out states and baseline-subtracted gradient responses at central amplitude a=0.02, selected from the public `B_CONTROL_SPACE_RAW.tsv`.
- `directions.tsv`: the 12 predeclared unit-direction coordinate rows reconstructed from the positive a=0.02 states.
- `controlled_rate_threshold.csv`: predicted and observed controlled growth rates used in the controlled-formation figure, recovered from the frozen public formation release.
- `rankk_heldout_error.csv`: rank-k held-out errors reported in the paper.
- `mode_conversion_six_cells.csv`: frozen allowed-channel/mixed-opening calibration, including the canonical SmolLM2/SwiGLU g=4 value.
- `verify_primary_olmoe.py`: reconstructs the source-Hessian full/block-diagonal errors, same-amplitude secant residual, mixed-block rank-k curve, participation dimension, and whitened coupling from the compact arrays.

Coordinate order is expert radial coordinates x_0,...,x_5 followed by router radial coordinates u_0,...,u_5, matching the paper. The held-out responses use native routing and the declared evaluation loss.

## Provenance

Primary source-Hessian/held-out-response artifacts derive from public commit `ac976549e854196450dbeee6f62dda880c744d86`. Controlled-rate and mode-conversion derived data derive from the earlier immutable public release commit beginning `4c1000900e` and are copied here to make the paper figure/effect-size data directly discoverable.

## Known boundary

The complete independently remeasured 12x12 held-out Hessian H_ho is not present in the public repository or in the retained public-release assets. It is **not** reconstructed from same-amplitude responses, because such a secant fit is a different object. Adding the original measured H_ho would require recovery from the executed archive and validation against the reported same-set error and spectral-overlap diagnostics.
