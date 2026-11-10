# Mixed-block orientation experiment

This experiment tests the measured router–expert mixed-block orientation against spectrum-matched rotations while holding the diagonal Hessian blocks, mixed singular spectrum, empirical states, and responses fixed.

The reference ensemble uses 50,000 draws with NumPy PCG64 seed 2026110901. The reported rank is the conditional geometric rank
`(1 + # {E_null <= E_true}) / 50001`; it is not a population p-value.

Only the central-amplitude source-Hessian-on-heldout-bank experiment is included here. The script also supports one-sided router/expert Haar and signed-permutation diagnostic families at the same amplitude.
