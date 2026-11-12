# Sparse top-k formation control

This experiment tests the mixed router--expert local-instability mechanism in the controlled SmolLM2-135M duplication construction using an explicitly sparse, renormalized top-2 router on a fixed-support branch.

At the symmetric reference state, experts 0 and 1 form the selected pair and experts 2 and 3 are separated by a fixed logit margin. With the active contrast

[
c=(1,-1,0,0)/sqrt{2},
]

the expert-relative routing coefficient is

[
eta(t)=rac{1}{sqrt{2}}	anh!left(rac{t}{sqrt{2}}ight),
qquad eta'(0)=rac12.
]

The mixed operator is measured directly on the fixed historical `F0` bank. The nonlinear control uses one stable and one unstable cell at the same dimensionless update scale, with sign-paired initial conditions. The selected expert set remains unchanged throughout the reported trajectories.

`source/` contains the fixed configuration, deterministic runner, and verifier. `reference/` contains compact sufficient statistics for the mixed-operator measurement and nonlinear trajectories. The full saved spectrum array is identified by SHA-256 in `reference/preflight.json`; third-party model weights and the historical `F0` bytes are not redistributed.

This control concerns the smooth fixed-support sparse branch. It does not model routing-support boundary dynamics or the native no-renormalization routing convention of OLMoE.
