# Code-paper protocol audit — public v1

This document records an independent source-to-paper protocol audit of the numerical experiments in the public v1 paper.

The audit question is stricter than whether saved tables can be regenerated: for each reported experiment, does the implementation encode the model/checkpoint, coordinate definition, routing semantics, data split, perturbation, fitting rule, statistical unit, and control logic stated in the paper?

## Overall result

No systematic case was found in which the paper describes one experiment while the audited implementation executes a scientifically different experiment.

One wording ambiguity was identified in the controlled-dynamics methods and synchronized with the executed source: the logarithmic rate is fitted by ordinary least squares over the prespecified cell-specific horizon `n = 0,...,N_cell`, with the same horizon used for the two sign arms of that cell. It is not a single common horizon shared by every parameter cell.

## Experiment-level findings

| Experiment family | Audited source-level checks | Result |
|---|---|---|
| Fixed-routing expert-identity response | native 64-expert/top-8 labels and weights held fixed; 16 global label permutations; source/target split; expert-only pseudo-gradient; three finite-difference radii; 10,000 permutation bootstrap draws | **Protocol match; model-facing source public** |
| Response selection and full expert-block tangent | declared copied-expert construction; parent/child sectors; paired seeds; (g=1,2,4,8); full expert-block tangent; automatic-differentiation checks; whole-seed bootstrap | **Protocol match; model-facing source public** |
| Background-assisted mode conversion | nine backgrounds; five perturbation magnitudes with both signs; four nuisance directions per seed; odd-response fit; mixed-background fit; analytic and automatic-differentiation checks | **Protocol match; model-facing source public** |
| Controlled formation/locality | function-preserving four-expert construction; sign-paired trajectories; cell-specific frozen horizons; locality gates; rate-validity criteria and marginal-control exclusions | **Protocol match; model-facing source public** |
| Pythia cross-model replication | reported frozen output tables, gradient-evaluation count, and independent verification record | **Output match; original model-facing source package not redistributed** |
| Trained OLMoE 12D mixed curvature | declared checkpoint/layer; 64 experts/top-8 native routing; 448 documents over seven domains; disjoint source/held-out assignment; 12 radial coordinates; amplitude scan; basis/document resampling; fixed-membership routing; orientation null | **Protocol match; model-facing source public** |
| Five-checkpoint native-OLMoE response geometry | historical model-facing implementation checked for checkpoints/layers, fixed six-control family, native unnormalized selected support with full 64-way softmax router-bias mean, local/objective operators, split-half and bootstrap analyses | **Historical protocol match; current v1 exposes compact replay and provenance rather than the private execution scaffolding** |

## Public auditability boundaries

Two boundaries are deliberate.

1. **Pythia-70M cross-model replication.** Compact frozen outputs and the SHA-256 identity of the model-facing source package are public. The original frozen source package is not redistributed here.

2. **Longitudinal six-control construction and orchestration.** The frozen six-control basis, compact five-checkpoint operators, derived summaries, and clean operator replay are public. The exact historical builder/orchestration package is not redistributed in this clean v1 release.

These are release-boundary qualifications, not changes to the reported numerical results.

## Reproducibility policy

The release distinguishes model-facing source, compact-output replay, and provenance-only dependencies. Reconstructed or cleaned code is never labeled as the historical executed source unless execution identity is independently established.
