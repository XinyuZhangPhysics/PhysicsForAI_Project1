# Code-paper protocol audit — public v1

This document records an independent source-to-paper protocol audit of the numerical experiments in the public v1 paper.

The audit question is stricter than whether saved tables can be regenerated: for each reported experiment, does the public model-facing implementation actually encode the model/checkpoint, coordinate definition, routing semantics, data split, perturbation, fitting rule, statistical unit, and control logic stated in the paper?

## Overall result

No systematic case was found in which the paper describes one experiment while the released implementation executes a scientifically different experiment.

One wording ambiguity was identified in the controlled-dynamics methods and synchronized with the executed source: the logarithmic rate is fitted by ordinary least squares over the prespecified cell-specific horizon
`n = 0,...,N_cell`, with the same horizon used for the two sign arms of that cell. It is not a single common horizon shared by every parameter cell.

## Experiment-level audit

| Experiment family | Source-level checks | Result |
|---|---|---|
| Fixed-routing expert-identity response | native 64-expert/top-8 labels and weights held fixed; 16 global label permutations; source/target split; expert-only pseudo-gradient; three finite-difference radii; 10,000 permutation bootstrap resamples | **Protocol match** |
| Response selection and full expert-block tangent | declared copied-expert construction; parent/child sectors; paired seeds; (g=1,2,4,8); full expert-block tangent; independent automatic-differentiation checks; whole-seed bootstrap | **Protocol match** |
| Background-assisted mode conversion | nine background values; five perturbation magnitudes with both signs; four nuisance directions per seed; odd finite-difference fit; mixed background fit; analytic chain-rule and automatic-differentiation checks | **Protocol match** |
| Controlled formation and locality | function-preserving four-expert construction; sign-paired trajectories; cell-specific frozen horizons; full-step locality gates; rate-validity criteria and marginal-control exclusions | **Protocol match** |
| Trained OLMoE 12-dimensional mixed curvature | declared checkpoint and layer; 64 experts/top-8 native routing; 448 documents over seven domains; disjoint source/held-out assignment; 12 radial coordinates; amplitude scan; basis/document resampling; fixed-membership routing; orientation null | **Protocol match** |
| Five-checkpoint native-OLMoE response geometry | declared checkpoints/layers; fixed six-control family; native unnormalized selected support with the full 64-way softmax entering the router-bias mean; local/objective operators; split-half and bootstrap analyses | **Measurement protocol match** |

## Public auditability boundaries

Two boundaries are deliberately stated rather than hidden.

1. **Pythia-70M cross-model replication.** Compact frozen outputs and the SHA-256 identity of the model-facing source package are public. The original frozen source package itself is not redistributed in this repository. The reported outputs are therefore checkable from the public compact evidence, but the original model-facing execution cannot be independently reconstructed from this repository alone.

2. **Longitudinal six-control construction.** The frozen six-control basis used by the five-checkpoint measurement and the downstream measurement source are public. The exact historical builder that produced the final frozen six-control basis is not redistributed. The repository therefore treats the builder as a provenance boundary rather than replacing it with retrospectively reconstructed code.

These qualifications concern public auditability, not changes to the reported numerical results.

## Reproducibility policy

Public source is labeled according to what it supports:

- **model-facing source mapped** — the implementation needed to audit the stated measurement protocol is present;
- **compact-output replay** — frozen sufficient statistics permit independent recomputation of reported derived quantities but do not substitute for the original model-facing execution;
- **provenance boundary** — the release records identities/hashes and states explicitly that an original execution dependency is not redistributed.

Reconstructed or cleaned code is never labeled as the historical executed source unless its execution identity is independently established.
