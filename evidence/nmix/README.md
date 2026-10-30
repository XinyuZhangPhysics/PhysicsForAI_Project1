# NMIX — mature native sparse-MoE mixed-block necessity

This directory is the compact public record for the manuscript's mature OLMoE held-out test (`Project1_NMIX_v1`). It is deliberately separate from the earlier pretrained-OLMoE response-geometry / utility program, which remains paused and is not positive evidence for the active manuscript.

## Scientific question

Does router–expert mixed curvature remain quantitatively necessary for local gradient-response prediction after the exact expert-duplication symmetry used by the analytic theory has been lost?

The frozen test uses `allenai/OLMoE-1B-7B-0924` at `step1200000-tokens5033B`, revision `b99743a400844e532bec7ce4575d97a3a39dcd4d`, zero-based layer 15, 64 experts, and native unrenormalized Top-8 routing. A fixed 12-dimensional radial subspace contains six expert down-projection radial coordinates and six native router-row radial coordinates. The restricted Hessian is measured on Bank A only and predicts nonlinear gradient increments on the disjoint Bank B. The ablation deletes only the router–expert mixed Hessian blocks.

## Frozen primary result

At amplitude 0.02,

- `E_full = 0.103451202`;
- `E_no_mixed = 0.586708`;
- `R = E_full/E_no_mixed = 0.176324712`;
- paired-direction 95% bootstrap CI for `R`: `[0.140634, 0.213240521]`;
- median full-map increment cosine: `0.996431295`;
- retaining mixed blocks improves all 12/12 direction pairs.

The advantage persists at amplitudes 0.01 and 0.04. All seven domainwise descriptive comparisons have `R<1` against the same Bank-A Hessian.

The frozen outcome is `GRADE_A_DELETE_NATIVE_QUALIFICATION`. Grade A+ is not reached because the support-turnover median (`0.0480375744`) and 95th percentile (`0.0517508371`) exceed the intentionally strict 1% and 5% thresholds. This leaves the fixed-support theoretical caveat in force: the experiment shows empirical predictivity through modest support turnover, not a theory of routing-boundary dynamics.

## Files

- `NMIX_PROTOCOL.json`: frozen model, bank, coordinate, objective, seed, bootstrap, and decision metadata.
- `NMIX_PRIMARY_RESULTS.tsv`: three-amplitude cross-bank headline plus the prespecified post-seal Bank-A locality control.
- `NMIX_DOMAIN_RESULTS.tsv`: seven descriptive domainwise comparisons at the primary amplitude.
- `NMIX_ADJUDICATION.tsv`: frozen Grade-A / Grade-A+ decision table.
- `verify_nmix_release.py`: deterministic consistency checks for the compact public record.
- `revision4_public/`: public-safe revision-4 production source, exact protocol/config bindings, Bank split identities, 72-state control-space evidence, Hessian/direction/contrast exports, and deterministic replay checker.
- `RELEASE_COMPLETENESS.md`: final public-release completeness record.

## Data boundary

The archived 448-record token bank is a derivative of a third-party dataset and is not redistributed. Its exact SHA-256, size, deterministic A/B split rule, upstream model revision, sequence rule, and seeds are public in `NMIX_PROTOCOL.json`. Third-party OLMoE weights are likewise not redistributed.

The active manuscript's claim is local: one mature checkpoint, layer 15, one frozen 12-dimensional radial subspace, one disjoint 224/224 bank split, and amplitudes 0.01–0.04. It does not establish optimizer-state-aware dynamics, long-time specialization, end-to-end utility, universal Hessian transport, or a theory across Top-k support jumps.
