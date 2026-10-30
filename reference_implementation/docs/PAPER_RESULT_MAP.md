# Paper result map — active manuscript

| ID | Manuscript component | Public implementation/evidence |
|---|---|---|
| P1-E01 | copied router-side two-sided sector selection | `reference_implementation/src/moe_private_response/selection.py` |
| P1-E02 | full expert-block tangent sector purity | `reference_implementation/src/moe_private_response/selection.py` |
| P1-E03 | expert-parent selection branch | `reference_implementation/src/moe_private_response/selection.py` |
| P1-E04 | background-assisted signed mode conversion | `reference_implementation/src/moe_private_response/mode_conversion.py` |
| P1-E05 | mixed curvature, local threshold, exact-discrete growth | `reference_implementation/src/moe_private_response/formation.py` |
| P1-E06 | stationary-background threshold/rate shift | `analysis/astra_r2_node6/background_absolute_lambda_data_with_provenance.csv` |
| P1-E07 | generic non-mode-aligned frozen-map propagation | `experiments/genericlocal/` + `reproducibility/final/GENERIC_LOCAL_HEADLINE_SUMMARY.tsv` |
| P1-E08 | leading-four mixed-block ablation | `analysis/astra_r2_node1/analyze_diag_baseline.py` + `analysis/astra_r2_node1/generic_local_dynamics_diag_baseline.csv` + summary JSON |
| P1-E09 | ordered/sequence-mean/reference-mean finite-time propagation | `reference_implementation/src/moe_private_response/finite_time.py` |
| P1-E10 | mature native OLMoE mixed-block necessity on disjoint Bank B | `evidence/nmix/` — source, protocol, compact results/adjudication, measured per-state/control-space evidence, and deterministic replay |

The saved-recurrence spectrum reconstruction in `analysis/astra_r2_node3/` is an independent reproducibility cross-check and is not represented as the original spectrum NPZ. The historical pretrained-OLMoE response-geometry / utility program remains paused and is not mapped as positive evidence.

P2J/P2L identity and E4B component records: `analysis/astra_r2_node6/archive_snapshots/` and `analysis/astra_r2_node6/ARCHIVE_SOURCE_BINDINGS.md`. Generic-local manifest/summary/endpoints/top-four records: `evidence/generic_local_dynamics/`. NMIX compact records: `evidence/nmix/`. All canonical authority bytes and transformations are otherwise bound in `reproducibility/PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv`. Scientific qualifications and nonredistributable dependencies: `reproducibility/README.md`.
