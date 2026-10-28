# Paper result map — active manuscript

| ID | Manuscript component | Public implementation/evidence |
|---|---|---|
| P1-E01 | copied router-side two-sided sector selection | `src/moe_private_response/selection.py` |
| P1-E02 | full expert-block tangent sector purity | `src/moe_private_response/selection.py` |
| P1-E03 | expert-parent selection branch | `src/moe_private_response/selection.py` |
| P1-E04 | background-assisted signed mode conversion | `src/moe_private_response/mode_conversion.py` |
| P1-E05 | mixed curvature, local threshold, exact-discrete growth | `src/moe_private_response/formation.py` |
| P1-E06 | stationary-background threshold/rate shift | `reproducibility/final/background_absolute_lambda_data_with_provenance.csv` |
| P1-E07 | generic non-mode-aligned frozen-map propagation | `experiments/genericlocal/GENERIC_LOCAL_PUBLIC_SYNC_READY.zip` + `reproducibility/final/GENERIC_LOCAL_HEADLINE_SUMMARY.tsv` |
| P1-E08 | leading-four mixed-block ablation | `reproducibility/final/analyze_diag_baseline.py` + summary JSON |
| P1-E09 | ordered/sequence-mean/reference-mean finite-time propagation | `src/moe_private_response/finite_time.py` |

The saved-recurrence spectrum reconstruction in `reproducibility/final/` is an independent reproducibility cross-check and is not represented as the original spectrum NPZ. Removed trained sparse-MoE experiment lines are intentionally absent.
