# PhysicsForAI Project 1 — source and reproducibility release

Code and project-owned reproducibility material accompanying *Local Symmetry Breaking and Router–Expert Coupling in Mixture-of-Experts*.

> **NMIX synchronization status.** The revision-4 public core is complete under `evidence/nmix/revision4_public/`: executed production source, frozen protocol/configuration, exact Bank-A/Bank-B identities and hashes, 72 signed Bank-B state metrics, raw/symmetrized Bank-A Hessians, frozen directions/contrast basis, and deterministic replay inputs for the reported bootstrap/adjudication. See `evidence/nmix/RELEASE_COMPLETENESS.md`.

The public tree otherwise tracks the active manuscript: copied-state sector selection, full-tangent and expert-side checks, background conversion, duplication-state local dynamics, amplitude-domain locality, finite-time analysis, the prospective small-bank held-out scope boundary, Pythia cross-model replication, and the mature native sparse-MoE mixed-block necessity test. Experiment families removed from the manuscript remain outside the active scope.

## Mature native sparse-MoE mixed-block necessity (NMIX)

`evidence/nmix/` records the frozen `Project1_NMIX_v1` test at the mature OLMoE 5033B-token checkpoint. A Bank-A restricted Hessian in a fixed 12-dimensional native radial subspace predicts nonlinear gradient increments on a disjoint 224-document Bank B. At the primary amplitude, `E_full=0.103451202`, `E_no_mixed=0.586708`, and `R=0.176324712` with paired-direction 95% bootstrap CI `[0.140634,0.213240521]`; the median increment cosine is `0.996431295`, and all 12/12 direction pairs improve when the router–expert mixed blocks are retained.

The frozen outcome is Grade A, not Grade A+: the empirical advantage survives several-percent support turnover, but the hard Top-k fixed-support theoretical caveat remains. This experiment is distinct from the historical pretrained-OLMoE response-geometry / utility program, which remains paused and is not positive evidence for the active manuscript.

## Reproducing the model-facing experiments

For the established S1/S2/Pythia experiment families, all project-owned information required to reconstruct the reported experimental inputs is public. Exact corpus row identities, hashes, pair-selection seeds, training-seed manifests, protocol metadata, and pinned upstream revisions are under `reproducibility/upstream_selection/`; the reconstruction workflow is documented in `reproducibility/UPSTREAM_RECONSTRUCTION.md`.

Third-party weights and corpus text are not redistributed. Readers obtain the pinned Pythia-70M, SmolLM2-135M, and SlimPajama-6B revisions recorded in `configs/models.json`. Generated token banks, representation captures, F0 arrays, mixed operators/spectra, checkpoints, and trajectory tables are reproducible intermediate outputs rather than additional private dependencies.

For NMIX, the public release supplies the pinned OLMoE revision, exact archived-bank hash and split identities, S96 rule, coordinate dimensions/seeds, objective/bootstrapping metadata, executed revision-4 source, compact raw control-space responses/states, Hessians, aggregate results, and adjudication. `evidence/nmix/revision4_public/verify_revision4_core.py` reconstructs the reported primary errors, paired-direction bootstrap interval, support-turnover summary, and frozen decision from the released records.

## Generic non-mode-aligned local dynamics

The recovered historical source and public-safe provenance handoff are canonical under `experiments/genericlocal/`. The full runner is `public_sanitized_source/scripts/run_generic_local_dynamics.py` with historical SHA-256 `b5be93f73c40bc4c10ac375813c2e91732dea47c4158d06e18e0183357d25744`. See `experiments/genericlocal/README_PUBLIC_SYNC.md`.

The E5 amplitude-domain extension is under `experiments/genericlocal/amplitude_domain/`, with its frozen preanalysis, protocol, one-step runner, and analyzer. Its all-cell primary table and compact aggregate are under `evidence/amplitude_domain/`. The frozen q=4,8,16,32 scaling window gives exponents 1.9763 (stable) and 1.9769 (unstable); neither the 1% nor 5% median-error threshold is crossed through q=256.

Compact generic-local records are under `evidence/generic_local_dynamics/`; diagonal, spectrum, growth, and stationary-background records are under `analysis/astra_r2_node1/`, `analysis/astra_r2_node3/`, and `analysis/astra_r2_node6/`. `reproducibility/PUBLIC_RELEASE_AUTHORITY_BINDINGS.tsv` records original/public hashes for canonical handoff/evidence files.

## Finite-sample operator scope and Pythia cross-model replication

The prospective held-out-bank tests reported in the Supplemental Material have compact bank-level evidence under `evidence/heldout_transfer/`, with frozen package identities and decision endpoints recorded in `reproducibility/postpaper_protocols.tsv`. E6 tests generic-state source-operator transfer to six primary and two blind 32-sequence target banks; E6b tests transfer of the source leading singular pair to a fresh eight-bank population. Both preserve the negative scope result used in the manuscript: at this sample size, a fixed numerical mixed operator from one small empirical bank is not an unseen-bank predictor, although remeasurement on each target bank restores local closure. This negative result is not overwritten by NMIX, which uses a different estimator/subspace/sampling regime.

The minimal Pythia-70M cross-model replication has compact aligned-rate and generic one-step evidence under `evidence/pythia_crossmodel/`, with its frozen package identity and protocol endpoint recorded in `reproducibility/postpaper_protocols.tsv`. E7v2 independently remeasures the local operator at Pythia layer 3 and reproduces the six aligned growth/decay rates and the non-mode-aligned one-step closure reported in the manuscript. It does not transport the SmolLM2 numerical matrix to Pythia.

## Verification

The completion audit used Python 3.14.7, NumPy 2.5.3, and SciPy 1.18.1; the clean operator release was also tested with Python 3.12. The model-facing E5/E6/E6b/E7v2 executions used pinned Python 3.12 environments with package versions bound by their frozen packages and verified the exact model/data assets before science.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s reference_implementation/tests -v
python evidence/nmix/verify_nmix_release.py
```

For model-facing reconstruction of the established copied-state/SmolLM2/Pythia families, follow `reproducibility/UPSTREAM_RECONSTRUCTION.md`. NMIX release completeness is separately tracked in `evidence/nmix/RELEASE_COMPLETENESS.md`.

## Evidence identity and claim discipline

The growth figure preserves the 11-cell P2J primary aggregate and the separately frozen P2L confirmation; P2L is never substituted into the primary aggregate. The stationary-background common-absolute-`lambda` record labels nonzero-background values at `lambda=lambda_c(0)` as theory-only references, not measured trajectories. The diagonal-only comparison is a leading-four projected-subspace ablation, not a full-state diagonal-only reconstruction. The E5 amplitude-domain claim is a one-step locality result; it does not establish long-time accuracy at q=256. E6/E6b are scope tests of a fixed finite-sample numerical operator. E7v2 is cross-model reproduction after remeasurement. NMIX validates local mixed-block necessity at one mature checkpoint/layer/subspace/bank split and does not establish optimizer-state-aware dynamics, long-time specialization, intervention utility, universal Hessian transport, or routing-boundary theory.

Project-owned code/documentation and selection metadata are MIT licensed; upstream assets retain their original licenses.
