# NMIX revision-4 public core

This directory binds the exact public-safe revision-4 production source and the compact numerical evidence underlying the mature OLMoE mixed-block result in the manuscript.

The source files are the executed revision-4 source, byte-identical except for the explicitly documented private-home path substitutions in `source/preflight.py` and `source/run.py`. Original and public hashes are recorded in `PUBLIC_EXPORT_MANIFEST.tsv` and `PATH_SANITIZATION.tsv`.

The public numerical record includes the deterministic Bank-A/Bank-B identities and hashes, all 72 signed Bank-B state metrics across amplitudes 0.01/0.02/0.04, the exact control-space states/responses and frozen bootstrap seed needed to reconstruct all 10,000 paired-direction bootstrap replicates, support-turnover diagnostics, domainwise summaries, the final adjudication, and exact text exports of the frozen contrast basis, direction array, and raw/symmetrized Bank-A Hessians. `ARRAY_EXPORT_METADATA.json` records the original `.npy` SHA-256 values and verifies exact numerical round-trip of each text export.

Run:

```sh
python evidence/nmix/revision4_public/verify_revision4_core.py
```

with NumPy installed. This deterministic check recomputes the published bootstrap confidence intervals and cross-checks the final adjudication from the released revision-4 records.

The 448-record raw token bank, third-party OLMoE weights/tokenizer, hidden-state/prefix acceleration caches, and large per-token routing-logit captures are not redistributed. The bank split file contains document identifiers and hashes, not corpus text or token payloads. The exact private archive is independently retained and restore-verified; these nonredistributed upstream/private inputs are not needed to reproduce the published compact statistics from the released measured records.

The full public-export candidate audited before repository synchronization has SHA-256:

`9767321bcb08442b5a3d0b8852a1e8bb7f259b0013e7ce485c1f00afae9ee754`.

Scientific scope remains unchanged: one mature OLMoE checkpoint, layer 15, one frozen 12-dimensional radial subspace, one disjoint 224/224 split, and amplitudes 0.01--0.04. Grade A passes; Grade A+ does not because the strict support-turnover thresholds fail. The fixed-support Top-k theoretical caveat therefore remains.