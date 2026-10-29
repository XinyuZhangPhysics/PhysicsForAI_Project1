# Public upstream selection metadata

This directory contains project-owned metadata needed to reconstruct the exact third-party corpus samples used by the retained manuscript experiments. It does not contain SlimPajama text.

- `S1/`: calibration/audit document manifests and eight frozen training-seed sequence manifests. Pairings are regenerated from the published bank identities with pair seeds 2026100191 and 2026100192 using the public S1 source.
- `S2/`: calibration/audit document identities, eight frozen training-seed document manifests, and `BANKS_INDEX.json`. Pairings are regenerated from the published bank identities with pair seeds 2026110291 and 2026110292 using the public S2 source.

The row identifier `train-00008:row-94542`, for example, means row 94542 of SlimPajama training parquet shard 8 at the pinned dataset revision. Published SHA-256 values verify the recovered text; S1 bank manifests additionally record the tokenized 256-token sequence hashes.

See `../UPSTREAM_RECONSTRUCTION.md` and `../verify_upstream_manifests.py`.
